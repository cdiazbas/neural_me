"""regularization.nullspace.NullSpacePenalty against dense matrices built independently, on small problems, and
against the per-map form of the inversion scripts on the PHI warp (skipped without the data)."""
import contextlib
import io
import math

import numpy as np
import pytest
import torch
from scipy.signal import convolve2d

from neural_me.optimization import SparseWarpOperator
from neural_me.regularization.nullspace import NullSpacePenalty

NY, NX = 13, 14              # instrument pixels
MY, MX = 11, 12              # model grid


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def valid_mask():
    """An off-centre ellipse: edges of the mask inside the frame, as the PHI crop."""
    y, x = np.mgrid[0:NY, 0:NX]
    return ((x - 7.5) / 6.0) ** 2 + ((y - 5.5) / 5.0) ** 2 < 1


def warp(K=4, seed=0):
    """A compressing warp with K samples per pixel, some outside the model grid."""
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(-1.1, 1.0, NY, dtype=torch.float64)
    x = torch.linspace(-0.9, 1.05, NX, dtype=torch.float64)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([gx + 0.2 * gy, 0.8 * gy], -1).unsqueeze(2) + 0.05 * torch.randn(NY, NX, K, 2, generator=g,
                                                                                      dtype=torch.float64)
    return quiet(SparseWarpOperator, grid, torch.as_tensor(valid_mask()), input_shape=(MY, MX))


def dense_conv(kernel):
    """The 'same' linear convolution with zero boundary as an (NY NX, NY NX) matrix, column by column."""
    G = np.zeros((NY * NX, NY * NX))
    for j in range(NY * NX):
        e = np.zeros(NY * NX)
        e[j] = 1.0
        G[:, j] = convolve2d(e.reshape(NY, NX), kernel, mode="same").ravel()
    return G


def gaussian_kernel(fwhm, ksize=15):
    s = fwhm / (2 * math.sqrt(2 * math.log(2)))
    r = np.arange(ksize) - ksize // 2
    g = np.exp(-r ** 2 / (2 * s ** 2))
    g /= g.sum()
    return np.outer(g, g)


def dense_K(fwhm=None, kernel=None, A=None):
    """K = P, A or P A as a dense matrix, P z = G(m z) / G(m)."""
    m = valid_mask().ravel().astype(float)
    n_model = A.shape[1] if A is not None else NY * NX
    K = A if A is not None else np.eye(n_model)
    if fwhm is None and kernel is None:
        return m[:, None] * K                       # K^T then multiplies by m, as the operator
    k = gaussian_kernel(fwhm) if kernel is None else kernel / kernel.sum()
    G = dense_conv(k)
    Gm = np.maximum(G @ m, 1e-6)
    return (G * m[None, :] / Gm[:, None]) @ K


def dense_penalty(X, w, K, mask=None):
    """sum_q w_q mean_mask (x_q - D^-1 K^T K x_q)^2, torch autograd through dense matrices."""
    K = torch.as_tensor(K)
    D = (K.T @ (K @ torch.ones(K.shape[1], dtype=K.dtype))).clamp_min(1e-6)
    if mask is None:
        mask = D > 1e-4
    R = X - (K.T @ (K @ X.T)).T / D
    return (torch.as_tensor(w) * (R[:, mask] ** 2).mean(1)).sum()


def dense_A(op):
    A, _ = op.matrices(torch.float64, torch.device("cpu"))
    return A.to_dense().numpy()


def maps(n, n_model, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, n_model, dtype=torch.float64, generator=g)


ASYM = np.exp(-((np.arange(7) - 3)[:, None] ** 2 + ((np.arange(7) - 2.5) * 1.4)[None, :] ** 2) / 3.0)


@pytest.mark.parametrize("case", ["P", "P kernel", "A", "P A"])
def test_value_and_gradient_against_dense_reference(case):
    op = warp()
    A = op if "A" in case else None
    fwhm = 2.5 if case in ("P", "P A") else None
    kernel = ASYM if case == "P kernel" else None
    if A is not None:
        pen = NullSpacePenalty.from_warp(A, valid_mask(), fwhm=fwhm, dtype=torch.float64)
        K = dense_K(fwhm, A=dense_A(A))
        mask = None
    else:
        pen = NullSpacePenalty(valid_mask(), fwhm=fwhm, kernel=kernel, dtype=torch.float64)
        K = dense_K(fwhm, kernel)
        mask = torch.as_tensor(valid_mask().ravel())
    w = [1.0, 2.0, 0.5]
    X = maps(3, pen.n_model).requires_grad_(True)
    Xd = X.detach().clone().requires_grad_(True)
    v = pen(list(X), w)
    vd = dense_penalty(Xd, w, K, mask)
    v.backward()
    vd.backward()
    assert v.item() == pytest.approx(vd.item(), rel=1e-12)
    assert torch.allclose(X.grad, Xd.grad, rtol=0, atol=1e-11 * Xd.grad.abs().max())


@pytest.mark.parametrize("case", ["P", "P kernel", "A", "P A"])
def test_gradcheck(case):
    op = warp()
    fwhm = 2.0 if case in ("P", "P A") else None
    kernel = ASYM if case == "P kernel" else None
    if "A" in case:
        pen = NullSpacePenalty.from_warp(op, valid_mask(), fwhm=fwhm, dtype=torch.float64)
    else:
        pen = NullSpacePenalty(valid_mask(), fwhm=fwhm, kernel=kernel, dtype=torch.float64)
    X = maps(2, pen.n_model).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: pen(list(x), [1.0, 3.0]), (X,))


@pytest.mark.parametrize("case", ["P", "P kernel", "A", "P A"])
def test_constant_maps_cost_nothing(case):
    """B K 1 = 1 by construction (D = K^T K 1): a constant is rebuilt to rounding.

    The measured-kernel path is the loosest in float32 (up to 2e-4 of the constant): the FFT leaves ~1e-8 noise
    where G(m) is 0, and the clamp at 1e-6 and the division in P^T amplify it. The Gaussian path is direct and
    exact there."""
    fwhm = 2.0 if case in ("P", "P A") else None
    kernel = ASYM if case == "P kernel" else None
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-3 if kernel is not None else 1e-6)):
        if "A" in case:
            pen = NullSpacePenalty.from_warp(warp(), valid_mask(), fwhm=fwhm, dtype=dtype)
        else:
            pen = NullSpacePenalty(valid_mask(), fwhm=fwhm, kernel=kernel, dtype=dtype)
        X = torch.tensor([1.0, -3.5, 40.0], dtype=dtype)[:, None].expand(3, pen.n_model)
        R = pen.residual(X)[:, pen.mask]
        assert (R.abs().max(1).values / X[:, 0].abs()).max().item() < tol
        assert pen(list(X)).item() < (tol * 40.0) ** 2
        assert pen(maps(1, pen.n_model).to(dtype)).item() > 1e-3


def test_nothing_erased_costs_nothing():
    """K = A with A a permutation of the model pixels (one sample per pixel at a pixel centre): K^T K = I, so
    every map is rebuilt and the penalty is 0 whatever the map."""
    y = torch.linspace(-1, 1, NY, dtype=torch.float64)
    x = torch.linspace(-1, 1, NX, dtype=torch.float64)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    op = quiet(SparseWarpOperator, torch.stack([gx, gy], -1).flip(0))        # mirror in y
    pen = NullSpacePenalty.from_warp(op, dtype=torch.float64)
    assert pen.mask.all()
    assert pen(list(maps(4, NY * NX))).item() < 1e-28


def test_smooth_maps_cost_less_than_noise():
    """The PSF null space is the small scales: a map smooth on the PSF scale costs < 1e-2 of white noise (9e-4
    measured)."""
    pen = NullSpacePenalty(valid_mask(), fwhm=2.0, dtype=torch.float64)
    y, x = np.mgrid[0:NY, 0:NX]
    smooth = torch.as_tensor(np.cos(2 * np.pi * x / 40.0) + 0.5 * np.sin(2 * np.pi * y / 50.0)).reshape(-1)
    assert pen([smooth]).item() < 1e-2 * pen(list(maps(1, NY * NX))).item()


def test_from_warp_equals_the_matrix_and_derives_valid():
    op = warp()
    A = torch.sparse_coo_tensor(op._indices, op._values, (NY * NX, MY * MX))
    X = list(maps(2, MY * MX).float())
    rows = torch.as_tensor(np.abs(dense_A(op)).sum(1) > 0).reshape(NY, NX)       # the pixels with a sample on the grid
    assert rows.sum() > 0.8 * valid_mask().sum()
    for fwhm in (None, 2.0):
        a = NullSpacePenalty(valid_mask(), fwhm=fwhm, A=A)(X, [1.0, 2.0])
        b = NullSpacePenalty.from_warp(op, valid_mask(), fwhm=fwhm)(X, [1.0, 2.0])
        c = NullSpacePenalty(rows, fwhm=fwhm, A=A)(X, [1.0, 2.0])
        d = NullSpacePenalty.from_warp(op, fwhm=fwhm)(X, [1.0, 2.0])
        assert torch.equal(a, b) and torch.equal(c, d)
    assert torch.equal(NullSpacePenalty.from_warp(op).m.reshape(NY, NX).bool(), rows)


def test_psf_on_warped_maps():
    """K = P on the instrument pixels applied to A x (the warped maps), gradient through the operator's A^T."""
    op = warp()
    pen = NullSpacePenalty(valid_mask(), fwhm=2.0, model_mask=valid_mask(), dtype=torch.float64)
    X = maps(2, MY * MX).requires_grad_(True)
    Xd = X.detach().clone().requires_grad_(True)
    v = pen(list(op((X.T,))[0].T))
    m = torch.as_tensor(valid_mask().ravel())
    vd = dense_penalty((torch.as_tensor(dense_A(op)) @ Xd.T).T, [1.0, 1.0], dense_K(2.0), m)
    v.backward()
    vd.backward()
    assert v.item() == pytest.approx(vd.item(), rel=1e-12)
    assert torch.allclose(X.grad, Xd.grad, rtol=0, atol=1e-11 * Xd.grad.abs().max())


def test_rejects_bad_shapes():
    op = warp()
    with pytest.raises(ValueError, match="rows"):
        NullSpacePenalty(np.ones((5, 5), bool), A=torch.sparse_coo_tensor(op._indices, op._values, (NY * NX, MY * MX)))
    with pytest.raises(ValueError, match="model_mask"):
        NullSpacePenalty(valid_mask(), fwhm=2.0, model_mask=np.ones(7, bool))
    pen = NullSpacePenalty(valid_mask(), fwhm=2.0)
    with pytest.raises(ValueError, match="elements"):
        pen([torch.zeros(7)])


@pytest.mark.parametrize("case", ["P", "A", "P A"])
def test_real_warp_against_the_per_map_version(case, data_path):
    """On the PHI warp (512 x 512 -> 512 x 512): the batched penalty against the per-map form first written in the
    inversion scripts (15 x 15 conv, one map at a time, COO sparse), float32 as there."""
    import torch.nn.functional as F

    npz = data_path("phi_solo_data/phi_native_warp.npz")
    valid = np.load(npz)["valid"].astype(bool)
    ny, nx = valid.shape
    op = quiet(SparseWarpOperator.from_file, npz)
    fwhm = 2.0 if "P" in case else None
    pen = NullSpacePenalty(valid, fwhm=fwhm) if case == "P" else NullSpacePenalty.from_warp(op, fwhm=fwhm)
    A = torch.sparse_coo_tensor(op._indices, op._values.float(), (ny * nx, ny * nx)).coalesce()
    m = torch.from_numpy(valid.astype(np.float32)).view(1, 1, ny, nx)
    GK = torch.from_numpy(gaussian_kernel(2.0).astype(np.float32)).view(1, 1, 15, 15)
    Gc = lambda z: F.conv2d(z, GK, padding=7)
    Gm = Gc(m).clamp_min(1e-6)
    P = lambda z: Gc(m * z) / Gm
    Pt = lambda y: m * Gc(y / Gm)
    Asp = lambda z: torch.sparse.mm(A, z.reshape(-1, 1)).view(1, 1, ny, nx)
    Atsp = lambda y: torch.sparse.mm(A.t(), y.reshape(-1, 1)).view(1, 1, ny, nx)
    K, Kt = {"P": (P, Pt), "A": (Asp, lambda y: Atsp(m * y)), "P A": (lambda z: P(Asp(z)), lambda y: Atsp(Pt(y)))}[case]
    KtK1 = Kt(K(torch.ones(1, 1, ny, nx))).clamp_min(1e-6)
    mask = torch.from_numpy(valid) if case == "P" else KtK1[0, 0] > 1e-4
    g = torch.Generator().manual_seed(0)
    base = [F.avg_pool2d(torch.randn(1, 1, ny + 4, nx + 4, generator=g), 5, 1)[0, 0] for _ in range(3)]
    w = [1.0, 2.0, 1.0]
    X = [b.clone().requires_grad_(True) for b in base]
    Y = [b.clone().requires_grad_(True) for b in base]
    v = pen(X, w)
    vr = sum(wq * ((y.view(1, 1, ny, nx) - Kt(K(y.view(1, 1, ny, nx))) / KtK1)[0, 0][mask] ** 2).mean()
             for wq, y in zip(w, Y))
    v.backward()
    vr.backward()
    g1, g0 = torch.stack([x.grad for x in X]), torch.stack([y.grad for y in Y])
    assert v.item() == pytest.approx(vr.item(), rel=1e-5)
    assert ((g1 - g0).abs().max() / g0.abs().max()).item() < 1e-4
