"""SparseWarpOperator, ComposeOperator and the mask of ExtendedPSFOperator."""
import contextlib
import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere
from neural_me.optimization import ComposeOperator, ExtendedPSFOperator, SparseWarpOperator, StrayLightOperator


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def identity_grid(h, w, input_shape=None):
    ny, nx = input_shape or (h, w)
    y = torch.linspace(-1, 1, h, dtype=torch.float64)
    x = torch.linspace(-1, 1, w, dtype=torch.float64)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([gx, gy], -1)


def stokes(n, nw=3, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(n, nw, dtype=dtype, generator=g) for _ in range(4))


def grid_sample_reference(S, grid, input_shape, valid=None):
    """The dense reference: mean of grid_sample over the K samples of each output pixel."""
    ny, nx = input_shape
    H, W, K, _ = grid.shape
    out = F.grid_sample(S.T.reshape(1, S.shape[-1], ny, nx), grid.reshape(1, H, W * K, 2).to(S.dtype), mode="bilinear",
                        padding_mode="zeros", align_corners=True)[0]         # (nw, H, W K)
    out = out.reshape(S.shape[-1], H, W, K).mean(-1)
    if valid is not None:
        out = out * valid.to(S.dtype)
    return out.reshape(S.shape[-1], H * W).T


def test_identity_grid_is_the_identity():
    op = quiet(SparseWarpOperator, identity_grid(6, 7))
    s = stokes(42)
    for a, b in zip(op(s), s):
        assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("K", [1, 4])
def test_matches_grid_sample_including_outside_the_input(dtype, K):
    """Random sub-pixel samples, some outside the input grid (padding_mode='zeros'), rectangular shapes."""
    g = torch.Generator().manual_seed(1)
    H, W, ny, nx = 5, 8, 7, 9
    grid = (torch.rand(H, W, K, 2, dtype=torch.float64, generator=g) * 2.6 - 1.3)
    valid = torch.rand(H, W, generator=g) > 0.3
    op = quiet(SparseWarpOperator, grid, valid, input_shape=(ny, nx))
    s = stokes(ny * nx, dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    for a, S in zip(op(s), s):
        assert a.dtype == dtype and a.shape == (H * W, 3)
        assert torch.allclose(a, grid_sample_reference(S, grid, (ny, nx), valid), atol=tol)


def test_adjoint_and_gradient():
    g = torch.Generator().manual_seed(2)
    grid = torch.rand(4, 5, 3, 2, dtype=torch.float64, generator=g) * 2 - 1
    op = quiet(SparseWarpOperator, grid)
    A, At = op.matrices(torch.float64, torch.device("cpu"))
    x, y = torch.randn(20, 2, dtype=torch.float64, generator=g), torch.randn(20, 2, dtype=torch.float64, generator=g)
    assert (A.matmul(x) * y).sum().item() == pytest.approx((x * At.matmul(y)).sum().item(), rel=1e-12)
    x.requires_grad_()
    assert torch.autograd.gradcheck(lambda v: op((v,))[0], (x,))


def test_invalid_pixels_are_zero():
    grid = identity_grid(4, 4)
    valid = torch.ones(4, 4, dtype=torch.bool)
    valid[0, :] = False
    out = quiet(SparseWarpOperator, grid, valid)(stokes(16))[0].reshape(4, 4, 3)
    assert (out[0] == 0).all() and (out[1:] != 0).all()


def test_constant_field_is_preserved_where_all_samples_are_inside():
    grid = identity_grid(5, 5).unsqueeze(2) * 0.8
    op = quiet(SparseWarpOperator, grid)
    out = op((torch.full((25, 2), 1.7, dtype=torch.float64),))[0]
    assert torch.allclose(out, torch.full_like(out, 1.7), atol=1e-12)


def test_from_file_reads_the_notebook_product(tmp_path):
    grid = identity_grid(4, 4)
    sub = torch.stack([grid, grid], 2).numpy().astype(np.float32)
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    path = tmp_path / "warp.npz"
    np.savez(path, grid=grid.numpy().astype(np.float32), grid_sub=sub, valid=valid)
    s = stokes(16)
    for footprint in (True, False):
        out = quiet(SparseWarpOperator.from_file, path, footprint=footprint)(s)[0]
        assert out[0].abs().max() == 0
        assert torch.allclose(out[1:], s[0][1:], atol=1e-5)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError, match="grid must"):
        SparseWarpOperator(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="valid must"):
        SparseWarpOperator(identity_grid(4, 4), torch.ones(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        ComposeOperator()


def test_extended_psf_without_mask_is_unchanged():
    """The formula of the unmasked operator, by hand."""
    atm = quiet(StereoscopicAtmosphere, grid_shape=(8, 8))
    s = stokes(64)
    op = quiet(ExtendedPSFOperator, fwhm=0.0, epsilon=0.3, tune_fwhm=False, tune_epsilon=False)
    for a, S in zip(op(s, atm), s):
        assert torch.allclose(a, 0.7 * S + 0.3 * S.mean(0, keepdim=True), atol=1e-6)


@pytest.mark.parametrize("fwhm", [0.0, 2.0])
@pytest.mark.parametrize("mean_source", ["pre_psf", "post_psf"])
def test_extended_psf_mask(fwhm, mean_source):
    atm = quiet(StereoscopicAtmosphere, grid_shape=(16, 16))
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[4:12, 3:13] = True
    m = mask.reshape(-1, 1)
    op = quiet(ExtendedPSFOperator, fwhm=fwhm, epsilon=0.2, kernel_size=9, tune_fwhm=False, tune_epsilon=False,
               mean_source=mean_source, mask=mask)
    a = stokes(256)
    out = op(a, atm)
    # outside: 0; the values there have no effect inside
    other = tuple(torch.where(m, S, S + 100.0) for S in a)
    for x, y in zip(out, op(other, atm)):
        assert (x[~m[:, 0]] == 0).all()
        assert torch.allclose(x, y, atol=1e-5)
    # a constant inside the mask stays constant (normalised convolution, mean over the mask)
    flat = tuple(torch.where(m, torch.full_like(S, 2.5), torch.full_like(S, -9.0)) for S in a)
    for x in op(flat, atm):
        assert torch.allclose(x[m[:, 0]], torch.full_like(x[m[:, 0]], 2.5), atol=1e-5)


def test_compose_applies_in_order_and_exposes_parameters_and_clamp():
    atm = quiet(StereoscopicAtmosphere, grid_shape=(4, 4))
    s = stokes(16)
    warp = quiet(SparseWarpOperator, identity_grid(4, 4).flip(1))     # mirror in x
    stray = quiet(StrayLightOperator, epsilon=0.2, tune_epsilon=True)
    op = ComposeOperator(warp, stray)
    assert [p is stray.raw_epsilon for p in op.parameters()] == [True]
    expected = stray(warp(s), atm)
    for a, b in zip(op(s, atm), expected):
        assert torch.allclose(a, b)
    with torch.no_grad():
        stray.raw_epsilon.fill_(1e3)
    op.clamp_parameters()
    assert stray.raw_epsilon.item() <= 12.0


def test_stereo_solver_trains_through_the_chain():
    """The operator's epsilon gets a gradient and the view-2 loss is computed on the warped Stokes."""
    from neural_me import MEProblem, Observation, StereoscopicNFAtmosphere
    from neural_me.optimization import StereoSolver

    rng = np.random.default_rng(3)
    wl = np.linspace(-0.3, 0.3, 8)
    d = rng.normal(1, 0.1, (4, 4, 4, 8)).astype("float32")
    torch.manual_seed(0)
    atm = quiet(StereoscopicNFAtmosphere, grid_shape=(4, 4), encoding="grid", bounds="clamp")
    p1 = quiet(MEProblem, observation=Observation(data=d, wavelengths=wl), line=6302, backend="torch")
    p2 = quiet(MEProblem, observation=Observation(data=d, wavelengths=wl), line=6173, backend="torch")
    valid = torch.ones(4, 4, dtype=torch.bool)
    valid[0, 0] = False
    psf = quiet(ExtendedPSFOperator, fwhm=1.5, epsilon=0.2, kernel_size=3, mask=valid)
    op = ComposeOperator(quiet(SparseWarpOperator, identity_grid(4, 4), valid), psf)
    solver = quiet(StereoSolver, atmosphere=atm, problem1=p1, problem2=p2, lr=1e-2, degradation_operators={2: op})
    before = [p.detach().clone() for p in op.parameters()]
    quiet(solver.solve, n_iterations=2, verbose=False)
    assert len(before) == 2
    assert all((p != b).any() for p, b in zip(op.parameters(), before))


def test_next_fast_len():
    from neural_me.optimization.degradation import _next_fast_len
    assert [_next_fast_len(n) for n in (1, 7, 526, 540, 542)] == [1, 8, 540, 540, 576]


@pytest.mark.parametrize("shape", [(31, 37), (64, 64), (5, 6)])
def test_batched_fft_equals_direct_convolution(shape):
    """The FFT branch, padded to a fast length, against conv2d, per channel and with different channel counts."""
    from neural_me.optimization import SpatialPSFOperator

    ny, nx = shape
    atm = quiet(StereoscopicAtmosphere, grid_shape=shape)
    r = torch.arange(7, dtype=torch.float64) - 3
    kernel = torch.exp(-0.5 * (r[None] ** 2 + r[:, None] ** 2) / 2.0)
    kernel = (kernel * (1 + 0.1 * r[None])).clamp_min(0)                  # asymmetric: catches a flipped kernel
    s = stokes(ny * nx, nw=3) + (torch.randn(ny * nx, 1, dtype=torch.float64),)
    fft = quiet(SpatialPSFOperator, kernel=kernel, method="fft")(s, atm)
    conv = quiet(SpatialPSFOperator, kernel=kernel, method="conv2d")(s, atm)
    for a, b, S in zip(fft, conv, s):
        assert a.shape == S.shape
        assert torch.allclose(a, b, atol=1e-10)
