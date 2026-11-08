"""Multi-resolution grid encoding of the neural field (nf.GridField, NFAtmosphere(encoding="grid"))."""
import contextlib
import io

import pytest
import torch
import torch.nn.functional as F

from neural_me.core.atmospheres import NFAtmosphere
from neural_me.core.nf import GridField, to_coordinates

INIT = {"vlos": 0.21, "eta0": 12.9, "a": 0.23, "ddop": 0.037, "S0": 0.31}


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def test_upsampling_equals_sampling_at_the_nf_coordinates():
    """forward() must follow the row-major (y, x) order of to_coordinates."""
    torch.manual_seed(0)
    shape = (12, 17)
    f = GridField(shape, dim_out=1, n_levels=3, base_res=3, features_per_level=1, dim_hidden=4)
    g = f.levels[0][1]                                        # one (1, 1, ry, rx) level
    up = F.interpolate(g, size=shape, mode="bilinear", align_corners=True).reshape(-1)
    yx = to_coordinates(shape)                                # (N, 2) as (y, x)
    smp = F.grid_sample(g, yx[:, [1, 0]].reshape(1, 1, -1, 2), mode="bilinear", align_corners=True).reshape(-1)
    assert torch.allclose(up, smp, atol=1e-6)


def test_groups_must_cover_every_output_once():
    with pytest.raises(ValueError, match="cover channels"):
        GridField((8, 8), dim_out=3, groups=[((0, 1), 8), ((1,), 4)])


def test_grid_nf_starts_exactly_at_init_with_a_weak_uniform_field():
    torch.manual_seed(0)
    atm = quiet(NFAtmosphere, grid_shape=(32, 24), encoding="grid", init=INIT, bounds="clamp")
    st = atm.to_stack().detach()
    for k, i in (("vlos", 3), ("eta0", 4), ("a", 5), ("ddop", 6), ("S0", 7)):
        assert torch.allclose(st[..., i], torch.full_like(st[..., i], INIT[k]), rtol=1e-5)
    b = st[..., 0]
    assert float(b.min()) > 1.0                       # B != 0, so Q/U have a gradient
    assert float(b.std() / b.mean()) < 0.05           # and it is nearly uniform


def test_transverse_field_receives_gradient():
    torch.manual_seed(0)
    atm = quiet(NFAtmosphere, grid_shape=(16, 16), encoding="grid", init=INIT, bounds="clamp")
    bperp2 = (atm.to_stack()[..., 0] * torch.sin(atm.to_stack()[..., 1])) ** 2
    bperp2.mean().backward()
    head_last = [h[-1].weight.grad for h in atm.NF.heads]
    assert head_last[0][1].abs().sum() > 0 and head_last[0][2].abs().sum() > 0


def test_masked_grid_nf_returns_only_masked_pixels():
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[2:7, 3:9] = True
    atm = quiet(NFAtmosphere, grid_shape=(10, 10), mask=mask, encoding="grid")
    assert atm.to_stack().shape == (int(mask.sum()), 9)


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="encoding"):
        quiet(NFAtmosphere, grid_shape=(8, 8), encoding="hash")


@pytest.mark.parametrize("encoding", ["fourier", "grid"])
def test_source_continuum_reproduces_the_observed_continuum(encoding):
    torch.manual_seed(0)
    ic = torch.rand(12, 10) * 0.8 + 0.2                       # umbra-like 0.2 .. quiet 1.0
    atm = quiet(NFAtmosphere, grid_shape=(12, 10), encoding=encoding, m_freqs=8, dim_hidden=8,
                num_resnet_blocks=1, source="continuum", continuum=ic, init={"S0": 0.3}, bounds="clamp")
    st = atm.to_stack().detach()
    assert torch.allclose(st[..., 7] + st[..., 8], ic, rtol=1e-5)
    if encoding == "grid":                                    # exact start: S0 = 0.3 Ic
        assert torch.allclose(st[..., 7], 0.3 * ic, rtol=1e-4)


@pytest.mark.parametrize("encoding", ["fourier", "grid"])
def test_source_free_has_independent_s1(encoding):
    torch.manual_seed(0)
    atm = quiet(NFAtmosphere, grid_shape=(8, 8), encoding=encoding, m_freqs=8, dim_hidden=8,
                num_resnet_blocks=1, sigma=[1.0] * 8, source="free",
                init={"S0": 0.3, "S1": 0.74}, bounds="clamp")
    assert atm.NF(atm.coords).shape[-1] == 9 if encoding == "fourier" else atm.NF().shape[-1] == 9
    st = atm.to_stack().detach()
    if encoding == "grid":
        assert torch.allclose(st[..., 7], torch.full((8, 8), 0.3), rtol=1e-4)
        assert torch.allclose(st[..., 8], torch.full((8, 8), 0.74), rtol=1e-4)
    assert not torch.allclose(st[..., 7] + st[..., 8], torch.ones(8, 8))


def test_source_arguments_are_checked():
    with pytest.raises(ValueError, match="continuum map"):
        quiet(NFAtmosphere, grid_shape=(8, 8), encoding="grid", source="continuum")
    with pytest.raises(ValueError, match="continuum map"):
        quiet(NFAtmosphere, grid_shape=(8, 8), encoding="grid", continuum=torch.ones(8, 8))
    with pytest.raises(ValueError, match="unknown init keys"):
        quiet(NFAtmosphere, grid_shape=(8, 8), encoding="grid", source="continuum",
              continuum=torch.ones(8, 8), init={"S1": 0.7})        # S1 is not fitted with "continuum"
    with pytest.raises(ValueError, match="source"):
        quiet(NFAtmosphere, grid_shape=(8, 8), encoding="grid", source="unit")   # removed


@pytest.mark.parametrize("source", ["free", "continuum"])
def test_stereo_grid_nf_starts_at_the_per_view_init(source):
    from neural_me.core.atmospheres import StereoscopicNFAtmosphere
    torch.manual_seed(0)
    ic = (torch.rand(10, 12) + 0.2, torch.rand(10, 12) + 0.2) if source == "continuum" else None
    init = {"thermo1_eta0": 12.9, "thermo2_eta0": 185.0, "thermo1_S0": 0.3, "thermo2_S0": 0.6}
    if source == "free":
        init.update(thermo1_S1=0.74, thermo2_S1=0.33)
    atm = quiet(StereoscopicNFAtmosphere, grid_shape=(10, 12), encoding="grid", source=source,
                continuum=ic, init=init, bounds="clamp")
    assert atm.NF().shape == (120, 15 if source == "free" else 13)
    v1, v2 = (atm.get_atmosphere(i).to_stack().detach() for i in (1, 2))
    assert torch.allclose(v1[..., 4], torch.full((10, 12), 12.9), rtol=1e-4)
    assert torch.allclose(v2[..., 4], torch.full((10, 12), 185.0), rtol=1e-4)
    if source == "free":
        assert torch.allclose(v2[..., 8], torch.full((10, 12), 0.33), rtol=1e-4)
    else:
        assert torch.allclose(v2[..., 7] + v2[..., 8], ic[1], rtol=1e-5)
    assert float(v1[..., 0].min()) > 1.0 and torch.allclose(v1[..., 0], v2[..., 0], rtol=1e-4)  # one global field


def test_stereo_solver_dynamic_balance_follows_the_log_losses():
    """balance='dynamic': gradient of (1-a) log L1 + a log L2, so scaling one view's chi2 changes nothing."""
    import numpy as np
    from neural_me import Observation, MEProblem, StereoscopicNFAtmosphere
    from neural_me.optimization import StereoSolver

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    wl = np.linspace(-0.3, 0.3, 8)
    grads = []
    for scale in (1.0, 100.0):
        torch.manual_seed(0)
        atm = quiet(StereoscopicNFAtmosphere, grid_shape=(4, 4), encoding="grid", bounds="clamp")
        d1 = rng.normal(1, 0.1, (4, 4, 4, 8)).astype("float32")
        d2 = rng.normal(1, 0.1, (4, 4, 4, 8)).astype("float32")
        p1 = quiet(MEProblem, observation=Observation(data=d1, wavelengths=wl), line=6302, backend="torch")
        p2 = quiet(MEProblem, observation=Observation(data=d2, wavelengths=wl), line=6173, backend="torch",
                   weights=[scale] * 4)
        solver = quiet(StereoSolver, atmosphere=atm, problem1=p1, problem2=p2, lr=0.0, balance="dynamic")
        quiet(solver.solve, n_iterations=1, verbose=False)
        grads.append(torch.cat([p.grad.flatten() for p in atm.parameters() if p.grad is not None]))
        rng = np.random.default_rng(0)
    assert torch.allclose(grads[0], grads[1], rtol=1e-3, atol=1e-8)
    with pytest.raises(ValueError, match="balance"):
        quiet(StereoSolver, atmosphere=atm, problem1=p1, problem2=p2, balance="auto")


def test_stereo_solver_warmup_scales_the_first_steps():
    import numpy as np
    from neural_me import Observation, MEProblem, StereoscopicNFAtmosphere
    from neural_me.optimization import StereoSolver

    rng = np.random.default_rng(1)
    wl = np.linspace(-0.3, 0.3, 8)
    d = rng.normal(1, 0.1, (4, 4, 4, 8)).astype("float32")
    moved = []
    for warmup in (0, 10):
        torch.manual_seed(0)
        atm = quiet(StereoscopicNFAtmosphere, grid_shape=(4, 4), encoding="grid", bounds="clamp")
        p1 = quiet(MEProblem, observation=Observation(data=d, wavelengths=wl), line=6302, backend="torch")
        p2 = quiet(MEProblem, observation=Observation(data=d, wavelengths=wl), line=6173, backend="torch")
        before = [p.detach().clone() for p in atm.parameters()]
        solver = quiet(StereoSolver, atmosphere=atm, problem1=p1, problem2=p2, lr=1e-2)
        quiet(solver.solve, n_iterations=1, verbose=False, warmup=warmup)
        moved.append(max(float((p - b).abs().max()) for p, b in zip(atm.parameters(), before)))
        assert solver.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2 if warmup == 0 else 1e-3)
    assert moved[1] == pytest.approx(moved[0] / 10, rel=0.05)
