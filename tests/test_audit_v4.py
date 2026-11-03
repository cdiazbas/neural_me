"""Open findings of audit neural_me v4, one strict xfail per finding.

Each test asserts the *correct* behaviour, so it fails today.  When the
defect is fixed the test passes, pytest reports XPASS as a failure, and the
marker must be removed in the same change.  Only an AssertionError counts
as the defect (``raises=AssertionError``): any other exception fails the run,
because then the test is broken, not the code.  Finding IDs refer to
``audits/neural_me_v4_AUDIT.md`` in the NEURALME project folder.
"""

import ast
import contextlib
import inspect
import io
import re
import textwrap
import types

import numpy as np
import pytest
import torch


def xfail(finding_id: str, what: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"{finding_id}: {what}")


def code_only(obj) -> str:
    """Source of ``obj`` without comments or docstrings."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0] = ast.Pass()
    return ast.unparse(tree)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _me_profile(line=6173, B=200.0, gamma_deg=60.0, xi_deg=30.0):
    from neural_me.physics.milne_eddington import MilneEddington

    me = quiet(MilneEddington, line=line)
    wl = torch.linspace(-0.6, 0.6, 241, dtype=torch.float64)
    p = torch.tensor([[B, np.deg2rad(gamma_deg), np.deg2rad(xi_deg),
                       0.0, 10.0, 0.2, 0.035, 0.2, 0.8]], dtype=torch.float64)
    stokes = torch.stack(me(p, wl), 1).detach().numpy()
    return wl.numpy(), stokes


# --------------------------------------------------------------------- #
#  High
# --------------------------------------------------------------------- #

def test_c01_wfa_recovers_transverse_field():
    """C01 (fixed): the WFA must recover BtQ, BtU of a weak-field profile."""
    from neural_me.physics.wfa import wfa_initial_conditions

    wl, st = _me_profile()
    _, btq, btu = (float(np.ravel(x)[0]) for x in wfa_initial_conditions(wl, st, line_id=7)[:3])
    assert btq == pytest.approx(86.6, rel=0.2)
    assert btu == pytest.approx(150.0, rel=0.2)


@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
def test_c02_tps_identity_warp(padding_mode):
    """C02 (fixed): identical control points must give the identity."""
    from neural_me.core.warp import differentiable_tps_warp, get_points_src

    torch.manual_seed(0)
    img = torch.rand(1, 1, 32, 40)
    p = get_points_src(9)
    out = differentiable_tps_warp(img, p, p.clone(), padding_mode=padding_mode)
    assert torch.allclose(out, img, atol=1e-4)


@pytest.mark.parametrize("dx_px", [3, -5])
def test_c02_tps_integer_shift_is_exact(dx_px):
    """C02 (fixed): shifting every control point by k pixels moves features by k pixels."""
    from neural_me.core.warp import differentiable_tps_warp, get_points_src

    H, W = 41, 61
    spot = torch.zeros(1, 1, H, W)
    spot[..., 5, 50] = 1.0  # off-centre: a scale error shows up here, not at the centre
    p = get_points_src(9)
    d = p.clone()
    d[..., 0] += dx_px * 2.0 / (W - 1)
    out = differentiable_tps_warp(spot, p, d, padding_mode="zeros")[0, 0]
    assert divmod(int(out.argmax()), W) == (5, 50 + dx_px)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-4)


def test_c03_pixel_solver_wfa_init_moves_B():
    """C03 (fixed): WFA init must set B on the default spherical atmosphere."""
    from neural_me.core import MEProblem
    from neural_me.core.observation import Observation
    from neural_me.optimization import PixelSolver

    ny, nx, nw = 2, 2, 30
    wl = torch.linspace(-0.4, 0.4, nw)
    prob0 = quiet(MEProblem, Observation(torch.zeros(ny, nx, 4, nw), wl),
                  line=6302, backend="torch", compile=False)
    p = torch.tensor([[1500.0, 0.5, 0.6, 0.3, 10.0, 0.2, 0.03, 0.2, 0.8]]).repeat(ny * nx, 1)
    I, Q, U, V = prob0.model(p, wl.expand(ny * nx, -1))
    obs = Observation(torch.stack([I, Q, U, V], 1).reshape(ny, nx, 4, nw).detach(), wl)
    ps = quiet(PixelSolver, quiet(MEProblem, obs, line=6302, backend="torch", compile=False))
    b0 = ps.atmosphere.B.detach().clone()
    quiet(ps.init_from_wfa)
    assert not torch.allclose(ps.atmosphere.B.detach(), b0)


def test_c04_levmar_per_pixel_lambda_beats_global():
    """C04 (fixed): per-pixel lambda/accept must fit a heterogeneous map better than one global lambda."""
    from neural_me.core import MEProblem
    from neural_me.core.observation import Observation
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    n, nw = 8, 40
    g = torch.Generator().manual_seed(0)
    wl = torch.linspace(-0.4, 0.4, nw)
    lo = torch.tensor([100.0, 0.2, 0.1, -2.0, 3.0, 0.05, 0.02, 0.1, 0.5])
    hi = torch.tensor([2000.0, 2.9, 3.0, 2.0, 40.0, 0.6, 0.045, 0.3, 0.9])
    truth = lo + (hi - lo) * torch.rand(n * n, 9, generator=g)
    p0 = quiet(MEProblem, Observation(torch.zeros(n, n, 4, nw), wl),
               line=6302, backend="torch", compile=False)
    data = torch.stack(p0.model(truth, wl.expand(n * n, -1)), 1).reshape(n, n, 4, nw).detach()
    data = data + 1e-3 * torch.randn(data.shape, generator=g)

    chi2 = {}
    for per_pixel in (False, True):
        prob = quiet(MEProblem, Observation(data, wl), line=6302, backend="torch", compile=False)
        atm = quiet(SphericalSoftplusAtmosphere, grid_shape=(n, n))
        solver = quiet(LevMarSolver, prob, atmosphere=atm)
        solver.set_weights(1e-3, 1e-3, 1e-3, 1e-3)
        chi2[per_pixel] = quiet(solver.solve, max_iter=60, verbose=False, per_pixel=per_pixel)["chi2"]
    assert chi2[True] < 0.5 * chi2[False]


# --------------------------------------------------------------------- #
#  Medium
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("crota2", [30.0, -12.0, 0.0, 3.8])
def test_c05_wcs_roll_angle_is_peff(crota2):
    """C05 (fixed): wcs_roll_angle equals ISPy remap2cea's peff = -CROTA2."""
    pytest.importorskip("sunpy")
    import sunpy.map
    from astropy.io.fits import Header
    from neural_me.core.geometry import wcs_roll_angle

    h = Header()
    for k, v in dict(NAXIS=2, NAXIS1=10, NAXIS2=10, CRPIX1=5, CRPIX2=5, CRVAL1=0.0,
                     CRVAL2=0.0, CDELT1=1.0, CDELT2=1.0, CTYPE1="HPLN-TAN",
                     CTYPE2="HPLT-TAN", CUNIT1="arcsec", CUNIT2="arcsec",
                     CROTA2=crota2).items():
        h[k] = v
    h["DATE-OBS"] = "2022-03-31T19:39:59"
    smap = sunpy.map.Map(np.zeros((10, 10)), h)
    peff = -1.0 * crota2 * np.pi / 180.0  # ISPy.util.remap.remap2cea
    assert wcs_roll_angle(smap) == pytest.approx(peff, abs=1e-12)


@xfail("C06", "StereoSolver loss uses compute_loss_from_syn only, no regularization")
def test_c06_stereo_solver_regularizes():
    from neural_me.optimization import stereo_solver

    assert "regularization" in code_only(stereo_solver.StereoSolver)


@xfail("C07", "regularization in raw physical units; normalization dict unused")
def test_c07_regularization_is_normalized():
    from neural_me.core.problem import MEProblem

    assert "normalization" in code_only(MEProblem._compute_regularization)


def test_c08_nf_network_runs_once_per_stack():
    """C08 (fixed): to_stack must evaluate the network once."""
    from neural_me.core.atmospheres import NFAtmosphere

    atm = quiet(NFAtmosphere, grid_shape=(4, 4), m_freqs=16, dim_hidden=8, num_resnet_blocks=1)
    calls = {"n": 0}
    orig = atm._evaluate_nf

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    atm._evaluate_nf = counted
    atm.to_stack()
    assert calls["n"] == 1


def test_a01_view_magnetic_computed_once():
    """A01 (fixed): a view rotates the global field once per to_stack()."""
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere, ViewAtmosphere

    atm = quiet(StereoscopicAtmosphere, grid_shape=(4, 4))
    calls = {"n": 0}
    orig = ViewAtmosphere._compute_magnetic

    def counted(self):
        calls["n"] += 1
        return orig(self)

    ViewAtmosphere._compute_magnetic = counted
    try:
        atm.get_atmosphere(2).to_stack()
    finally:
        ViewAtmosphere._compute_magnetic = orig
    assert calls["n"] == 1


def test_c10_voigt_keeps_float64():
    """C10 (fixed): float64 inputs give float64 Voigt profiles."""
    from neural_me.physics.zeeman import fvoigt

    H, F = fvoigt(torch.tensor([0.1], dtype=torch.float64),
                  torch.linspace(-2, 2, 5, dtype=torch.float64))
    assert H.dtype == torch.float64 and F.dtype == torch.float64


def test_c11_psf_paths_agree_for_asymmetric_kernel():
    """C11 (fixed): conv2d and FFT paths both convolve."""
    from neural_me.optimization.degradation import SpatialPSFOperator

    atm = types.SimpleNamespace(grid_shape=(16, 20))
    torch.manual_seed(0)
    k = torch.rand(5, 5)
    k[0, 0] = 5
    S = torch.rand(16 * 20, 3)
    a = quiet(SpatialPSFOperator, k, method="fft", verbose=False)((S,) * 4, atm)[0]
    b = quiet(SpatialPSFOperator, k, method="conv2d", verbose=False)((S,) * 4, atm)[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_c12_spectral_psf_preserves_flat_continuum():
    """C12 (fixed): the spectral PSF keeps a flat continuum flat to the band edges."""
    from neural_me.optimization.degradation import SpectralPSFOperator

    atm = types.SimpleNamespace(grid_shape=(2,))
    I = torch.ones(2, 10)
    out = quiet(SpectralPSFOperator, torch.tensor([1.0, 2, 3, 2, 1]))((I,) * 4, atm)[0]
    assert torch.allclose(out, I, atol=1e-6)


def test_c13_cpp_unsupported_line_raises_clearly():
    """C13 (crash, fixed): an unsupported line raises ValueError, not KeyError from pyMilne."""
    from neural_me.physics.milne_eddington import MilneEddington

    wl = torch.linspace(-0.5, 0.5, 20)
    with pytest.raises(ValueError, match="cpp backend implements lines"):
        me = quiet(MilneEddington, line=15648, backend="cpp", lambda_axis=wl)
        quiet(me, torch.tensor([[500.0, 1.0, 0.5, 0.0, 10.0, 0.2, 0.03, 0.2, 0.8]]), wl)


@xfail("C13", "cpp backend uses its own line table (cw, g) instead of LINEAS")
def test_c13_cpp_line_table_consistent():
    from neural_me.physics.milne_eddington_cpp import MilneEddington as cppmod

    assert "6173.334" not in code_only(cppmod).replace(" ", "")


def test_c14_torch_backend_rejects_mu():
    """C14 (fixed): the torch backend refuses mu != 1 instead of ignoring it."""
    from neural_me.physics.milne_eddington import MilneEddington

    with pytest.raises(ValueError, match="mu = 1"):
        quiet(MilneEddington, line=6302, backend="torch", mu=0.5)
    quiet(MilneEddington, line=6302, backend="torch", mu=1.0)


def test_c15_local_vertical_state_dict_round_trip():
    """C15 (fixed): a checkpoint with geometry loads into a fresh atmosphere, values intact."""
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

    a = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    rot = np.broadcast_to(np.diag([1.0, -1.0, 1.0]), (2, 2, 3, 3)).copy()
    mask = np.array([[True, False], [True, True]])
    a.set_local_vertical_geometry(rot, mask)
    b = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    b.load_state_dict(a.state_dict())
    assert b.has_local_vertical_geometry
    assert torch.equal(b.lv_rotation, a.lv_rotation)
    assert torch.equal(b.lv_on_disk, a.lv_on_disk)
    # a checkpoint without geometry still loads strictly into a fresh atmosphere
    c = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    c.load_state_dict(quiet(StereoscopicAtmosphere, grid_shape=(2, 2)).state_dict())
    assert not c.has_local_vertical_geometry


def test_c16_extended_psf_doc_matches_default():
    """C16 (fixed): the docstring names the actual default mean_source."""
    from neural_me.optimization.degradation import ExtendedPSFOperator

    default = inspect.signature(ExtendedPSFOperator.__init__).parameters["mean_source"].default
    assert f"``mean_source='{default}'`` (default)" in (ExtendedPSFOperator.__doc__ or "")


def test_c32_docstring_drift_quotes_gone():
    """C32 (fixed): the wrong statements are no longer in the docstrings."""
    from neural_me.optimization import levmar_solver, stereo_solver
    from neural_me.physics import zeeman

    assert "(0 or 1)" not in inspect.getsource(stereo_solver)
    assert "(J^T J + (1 + lambda) * diag(J^T J))" not in inspect.getsource(levmar_solver)
    assert "More accurate than Hui" not in inspect.getsource(zeeman)
    assert "less accurate than Humlicek" not in inspect.getsource(zeeman)


@xfail("C17", "2D Lorentzian kernel shape depends on kernel_size")
def test_c17_lorentzian_kernel_independent_of_support():
    from neural_me.optimization.degradation import PSFKernel

    def core_energy(size):
        k = quiet(PSFKernel, model="lorentzian", fwhm=2.0, kernel_size=size, tune_fwhm=False)
        kern = k().detach().squeeze()
        c = size // 2
        yy, xx = np.mgrid[:size, :size]
        return float(kern[torch.from_numpy((yy - c) ** 2 + (xx - c) ** 2 <= 1)].sum())

    assert core_energy(31) == pytest.approx(core_energy(101), rel=0.05)


@xfail("C18", "COG Btrans factor 1.5 underestimates Bt by ~30 %")
def test_c18_cog_recovers_transverse_field():
    from neural_me.physics.cog import cog_initial_conditions

    wl, st = _me_profile()
    _, btq, btu = (float(np.ravel(x)[0]) for x in cog_initial_conditions(wl, st, line_id=7)[:3])
    assert np.hypot(btq, btu) == pytest.approx(np.hypot(86.6, 150.0), rel=0.15)


# --------------------------------------------------------------------- #
#  Low
# --------------------------------------------------------------------- #

def test_c19_stereo_clamp_uses_storage_normalization():
    """C19 (fixed): a non-default normalisation does not move the physical clamp bound."""
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

    atm = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    atm._normalization["eta0"] = 10.0
    atm.set_parameters({"thermo1_eta0": np.full((2, 2), 500.0)})
    atm.clamp_parameters()
    assert float(atm.get_atmosphere(1).eta0.mean()) == pytest.approx(500.0, rel=1e-3)


@xfail("C20", "NF atmosphere S1 = 1 - S0 ignores norm['S1'] and goes negative")
def test_c20_nf_source_function_positive():
    from neural_me.core.atmospheres import NFAtmosphere

    atm = quiet(NFAtmosphere, grid_shape=(4, 4), m_freqs=16, dim_hidden=8,
                num_resnet_blocks=1, norm={"S0": 3.0})
    with torch.no_grad():
        for p in atm.parameters():
            p.fill_(0.5)
    assert float(atm.S1.min()) >= 0.0


def test_c21_stereo_angles_are_non_persistent_buffers():
    """C21 (fixed): angles follow .to() but stay out of the state_dict (old checkpoints load)."""
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

    atm = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    assert "gamma_rad" in dict(atm.named_buffers())
    assert not any(k.endswith("_rad") for k in atm.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_c21_stereo_atmospheres_follow_to_cuda():
    """C21 (fixed): after .to('cuda') device, angles, views and new geometry are on CUDA."""
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere
    from neural_me.core.atmospheres.nf_stereoscopic import StereoscopicNFAtmosphere

    a = quiet(StereoscopicAtmosphere, grid_shape=(4, 4)).to("cuda")
    n = quiet(StereoscopicNFAtmosphere, grid_shape=(4, 4), m_freqs=16, dim_hidden=8,
              num_resnet_blocks=1).to("cuda")
    for m in (a, n):
        assert m.device.type == "cuda" and m.gamma_rad.device.type == "cuda"
        assert m.get_atmosphere(2).device.type == "cuda"
    a.set_local_vertical_geometry(np.broadcast_to(np.eye(3), (4, 4, 3, 3)).copy())
    assert a.lv_rotation.device.type == "cuda"


@xfail("C22", "atan2(BU, BQ + 1e-15) has a ~1e17 gradient at zero transverse field")
def test_c22_cartesian_xi_gradient_finite_scale():
    from neural_me.core.atmospheres import CartesianAtmosphere

    atm = quiet(CartesianAtmosphere, grid_shape=(2,), init_type="zeros")
    atm.xi.sum().backward()
    g = max(float(p.grad.abs().max()) for p in atm.parameters() if p.grad is not None)
    assert g < 1e6


def test_c23_sliced_atmosphere_flat_indices():
    """C23 (fixed): slicing a 2-D atmosphere picks pixels, matching Observation's flat layout."""
    from neural_me.core.atmospheres import SphericalAtmosphere

    atm = quiet(SphericalAtmosphere, grid_shape=(8, 8))
    idx = torch.tensor([0, 9, 63])
    assert torch.equal(atm[idx].B, atm.B.reshape(-1)[idx])


def test_c23_compute_loss_with_indices_on_2d_grid():
    """C23 (fixed): MEProblem.compute_loss(indices=...) works on a 2-D grid."""
    from neural_me.core import MEProblem
    from neural_me.core.observation import Observation
    from neural_me.core.atmospheres import SphericalAtmosphere

    wl = torch.linspace(-0.4, 0.4, 20)
    prob = quiet(MEProblem, Observation(torch.rand(4, 4, 4, 20), wl), line=6302,
                 backend="torch", compile=False)
    atm = quiet(SphericalAtmosphere, grid_shape=(4, 4))
    loss = prob.compute_loss(atm, indices=torch.tensor([0, 5, 15]))
    assert torch.isfinite(loss)


# --------------------------------------------------------------------- #
#  Found by the performance pass
# --------------------------------------------------------------------- #

def _torch_problem(precision="float32"):
    from neural_me.core import MEProblem
    from neural_me.core.observation import Observation

    wl = torch.linspace(-0.4, 0.4, 30)
    obs = Observation(torch.zeros(2, 2, 4, 30), wl)
    return quiet(MEProblem, obs, line=6302, backend="torch", compile=False,
                 precision=precision), wl


def test_c34_torch_backend_honours_float32():
    """C34 (fixed): precision='float32' must give float32 Stokes."""
    prob, wl = _torch_problem("float32")
    p = torch.tensor([[800.0, 0.7, 0.4, 0.5, 12.0, 0.2, 0.035, 0.2, 0.8]]).repeat(4, 1)
    I, Q, U, V = prob.model(p, wl.expand(4, -1))
    assert I.dtype == torch.float32


def test_c35_levmar_runs_with_torch_backend():
    """C35 (fixed): LevMarSolver must run with the torch backend."""
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    prob, _ = _torch_problem("float32")
    solver = quiet(LevMarSolver, prob,
                   atmosphere=quiet(SphericalSoftplusAtmosphere, grid_shape=(2, 2)))
    try:
        quiet(solver.solve)
    except RuntimeError as e:
        raise AssertionError(f"LevMarSolver(torch) crashed: {e}") from e


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_c36_psf_clamps_on_cuda():
    """C36 (fixed): PSF clamp bounds must not be CPU tensors."""
    from neural_me.optimization.degradation import ExtendedPSFOperator, TwoPSFOperator

    ext = quiet(ExtendedPSFOperator, fwhm=2.0, epsilon=0.1).to("cuda")
    ext.clamp_parameters()
    two = quiet(TwoPSFOperator, torch.ones(3, 3), tune_sigma_stray_pixels=True).to("cuda")
    two.clamp_parameters()


# --------------------------------------------------------------------- #
#  Found while certifying fixes
# --------------------------------------------------------------------- #

def _lm_synthetic(seed=0, ny=4, nx=4, nw=40, noise=1e-3):
    from neural_me.core import MEProblem
    from neural_me.core.observation import Observation

    g = torch.Generator().manual_seed(seed)
    n = ny * nx
    wl = torch.linspace(-0.4, 0.4, nw)
    truth = torch.tensor([[900.0, 0.9, 0.6, 0.4, 12.0, 0.25, 0.032, 0.2, 0.8]]).repeat(n, 1)
    truth = truth * (1 + 0.2 * (torch.rand(truth.shape, generator=g) - 0.5))
    p0 = quiet(MEProblem, Observation(torch.zeros(ny, nx, 4, nw), wl),
               line=6302, backend="torch", compile=False)
    data = torch.stack(p0.model(truth, wl.expand(n, -1)), 1).reshape(ny, nx, 4, nw).detach()
    data = data + noise * torch.randn(data.shape, generator=g)
    prob = quiet(MEProblem, Observation(data, wl), line=6302, backend="torch", compile=False)
    return prob, data, wl, truth


def _lm_fit(noise_weights=None, **solve_kwargs):
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    prob, data, wl, truth = _lm_synthetic()
    ny, nx = data.shape[:2]
    atm = quiet(SphericalSoftplusAtmosphere, grid_shape=(ny, nx))
    solver = quiet(LevMarSolver, prob, atmosphere=atm)
    if noise_weights is not None:
        solver.set_weights(*([noise_weights] * 4))
    quiet(solver.solve, max_iter=60, verbose=False, **solve_kwargs)
    st = atm.to_stack().reshape(-1, 9).detach()
    syn = torch.stack(prob.model(st, wl.expand(st.shape[0], -1)), 1).reshape(data.shape)
    return solver, float((syn - data).pow(2).mean().sqrt())


def test_c37_levmar_default_solve_reaches_the_noise():
    """C37 (fixed): with unit weights chi2_threshold must not stop the fit after one step."""
    with pytest.warns(UserWarning, match="chi2_threshold is ignored"):
        solver, rms = _lm_fit()
    assert len(solver.chi2_history) > 3
    assert rms < 2e-3  # noise is 1e-3


def test_c37_levmar_threshold_is_reduced_chi2_with_noise_set():
    """C37 (fixed): with the noise set, chi2_threshold=1 stops at a fit at the noise."""
    solver, rms = _lm_fit(noise_weights=1e-3)
    assert solver.chi2_history[-1] < 1.0
    assert rms < 2e-3


@pytest.mark.parametrize("per_pixel", [True, False])
def test_c39_levmar_warns_when_max_iter_binds(per_pixel):
    """C39 (fixed): stopping on max_iter with the fit still improving must warn."""
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    prob, data, wl, truth = _lm_synthetic()
    atm = quiet(SphericalSoftplusAtmosphere, grid_shape=data.shape[:2])
    solver = quiet(LevMarSolver, prob, atmosphere=atm)
    solver.set_weights(1e-3, 1e-3, 1e-3, 1e-3)
    with pytest.warns(UserWarning, match="max_iter=2 reached"):
        quiet(solver.solve, max_iter=2, verbose=False, per_pixel=per_pixel)


def test_c39_levmar_silent_when_converged():
    """C39 (fixed): a fit that stops on its own criteria does not warn about max_iter."""
    import warnings as _w
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    prob, data, wl, truth = _lm_synthetic()
    atm = quiet(SphericalSoftplusAtmosphere, grid_shape=data.shape[:2])
    solver = quiet(LevMarSolver, prob, atmosphere=atm)
    solver.set_weights(1e-3, 1e-3, 1e-3, 1e-3)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        res = quiet(solver.solve, max_iter=200, verbose=False)
    assert res["n_active"] == 0
    assert not [w for w in caught if "max_iter" in str(w.message)]


def test_c24_stereo_set_parameters_accepts_scalars():
    """C24 (fixed): one value for the whole grid is broadcast, full arrays still work."""
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

    atm = quiet(StereoscopicAtmosphere, grid_shape=(2, 3))
    atm.set_parameters({"thermo1_eta0": 42.0, "thermo2_vlos": np.arange(6.0)})
    assert torch.allclose(atm.get_atmosphere(1).eta0, torch.full((2, 3), 42.0), rtol=1e-4)
    assert torch.allclose(atm.get_atmosphere(2).vlos.flatten(), torch.arange(6.0), atol=1e-5)


def test_c25_even_kernels_rejected_where_unsupported():
    """C25 (fixed): even kernels raise a clear error for conv2d and spectral PSFs; FFT still accepts them."""
    from neural_me.optimization.degradation import SpatialPSFOperator, SpectralPSFOperator

    with pytest.raises(ValueError, match="odd"):
        quiet(SpatialPSFOperator, torch.ones(4, 4), method="conv2d", verbose=False)
    with pytest.raises(ValueError, match="odd"):
        quiet(SpectralPSFOperator, torch.ones(4))
    quiet(SpatialPSFOperator, torch.ones(4, 4), method="fft", verbose=False)
    quiet(SpatialPSFOperator, torch.ones(5, 5), method="conv2d", verbose=False)
    quiet(SpectralPSFOperator, torch.ones(5))


def test_c27_levmar_l1_history_matches_meproblem():
    """C27 (fixed): LevMarSolver's L1 diagnostic equals MEProblem's L1 loss."""
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    prob, data, wl, truth = _lm_synthetic()
    prob.loss_type = "l1"
    atm = quiet(SphericalSoftplusAtmosphere, grid_shape=data.shape[:2])
    solver = quiet(LevMarSolver, prob, atmosphere=atm)
    p = solver._params_from_atmosphere()
    syn = solver._forward_only(p)
    lm_l1 = solver._compute_l1_loss(solver._obs_flat, syn)
    assert lm_l1 == pytest.approx(float(prob.compute_loss(atm)), rel=1e-5)


def test_c28_levmar_global_jacobians_only_after_accepted_steps():
    """C28 (fixed): the global LM computes one Jacobian at the start and one per accepted step."""
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    prob, data, wl, truth = _lm_synthetic()
    atm = quiet(SphericalSoftplusAtmosphere, grid_shape=data.shape[:2])
    solver = quiet(LevMarSolver, prob, atmosphere=atm)
    solver.set_weights(1e-3, 1e-3, 1e-3, 1e-3)
    calls = {"n": 0}
    orig = solver._forward_and_jacobian

    def counted(params):
        calls["n"] += 1
        return orig(params)

    solver._forward_and_jacobian = counted
    quiet(solver.solve, max_iter=40, verbose=False, per_pixel=False, recompute_jacobian_after=1)
    h = solver.chi2_history
    accepted = sum(1 for a, b in zip(h, h[1:]) if b < a)
    assert calls["n"] == 1 + accepted


def test_c29_cpp_warns_when_vlos_is_clipped():
    """C29 (fixed): the cpp backend warns when |vlos| exceeds pyMilne's 13 km/s clip."""
    import warnings as _w
    from neural_me.physics.milne_eddington import MilneEddington

    wl = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    me = quiet(MilneEddington, line=6302, backend="cpp", lambda_axis=wl)
    p = torch.tensor([[500.0, 1.0, 0.5, 16.0, 10.0, 0.2, 0.03, 0.2, 0.8]], dtype=torch.float64)
    with pytest.warns(UserWarning, match="clipped"):
        quiet(me, p, wl)
    p[0, 3] = 12.0
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        quiet(me, p, wl)
    assert not [w for w in caught if "clipped" in str(w.message)]


def test_a10_cartesian_field_computed_once_per_stack():
    """A10 (fixed): CartesianAtmosphere.to_stack builds B once and matches the properties."""
    from neural_me.core.atmospheres import CartesianAtmosphere

    atm = quiet(CartesianAtmosphere, grid_shape=(3, 3))
    calls = {"n": 0}
    orig = atm._field

    def counted():
        calls["n"] += 1
        return orig()

    atm._field = counted
    st = atm.to_stack()
    assert calls["n"] == 1
    atm._field = orig
    assert torch.equal(st[..., 0], atm.B) and torch.equal(st[..., 1], atm.gamma)
    assert torch.equal(st[..., 2], atm.xi)


def test_c10_me_model_passes_gradcheck_in_float64():
    """C10 (fixed): with float64 Voigt profiles the ME Jacobian matches finite differences."""
    from neural_me.physics.milne_eddington import MilneEddington

    me = quiet(MilneEddington, line=6173)
    wl = torch.linspace(-0.5, 0.5, 30, dtype=torch.float64)
    p = torch.tensor([[800.0, 0.8, 0.5, 0.7, 12.0, 0.25, 0.03, 0.2, 0.8]],
                     dtype=torch.float64, requires_grad=True)
    f = lambda q: torch.stack(me(q, wl.expand(1, -1)), 1)
    assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("a", [0.0, 0.3])
def test_c26_hui_faraday_slope_at_line_centre(a):
    """C26 (fixed): dF/dv at v = 0 equals the finite-difference slope, not 0."""
    from neural_me.physics.zeeman import fvoigt

    v = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    damp = torch.tensor([a], dtype=torch.float64)
    fvoigt(damp, v, method="hui")[1].sum().backward()
    h = 1e-6
    fd = (fvoigt(damp, torch.tensor([h], dtype=torch.float64))[1]
          - fvoigt(damp, torch.tensor([-h], dtype=torch.float64))[1]) / (2 * h)
    assert float(v.grad) == pytest.approx(float(fd), rel=1e-6)


def test_c11_spatial_psf_is_a_true_convolution():
    """C11 (fixed): both paths equal scipy.signal.convolve2d for an asymmetric PSF."""
    from scipy.signal import convolve2d
    from neural_me.optimization.degradation import SpatialPSFOperator

    ny, nx = 24, 30
    atm = types.SimpleNamespace(grid_shape=(ny, nx))
    g = torch.Generator().manual_seed(0)
    img = torch.rand(ny, nx, dtype=torch.float64, generator=g)
    k = torch.rand(5, 5, dtype=torch.float64, generator=g)
    k[0, 0] = 5.0
    k = k / k.sum()
    ref = convolve2d(img.numpy(), k.numpy(), mode="same")[2:-2, 2:-2]
    for method in ("fft", "conv2d"):
        op = quiet(SpatialPSFOperator, k, method=method, verbose=False, padding_mode="replicate")
        out = op((img.reshape(-1, 1),) * 4, atm)[0].reshape(ny, nx).numpy()[2:-2, 2:-2]
        assert np.allclose(out, ref, atol=1e-12)


def test_c12_spectral_psf_is_a_true_convolution():
    """C12 (fixed): an asymmetric spectral kernel gives numpy.convolve's response."""
    from scipy.signal import convolve
    from neural_me.optimization.degradation import SpectralPSFOperator

    x = torch.zeros(1, 21, dtype=torch.float64)
    x[0, 10] = 1.0
    k = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64)
    out = quiet(SpectralPSFOperator, k)((x,) * 4, types.SimpleNamespace(grid_shape=(1,)))[0][0]
    ref = convolve(x[0].numpy(), (k / k.sum()).numpy(), mode="same")
    assert np.allclose(out.numpy(), ref, atol=1e-12)
