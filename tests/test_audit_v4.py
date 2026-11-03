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


@xfail("C04", "LevMarSolver keeps one scalar lambda and one accept/reject for all pixels")
def test_c04_levmar_lambda_is_per_pixel():
    from neural_me.optimization.levmar_solver import LevMarSolver

    src = code_only(LevMarSolver.solve)
    assert not re.search(r"^\s*lam\s*=\s*check_lambda\(", src, re.M)


# --------------------------------------------------------------------- #
#  Medium
# --------------------------------------------------------------------- #

@xfail("C05", "wcs_roll_angle returns +CROTA2, local_vertical_matrix expects peff=-CROTA2")
def test_c05_wcs_roll_angle_is_peff():
    pytest.importorskip("sunpy")
    import sunpy.map
    from astropy.io.fits import Header
    from neural_me.core.geometry import wcs_roll_angle

    h = Header()
    for k, v in dict(NAXIS=2, NAXIS1=10, NAXIS2=10, CRPIX1=5, CRPIX2=5, CRVAL1=0.0,
                     CRVAL2=0.0, CDELT1=1.0, CDELT2=1.0, CTYPE1="HPLN-TAN",
                     CTYPE2="HPLT-TAN", CUNIT1="arcsec", CUNIT2="arcsec",
                     CROTA2=30.0).items():
        h[k] = v
    h["DATE-OBS"] = "2022-03-31T19:39:59"
    smap = sunpy.map.Map(np.zeros((10, 10)), h)
    assert np.degrees(wcs_roll_angle(smap)) == pytest.approx(-30.0)


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


@xfail("A01", "ViewAtmosphere._compute_magnetic runs once per B/gamma/xi")
def test_a01_view_magnetic_computed_once():
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


@xfail("C10", "Voigt functions cast inputs to float32")
def test_c10_voigt_keeps_float64():
    from neural_me.physics.zeeman import fvoigt

    H, F = fvoigt(torch.tensor([0.1], dtype=torch.float64),
                  torch.linspace(-2, 2, 5, dtype=torch.float64))
    assert H.dtype == torch.float64 and F.dtype == torch.float64


@xfail("C11", "conv2d PSF path is cross-correlation, FFT path is convolution")
def test_c11_psf_paths_agree_for_asymmetric_kernel():
    from neural_me.optimization.degradation import SpatialPSFOperator

    atm = types.SimpleNamespace(grid_shape=(16, 20))
    torch.manual_seed(0)
    k = torch.rand(5, 5)
    k[0, 0] = 5
    S = torch.rand(16 * 20, 3)
    a = quiet(SpatialPSFOperator, k, method="fft", verbose=False)((S,) * 4, atm)[0]
    b = quiet(SpatialPSFOperator, k, method="conv2d", verbose=False)((S,) * 4, atm)[0]
    assert torch.allclose(a, b, atol=1e-5)


@xfail("C12", "spectral PSF zero-pads the wavelength axis")
def test_c12_spectral_psf_preserves_flat_continuum():
    from neural_me.optimization.degradation import SpectralPSFOperator

    atm = types.SimpleNamespace(grid_shape=(2,))
    I = torch.ones(2, 10)
    out = quiet(SpectralPSFOperator, torch.tensor([1.0, 2, 3, 2, 1]))((I,) * 4, atm)[0]
    assert torch.allclose(out, I, atol=1e-6)


@xfail("C13", "cpp backend uses its own line table and an invalid '{0 }' format string")
def test_c13_cpp_line_table_consistent():
    from neural_me.physics.milne_eddington_cpp import MilneEddington as cppmod

    src = code_only(cppmod)
    assert "{0 }" not in src
    assert "6173.334" not in src.replace(" ", "")


@xfail("C14", "torch backend ignores mu")
def test_c14_torch_backend_uses_mu():
    from neural_me.physics import milne_eddington_torch as mt

    assert re.search(r"\bmu\b", code_only(mt))


@xfail("C15", "LocalVerticalMixin registers its buffers lazily")
def test_c15_local_vertical_state_dict_round_trip():
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

    a = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    a.set_local_vertical_geometry(np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy())
    b = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    try:
        b.load_state_dict(a.state_dict())
    except RuntimeError as e:
        raise AssertionError(f"checkpoint does not load into a fresh atmosphere: {e}") from e


@xfail("C16", "ExtendedPSFOperator docstring calls post_psf the default")
def test_c16_extended_psf_doc_matches_default():
    from neural_me.optimization.degradation import ExtendedPSFOperator

    default = inspect.signature(ExtendedPSFOperator.__init__).parameters["mean_source"].default
    assert f"``mean_source='{default}'`` (default)" in (ExtendedPSFOperator.__doc__ or "")


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

@xfail("C19", "stereo thermo params stored with 'eta0' norm, clamped with 'thermo1_eta0' norm")
def test_c19_stereo_clamp_uses_storage_normalization():
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


@xfail("C21", "stereo angles are plain attributes, not buffers; device fixed at init")
def test_c21_stereo_angles_follow_module_device():
    from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

    atm = quiet(StereoscopicAtmosphere, grid_shape=(2, 2))
    assert "gamma_rad" in dict(atm.named_buffers())


@xfail("C22", "atan2(BU, BQ + 1e-15) has a ~1e17 gradient at zero transverse field")
def test_c22_cartesian_xi_gradient_finite_scale():
    from neural_me.core.atmospheres import CartesianAtmosphere

    atm = quiet(CartesianAtmosphere, grid_shape=(2,), init_type="zeros")
    atm.xi.sum().backward()
    g = max(float(p.grad.abs().max()) for p in atm.parameters() if p.grad is not None)
    assert g < 1e6


@xfail("C23", "SlicedAtmosphere indexes rows of a 2-D grid, not pixels")
def test_c23_sliced_atmosphere_flat_indices():
    from neural_me.core.atmospheres import SphericalAtmosphere

    atm = quiet(SphericalAtmosphere, grid_shape=(8, 8))
    assert atm[[0, 1, 2]].B.numel() == 3


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
