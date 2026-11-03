import torch
import numpy as np
import pytest
import sys
import os

# Ensure we can import neural_me
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neural_me.optimization.degradation import (
    ExtendedPSFOperator,
    SpatialPSFOperator,
    StrayLightOperator,
    TwoPSFOperator,
)
from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere

def test_spatial_psf_operator_methods():
    # Setup atmosphere with 4x4 grid
    atm = StereoscopicAtmosphere(grid_shape=(4, 4))
    
    # 3x3 mock kernel
    kernel = torch.tensor([
        [0.1, 0.2, 0.1],
        [0.2, 0.4, 0.2],
        [0.1, 0.2, 0.1]
    ], dtype=torch.float64)
    # Normalize kernel
    kernel /= kernel.sum()
    
    # Instantiate both operators
    op_fft = SpatialPSFOperator(kernel=kernel, method="fft")
    op_conv = SpatialPSFOperator(kernel=kernel, method="conv2d")
    
    stokes_in = (
        torch.randn((16, 5), dtype=torch.float64),
        torch.randn((16, 5), dtype=torch.float64),
        torch.randn((16, 5), dtype=torch.float64),
        torch.randn((16, 5), dtype=torch.float64),
    )
    
    out_fft = op_fft(stokes_in, atm)
    out_conv = op_conv(stokes_in, atm)
    
    # Verify outputs are equivalent
    for s_fft, s_conv in zip(out_fft, out_conv):
        np.testing.assert_allclose(s_fft.detach().numpy(), s_conv.detach().numpy(), rtol=1e-7, atol=1e-7)


def test_extended_psf_operator():
    # Setup atmosphere with 4x4 grid
    atm = StereoscopicAtmosphere(grid_shape=(4, 4))

    # Instantiate simplified operator with fixed FWHM and no stray light.
    op = ExtendedPSFOperator(
        fwhm=2.0,
        epsilon=0.0,
        kernel_size=9,
        model="gaussian",
        tune_fwhm=False,
        tune_epsilon=False,
    )
    
    # Dummy stokes: shape (16, 5) representing (pixels, wavelengths)
    stokes_in = (
        torch.randn((16, 5), dtype=torch.float64), # I
        torch.randn((16, 5), dtype=torch.float64), # Q
        torch.randn((16, 5), dtype=torch.float64), # U
        torch.randn((16, 5), dtype=torch.float64), # V
    )
    
    stokes_out = op(stokes_in, atm)
    stokes_psf_only = op.spatial_psf(stokes_in, atm)
    
    # Assert output shapes
    for s_out, s_psf in zip(stokes_out, stokes_psf_only):
        assert s_out.shape == s_psf.shape
        # epsilon=0 means ExtendedPSFOperator must reduce to pure PSF output.
        np.testing.assert_allclose(s_out.detach().numpy(), s_psf.detach().numpy(), rtol=1e-7, atol=1e-7)

    # Now verify with epsilon > 0 flat FoV stray-light component
    op_stray = ExtendedPSFOperator(
        fwhm=2.0,
        epsilon=0.1,
        kernel_size=9,
        model="gaussian",
        tune_fwhm=False,
        tune_epsilon=False,
    )
    stokes_out_stray = op_stray(stokes_in, atm)
    
    for s_out in stokes_out_stray:
        assert s_out.shape == (16, 5)

if __name__ == "__main__":
    test_extended_psf_operator()


# --------------------------------------------------------------------- #
#  Stray-light epsilon bounds
#
#  Epsilon used to live on StereoscopicAtmosphere as a raw parameter kept in
#  [0, 1] by a hard clamp.  It now lives on the degradation operators and is
#  logit-parameterized, so the unit interval is guaranteed by the sigmoid and
#  clamp_parameters() only guards against sigmoid saturation.  These tests
#  carry over the property the old atmosphere-level test protected.
# --------------------------------------------------------------------- #


def _stray_light(**kwargs):
    return StrayLightOperator(epsilon=kwargs.pop("epsilon", 0.2), **kwargs)


def _extended_psf(**kwargs):
    return ExtendedPSFOperator(
        fwhm=2.0, epsilon=kwargs.pop("epsilon", 0.2), kernel_size=5, **kwargs
    )


EPSILON_OPERATORS = [_stray_light, _extended_psf]


@pytest.mark.parametrize("make_operator", EPSILON_OPERATORS)
def test_epsilon_constructor_rejects_out_of_range(make_operator):
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="epsilon must be in"):
            make_operator(epsilon=bad)


@pytest.mark.parametrize("make_operator", EPSILON_OPERATORS)
@pytest.mark.parametrize("epsilon", [0.0, 0.245, 1.0])
def test_epsilon_round_trips_through_logit(make_operator, epsilon):
    """The value handed to the constructor is the value the forward pass sees."""
    op = make_operator(epsilon=epsilon, tune_epsilon=True)
    recovered = op._get_epsilon(torch.device("cpu"), torch.float32).item()
    # 0 and 1 are clamped off the boundary to keep the logit finite.
    assert recovered == pytest.approx(epsilon, abs=1e-5)


@pytest.mark.parametrize("make_operator", EPSILON_OPERATORS)
def test_epsilon_stays_in_unit_interval_under_extreme_drift(make_operator):
    """Successor to the old atmosphere-level clamp test.

    The optimizer can push the raw parameter arbitrarily far; the sigmoid must
    still deliver a physical mixing fraction.
    """
    op = make_operator(tune_epsilon=True)
    for extreme in (1e4, -1e4):
        op.raw_epsilon.data.fill_(extreme)
        eps = op._get_epsilon(torch.device("cpu"), torch.float32)
        assert torch.isfinite(eps).all()
        assert 0.0 <= eps.item() <= 1.0


@pytest.mark.parametrize("make_operator", EPSILON_OPERATORS)
def test_clamp_parameters_bounds_raw_epsilon(make_operator):
    """clamp_parameters keeps the pre-activation out of sigmoid saturation."""
    op = make_operator(tune_epsilon=True)
    for extreme in (1e4, -1e4):
        op.raw_epsilon.data.fill_(extreme)
        op.clamp_parameters()
        assert -12.0 <= op.raw_epsilon.item() <= 12.0
        # Strictly interior once clamped, so gradients stay alive.
        eps = op._get_epsilon(torch.device("cpu"), torch.float32).item()
        assert 0.0 < eps < 1.0


@pytest.mark.parametrize("make_operator", EPSILON_OPERATORS)
def test_fixed_epsilon_is_a_buffer_not_a_parameter(make_operator):
    op = make_operator(epsilon=0.3, tune_epsilon=False)
    assert "raw_epsilon" not in dict(op.named_parameters())
    assert "fixed_epsilon" in dict(op.named_buffers())
    assert op._get_epsilon(torch.device("cpu"), torch.float32).item() == pytest.approx(0.3)


def test_two_psf_stray_fraction_is_bounded():
    """TwoPSFOperator uses the same logit pattern for its stray fraction."""
    kernel = torch.zeros(5, 5)
    kernel[2, 2] = 1.0
    op = TwoPSFOperator(psf_pd_kernel=kernel, tune_stray_fraction=True)

    for extreme in (1e4, -1e4):
        op.raw_stray_fraction.data.fill_(extreme)
        frac = op._get_stray_fraction(torch.device("cpu"), torch.float32)
        assert torch.isfinite(frac).all()
        assert 0.0 <= frac.item() <= 1.0

        op.clamp_parameters()
        assert -12.0 <= op.raw_stray_fraction.item() <= 12.0
