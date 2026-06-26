import torch
import numpy as np
import pytest
import sys
import os

# Ensure we can import neural_me
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neural_me.optimization.degradation import ExtendedPSFOperator, SpatialPSFOperator
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
