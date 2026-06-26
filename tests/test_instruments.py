import torch
import numpy as np
import pytest
import sys
import os

# Ensure we can import neural_me
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neural_me.instruments import load_spectral_psf, load_spatial_psf, get_resource_path

def test_load_spectral_psfs():
    # 1. Load Hinode spectral PSF profile
    hinode_sp = load_spectral_psf("HINODE_SP_ins_prof.psf")
    assert isinstance(hinode_sp, torch.Tensor)
    assert hinode_sp.ndim == 1
    assert len(hinode_sp) > 0
    
    # 2. Load SO/PHI spectral PSFs
    phi_psf1 = load_spectral_psf("gaussian_phihrt.psf")
    phi_psf2 = load_spectral_psf("gaussian_phihrt_35mA.psf")
    
    assert isinstance(phi_psf1, torch.Tensor)
    assert phi_psf1.ndim == 1
    assert len(phi_psf1) > 0
    
    assert isinstance(phi_psf2, torch.Tensor)
    assert phi_psf2.ndim == 1
    assert len(phi_psf2) > 0

def test_load_spatial_psfs():
    # Load 2D Hinode spatial PSF
    hinode_spatial = load_spatial_psf("hinode_psf_size_256_def_-0.32.fits")
    assert isinstance(hinode_spatial, torch.Tensor)
    assert hinode_spatial.ndim == 2
    assert hinode_spatial.shape == (257, 257)

def test_invalid_resource():
    with pytest.raises(FileNotFoundError):
        get_resource_path("non_existent_file.psf")

if __name__ == "__main__":
    test_load_spectral_psfs()
    test_load_spatial_psfs()
    test_invalid_resource()
