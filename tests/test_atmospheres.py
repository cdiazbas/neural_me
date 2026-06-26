import torch
import numpy as np
import pytest
import sys
import os

# Ensure we can import neural_me
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere
from neural_me.core.atmospheres.cartesian import CartesianAtmosphere

def test_stereoscopic_normalization():
    # Setup atmosphere with 4x4 grid
    grid_shape = (4, 4)
    atm = StereoscopicAtmosphere(grid_shape=grid_shape)
    
    # 1. Verify that raw parameters are initialized around normalized values (order ~1.0)
    # Default Blos in physical units is 500.0, so Blos_raw should be around 1.0 (since norm is 500.0)
    blos_raw_mean = atm.global_Blos_raw.mean().item()
    assert 0.15 <= blos_raw_mean <= 0.25
    
    # 2. Verify that properties return physical values (order ~100.0)
    blos_phys_mean = atm.global_Blos.mean().item()
    assert 80.0 <= blos_phys_mean <= 120.0
    
    # 3. Verify that set_parameters correctly normalizes values
    atm.set_parameters({
        "global_Blos": np.full(grid_shape, 1500.0)
    })
    # raw parameter should be 1500.0 / 500.0 = 3.0
    assert torch.allclose(atm.global_Blos_raw, torch.tensor(3.0))
    # property should return 1500.0
    assert torch.allclose(atm.global_Blos, torch.tensor(1500.0))

    # 4. Verify clamp_parameters correctly clamps raw parameters
    # Set values outside limits: default clamp bounds for Blos is [-3500, 3500]
    # In normalized space, limits are [-7, 7]
    atm.global_Blos_raw.data.fill_(10.0)
    atm.clamp_parameters()
    # Should be clamped to 7.0
    assert torch.allclose(atm.global_Blos_raw, torch.tensor(7.0))


def test_cartesian_normalization():
    # Setup single-view cartesian atmosphere with 4x4 grid
    grid_shape = (4, 4)
    atm = CartesianAtmosphere(grid_shape=grid_shape)
    
    # Blos_raw should be around 1.0
    blos_raw_mean = atm._Blos.mean().item()
    assert 0.8 <= blos_raw_mean <= 1.2
    
    # Property should return physical value (around 500.0)
    blos_phys_mean = (atm._Blos * atm.norm["Blos"]).mean().item()
    assert 400.0 <= blos_phys_mean <= 600.0
