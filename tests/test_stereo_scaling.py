import os
import sys

import numpy as np
import torch

# Ensure we can import neural_me
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere


def test_global_field_gradient_uses_normalization_factor():
    """Guard against regressing to physical-space optimization for global B params.

    If global_Blos is computed as raw * norm, then d(global_Blos.mean)/d(raw.mean) == norm.
    """
    atm = StereoscopicAtmosphere(grid_shape=(4, 4))

    atm.zero_grad(set_to_none=True)
    loss = atm.global_Blos.mean()
    loss.backward()

    grad_mean = atm.global_Blos_raw.grad.abs().mean().item()
    expected = atm.normalization["global_Blos"] / np.prod((4, 4))

    assert np.isclose(grad_mean, expected, rtol=1e-5, atol=1e-7)


def test_global_b_set_and_clamp_are_in_normalized_space():
    """Ensure set/clamp for global B are implemented in normalized space but exposed in physical units."""
    atm = StereoscopicAtmosphere(grid_shape=(2, 2))

    atm.set_parameters({"global_Blos": np.full((2, 2), 1500.0, dtype=np.float32)})
    assert torch.allclose(atm.global_Blos_raw, torch.full((2, 2), 3.0))
    assert torch.allclose(atm.global_Blos, torch.full((2, 2), 1500.0))

    atm.global_Blos_raw.data.fill_(10.0)  # 10 * 500 = 5000G (should clamp to 3500G => raw=7)
    atm.clamp_parameters()

    assert torch.allclose(atm.global_Blos_raw, torch.full((2, 2), 7.0))
    assert torch.allclose(atm.global_Blos, torch.full((2, 2), 3500.0))


def test_global_b_update_is_amplified_vs_physical_paramization():
    """Detect the exact symptom: tiny updates when optimizing physical-space B directly.

    We compare one SGD step with the same LR on:
    - raw normalized parameter (current expected design)
    - equivalent physical parameterization baseline
    """
    lr = 1e-2
    norm = 500.0

    # Current design path: B = raw * norm
    raw = torch.nn.Parameter(torch.tensor([0.2]))  # 100G
    opt_raw = torch.optim.SGD([raw], lr=lr)
    target_b = torch.tensor([150.0])

    opt_raw.zero_grad()
    b_from_raw = raw * norm
    loss_raw = (b_from_raw - target_b).pow(2).mean()
    loss_raw.backward()
    opt_raw.step()
    delta_b_raw = ((raw.detach() * norm) - torch.tensor([100.0])).abs().item()

    # Physical-space baseline: B = param (the old problematic design)
    b_phys = torch.nn.Parameter(torch.tensor([100.0]))
    opt_phys = torch.optim.SGD([b_phys], lr=lr)

    opt_phys.zero_grad()
    loss_phys = (b_phys - target_b).pow(2).mean()
    loss_phys.backward()
    opt_phys.step()
    delta_b_phys = (b_phys.detach() - torch.tensor([100.0])).abs().item()

    # With same LR and this quadratic loss, normalized update in physical units should be much larger.
    assert delta_b_raw > 100.0 * delta_b_phys


def test_vlos_gradient_uses_normalization_factor():
    """Guard against regressing thermo linear params to physical-space optimization."""
    atm = StereoscopicAtmosphere(grid_shape=(2, 2))

    atm.zero_grad(set_to_none=True)
    loss = atm.get_atmosphere(1).vlos.mean()
    loss.backward()

    grad_mean = atm.thermo1_vlos.grad.abs().mean().item()
    expected = atm.normalization["vlos"] / np.prod((2, 2))

    assert np.isclose(grad_mean, expected, rtol=1e-5, atol=1e-7)


def test_eta0_set_and_clamp_are_normalized_through_softplus_raw():
    """Ensure softplus parameterization keeps eta0 in physical range through normalized raw space."""
    atm = StereoscopicAtmosphere(grid_shape=(2, 2))

    atm.set_parameters({"thermo1_eta0": np.full((2, 2), 200.0, dtype=np.float32)})
    eta0_phys = atm.get_atmosphere(1).eta0
    assert torch.allclose(eta0_phys, torch.full((2, 2), 200.0), rtol=1e-4, atol=1e-4)

    # Drive raw very high and ensure clamp enforces upper bound.
    atm.thermo1_eta0_raw.data.fill_(100.0)
    atm.clamp_parameters()

    eta0_max = atm.get_atmosphere(1).eta0.max().item()
    assert eta0_max <= 600.0 + 1e-3


