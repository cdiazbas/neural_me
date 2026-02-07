
import torch
import pytest
import os
import sys

# Ensure we can import neural_me
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from neural_me.physics.milne_eddington import MilneEddington

def test_consistency():
    """
    Verifies that the current model implementation matches the 'golden' reference data.
    """
    golden_path = os.path.join(os.path.dirname(__file__), "golden_data.pt")
    
    if not os.path.exists(golden_path):
        pytest.skip(f"Golden data not found at {golden_path}. Run generate_golden.py first.")
    
    print(f"Loading golden data from {golden_path}...")
    data = torch.load(golden_path)
    
    params = data["params"]
    wl = data["wl"]
    expected_stokes = data["stokes"]
    nlinea = data["nlinea"]
    
    print(f"Initializing model (nlinea={nlinea})...")
    model = MilneEddington(nlinea=nlinea)
    
    print("Running forward pass...")
    with torch.no_grad():
        I, Q, U, V = model(params, wl)
    
    current_stokes = torch.stack([I, Q, U, V], dim=1)
    
    # Check for NaNs
    assert not torch.isnan(current_stokes).any(), "Output contains NaNs!"
    
    # Check consistency
    # We use a relatively strict tolerance because we want to catch even small logic changes
    # float32 precision is around 1e-7. Complex calculations might accumulate error.
    # We'll start with typical float32 tolerance.
    
    # atol=1e-5, rtol=1e-4 should be safe for float32 matrix ops vs original
    # If we change algorithms significantly (e.g. approximations), we might need to relax this.
    
    # Using default torch.allclose for now which is rtol=1e-05, atol=1e-08 usually?
    # No, defaults are rtol=1e-05, atol=1e-08.
    
    print("Verifying consistency...")
    # I, Q, U, V match?
    
    # We can check max difference
    diff = (current_stokes - expected_stokes).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if max_diff > 1e-5:
        print("WARNING: Max difference is larger than 1e-5. Check for precision issues.")
    
    assert torch.allclose(current_stokes, expected_stokes, atol=1e-5, rtol=1e-4), \
        f"Model output mismatch! Max diff: {max_diff}"
    
    print("Consistency test passed!")

if __name__ == "__main__":
    test_consistency()
