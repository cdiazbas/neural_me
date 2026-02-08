"""
Verify PyTorch backend fix for Doppler width normalization.
This tests that removing the /ddop normalization produces physically reasonable results.
"""
import torch
import numpy as np
from neural_me.physics.milne_eddington import MilneEddington

# Test parameters (same as comparison script)
lambda0 = 6301.5080
JUp = 2.0
JLow = 2.0
gUp = 1.5
gLow = 1.833
lambdaStart = 6300.8
lambdaStep = 0.03
nLambda = 50
lambda_axis = np.linspace(lambdaStart, lambdaStep * nLambda + lambdaStart, nLambda)

# Physical parameters
B = torch.tensor([[992.]])           # 992 G
gamma = torch.tensor([[np.deg2rad(134.)]])  # 134°
xi = torch.tensor([[np.deg2rad(145.)]])     # 145°
vlos = torch.tensor([[0.0]])         # 0 km/s
eta0 = torch.tensor([[73.]])         # Line-to-continuum ratio
a = torch.tensor([[0.2 * 10]])       # Damping
ddop = torch.tensor([[0.02]])        # Doppler width (Å)
S_0 = torch.tensor([[0.5]])          # Source function
S_1 = torch.tensor([[0.5]])          # Source function gradient

print("=" * 60)
print("PyTorch Backend Verification (Post-Fix)")
print("=" * 60)
print(f"B = {B.item():.1f} G")
print(f"gamma = {np.rad2deg(gamma.item()):.1f}°")
print(f"ddop = {ddop.item():.3f} Å")
print()

# Create model and synthesize
model = MilneEddington(nlinea=1, backend='torch')
wl_tensor = torch.from_numpy(lambda_axis).float()

# Stack parameters
params = torch.cat([B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1], dim=1)

# Synthesize
stokes = model(params, wl_tensor)

# Extract Stokes parameters
I = stokes[0, 0, :].detach().numpy()
Q = stokes[0, 1, :].detach().numpy()
U = stokes[0, 2, :].detach().numpy()
V = stokes[0, 3, :].detach().numpy()

print("Results:")
print(f"  Stokes I: [{I.min():.4f}, {I.max():.4f}]")
print(f"  Stokes Q: [{Q.min():.4f}, {Q.max():.4f}]")
print(f"  Stokes U: [{U.min():.4f}, {U.max():.4f}]")
print(f"  Stokes V: [{V.min():.4f}, {V.max():.4f}]")
print()

# Physical sanity checks
print("Sanity Checks:")
print(f"  ✓ I in [0, 1]: {I.min() >= 0 and I.max() <= 1}")
print(f"  ✓ I shows absorption: {I.min() < 0.9}")
print(f"  ✓ V is antisymmetric: {abs(V.min() + V.max()) < 0.01}")
print(f"  ✓ |V| < 0.2 (reasonable): {abs(V).max() < 0.2}")
print(f"  ✓ |Q|, |U| < 0.1 (reasonable): {abs(Q).max() < 0.1 and abs(U).max() < 0.1}")
print()

# Expected behavior after fix:
# - Stokes V should be O(0.1) for B~1000G (not O(0.02) as before)
# - Profiles should be physically reasonable
expected_v_amplitude = abs(V).max()
print(f"Stokes V amplitude: {expected_v_amplitude:.4f}")
if 0.05 < expected_v_amplitude < 0.2:
    print("✓ Stokes V amplitude is physically reasonable for B=992G")
else:
    print("✗ WARNING: Stokes V amplitude seems unusual")

print("=" * 60)
print("✓ PyTorch backend verification complete!")
print("=" * 60)
