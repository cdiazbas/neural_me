"""
Simple test of the C++ backend using the MILNE class directly.
This demonstrates the low-level interface to the C++ extension.
"""
import torch
import numpy as np

# Import the MILNE wrapper class
from neural_me.physics.milne_eddington_cpp.milne_cpp import MILNE

# Line parameters for Fe I 6301.5
lambda0 = 6301.5080
JUp = 2.0
JLow = 2.0
gUp = 1.5
gLow = 1.833

# Wavelength axis
lambdaStart = 6300.8
lambdaStep = 0.03
nLambda = 50
lambda_axis = np.linspace(lambdaStart, lambdaStep * nLambda + lambdaStart, nLambda)

# Atmospheric parameters (batch size = 1)
B = torch.tensor([[992.]], requires_grad=True)           # Magnetic field strength (G)
gamma = torch.tensor([[np.deg2rad(134.)]], requires_grad=True)  # Field inclination (rad)
xi = torch.tensor([[np.deg2rad(145.)]], requires_grad=True)     # Field azimuth (rad)
vlos = torch.tensor([[1.0]], requires_grad=True)         # Line-of-sight velocity (km/s)
eta0 = torch.tensor([[73.]], requires_grad=True)         # Line-to-continuum ratio
a = torch.tensor([[0.2 * 10]], requires_grad=True)       # Damping parameter
ddop = torch.tensor([[0.02]], requires_grad=True)        # Doppler width (Å)
S_0 = torch.tensor([[0.5]], requires_grad=True)          # Source function (continuum)
S_1 = torch.tensor([[0.5]], requires_grad=True)          # Source function (gradient)
mu = torch.tensor([1.0])                                  # Viewing angle cosine

print("=" * 60)
print("Testing C++ Backend (MILNE class)")
print("=" * 60)
print(f"Line: Fe I {lambda0} Å")
print(f"Wavelength points: {nLambda}")
print(f"B = {B.item():.1f} G")
print(f"gamma = {np.rad2deg(gamma.item()):.1f}°")
print(f"xi = {np.rad2deg(xi.item()):.1f}°")
print(f"vlos = {vlos.item():.1f} km/s")
print()

# Create MILNE instance
fn = MILNE(lambda0, JUp, JLow, gUp, gLow, lambda_axis, mu)

# Synthesize Stokes profiles
stokes = fn(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)

print(f"Output shape: {stokes.shape}")
print(f"Expected: (batch=1, n_stokes=4, n_lambda={nLambda})")
print()

# Display some statistics
print("Stokes I:")
print(f"  Range: [{stokes[0, 0, :].min():.4f}, {stokes[0, 0, :].max():.4f}]")
print(f"  Mean: {stokes[0, 0, :].mean():.4f}")
print()
print("Stokes Q:")
print(f"  Range: [{stokes[0, 1, :].min():.4f}, {stokes[0, 1, :].max():.4f}]")
print(f"  Mean: {stokes[0, 1, :].mean():.4f}")
print()
print("Stokes U:")
print(f"  Range: [{stokes[0, 2, :].min():.4f}, {stokes[0, 2, :].max():.4f}]")
print(f"  Mean: {stokes[0, 2, :].mean():.4f}")
print()
print("Stokes V:")
print(f"  Range: [{stokes[0, 3, :].min():.4f}, {stokes[0, 3, :].max():.4f}]")
print(f"  Mean: {stokes[0, 3, :].mean():.4f}")
print()

# Test gradient computation
print("Testing autograd...")
loss = stokes.sum()
loss.backward()

print(f"✓ Gradients computed successfully")
print(f"  dL/dB shape: {B.grad.shape}")
print(f"  dL/dB = {B.grad.item():.6f}")
print()
print("=" * 60)
print("✓ C++ backend test complete!")
print("=" * 60)
