"""
Verify C++ Backend Derivatives.
This script compares analytical derivatives (from backward())
with numerical derivatives (finite differences) to ensure the parameter ordering is correct.
"""
import torch
import numpy as np
import sys
import os

# Ensure we can import the package
sys.path.insert(0, os.path.abspath('src'))

from neural_me.physics.milne_eddington_cpp import MilneEddingtonCPP

print("=" * 60)
print("Verifying C++ Backend Derivatives")
print("=" * 60)

# Setup parameters
lambda0 = 6301.5012
JUp = 2.0
JLow = 2.0
gUp = 1.5
gLow = 1.833
# Use a small wavelength grid for testing
lambda_axis = torch.linspace(-0.5, 0.5, 20)
mu = 1.0

# Initialize model
print("Initializing MilneEddingtonCPP model...")
model = MilneEddingtonCPP(
    lambda0=lambda0,
    JUp=JUp,
    JLow=JLow,
    gUp=gUp,
    gLow=gLow,
    lambda_axis=lambda_axis,
    mu=mu
)

# Test parameters (require grad)
# Order: [B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1]
# Values chosen to be in a sensitive region for derivatives
# B=1000, gamma=45deg, xi=45deg, vlos=1.0, eta0=10, a=0.5, ddop=0.03, S0=0.5, S1=0.5
val_B = 1000.0
val_gamma = 0.78 # ~45 deg
val_xi = 0.78
val_vlos = 1.0
val_eta0 = 10.0
val_a = 0.5
val_ddop = 0.03
val_S0 = 0.5
val_S1 = 0.5

params_init = [val_B, val_gamma, val_xi, val_vlos, val_eta0, val_a, val_ddop, val_S0, val_S1]
params = torch.tensor([params_init], requires_grad=True, dtype=torch.float64)

print(f"Parameters: {params_init}")

# Forward pass
print("\nRunning forward pass...")
stokes_tuple = model(params) # returns (I, Q, U, V) tuple
# Stack to get single tensor
stokes = torch.stack(stokes_tuple, dim=1) # (batch, 4, n_lambda)

# Define loss function: sum of squares of Stokes Q and U (polarized components)
# to avoid dominance of Stokes I
# L = (Q^2 + U^2 + V^2).sum()
L = (stokes[:, 1:]**2).sum()
print(f"Loss value: {L.item()}")

# Analytical backward pass
print("Running analytical backward pass...")
L.backward()

if params.grad is None:
    print("Error: params.grad is None!")
    sys.exit(1)

analytical_grads = params.grad.clone().detach().numpy()[0]
params.grad.zero_()

print("\nComparing numerical vs analytical gradients:")
print(f"{'Param':<10} {'Index':<5} {'Analytical':<15} {'Numerical':<15} {'Diff':<15} {'Rel Diff (%)':<15} {'Status':<10}")
print("-" * 95)

param_names = ['B', 'gamma', 'xi', 'vlos', 'eta0', 'a', 'ddop', 'S_0', 'S_1']
epsilon = 1e-5

failures = 0

for i, name in enumerate(param_names):
    # Central difference for better accuracy
    
    # Perturb +epsilon
    params_plus = params.clone().detach()
    params_plus[0, i] += epsilon
    stokes_plus_tuple = model(params_plus)
    stokes_plus = torch.stack(stokes_plus_tuple, dim=1)
    L_plus = (stokes_plus[:, 1:]**2).sum()
    
    # Perturb -epsilon
    params_minus = params.clone().detach()
    params_minus[0, i] -= epsilon
    stokes_minus_tuple = model(params_minus)
    stokes_minus = torch.stack(stokes_minus_tuple, dim=1)
    L_minus = (stokes_minus[:, 1:]**2).sum()
    
    # Numerical gradient (central difference)
    numerical_grad = (L_plus.item() - L_minus.item()) / (2 * epsilon)
    
    grad = analytical_grads[i]
    diff = abs(grad - numerical_grad)
    # Handle potentially small gradients
    denom = abs(numerical_grad) + 1e-9
    rel_diff = diff / denom * 100
    
    # Pass criteria: < 1% error OR very small absolute diff (< 1e-6)
    passed = rel_diff < 1.0 or diff < 1e-6
    status = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        failures += 1
        
    print(f"{name:<10} {i:<5} {grad:>15.5e} {numerical_grad:>15.5e} {diff:>15.5e} {rel_diff:>13.2f} {status}")

print("-" * 95)
if failures == 0:
    print("\n✓ All derivative checks PASSED! The parameter ordering is correct.")
else:
    print(f"\n✗ {failures} derivative checks FAILED! Please check parameter ordering.")
