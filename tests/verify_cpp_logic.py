# Minimal Verification Script for C++ Backend (No Plotting)
import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from neural_me.physics.milne_eddington import MilneEddington, get_available_backends

print("="*60)
print("Backend Availability Check")
print("="*60)
backends = get_available_backends()
print(f"PyTorch backend: {backends['torch']}")
print(f"C++ backend: {backends['cpp']}")
print()

if not backends['cpp']:
    print("FATAL: C++ backend not available!")
    sys.exit(1)

# Test parameters (single pixel)
# Line 1: Fe I 6301.5 (as verified in LINEAS.txt and verify_pytorch_basic.py)
nlinea = 1 
B = torch.tensor([992.0])
gamma = torch.tensor([np.deg2rad(134.0)])
xi = torch.tensor([np.deg2rad(145.0)])
vlos = torch.tensor([0.0])
eta0 = torch.tensor([73.0])
a = torch.tensor([0.2])
ddop = torch.tensor([0.02])
S_0 = torch.tensor([0.5])
S_1 = torch.tensor([0.5])

# Wavelength axis
lambda_axis = torch.arange(-2.8, 2.8, 20e-3)

print("="*60)
print("Running Synthesis")
print("="*60)

# PyTorch
model_torch = MilneEddington(nlinea=nlinea, backend='torch')
stokes_torch = model_torch.stokesSyn(
    model_torch.param, lambda_axis, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1
)

# C++
model_cpp = MilneEddington(nlinea=nlinea, backend='cpp')
# Need to pass lambda_axis in kwarg for first call to initialize backend
stokes_cpp = model_cpp(
    B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, 
    lambda_axis=lambda_axis
)

# Comparison
diff = (stokes_torch - stokes_cpp).abs()
max_diff = diff.max().item()

print(f"Max absolute difference: {max_diff:.2e}")

if max_diff < 1e-4:
    print("✓ SUCCESS: Backends produce identical results!")
else:
    print("✗ FAILURE: Significant differences detected!")
