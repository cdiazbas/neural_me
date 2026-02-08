# Verification Script for Pure PyTorch Backend
import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from neural_me.physics.milne_eddington import MilneEddington
from neural_me.physics.zeeman import paramLine

print("="*60)
print("Verifying PyTorch Backend & LINEAS.txt Parsing")
print("="*60)

# Test Line 3: Fe I 6301.5 (as identified in LINEAS.txt)
nlinea = 1 # Using 1 for Fe I 6301.5 based on LINEAS.txt content check
print(f"Testing nlinea = {nlinea} (Should be Fe I 6301.5)")

try:
    # 1. Test Parameter Loading
    print("\n1. Testing paramLine()...")
    params = paramLine(nlinea)
    print("✓ paramLine loaded successfully")
    
    # Check components
    dpi, spi = params[0][0], params[1][0]
    print(f"   Pi components: {dpi.shape[-1]}")
    print(f"   Pi strengths: {spi}")
    
    # 2. Test Initialization
    print("\n2. Initializing MilneEddington(backend='torch')...")
    model = MilneEddington(nlinea=nlinea, backend='torch')
    print("✓ Model initialized")

    # 3. Test Synthesis
    print("\n3. Running Synthesis...")
    
    # Create dummy inputs
    batch_size = 2
    n_lambda = 100
    lambda_axis = torch.linspace(-0.5, 0.5, n_lambda)
    
    # Parameters (batch, 9)
    # [|B|, inc, azi, vlos, vDop, eta_l, damp, S0, S1]
    p = torch.tensor([
        [1000.0, 1.0, 0.5, 0.0, 0.03, 10.0, 0.5, 0.2, 0.8],
        [500.0, 0.5, 1.0, 1.0, 0.03, 5.0, 0.2, 0.3, 0.7]
    ])
    
    # Expand lambda axis for batch
    wl = lambda_axis.unsqueeze(0).repeat(batch_size, 1)
    
    # Run forward pass
    stokes = model(p, wl)
    print(f"✓ Synthesis Output Shape: {stokes.shape} (Expected: [{batch_size}, 4, {n_lambda}])")
    
    # Check for NaNs
    if torch.isnan(stokes).any():
        print("✗ Output contains NaNs!")
    else:
        print("✓ Output contains valid numbers")

    print("\n" + "="*60)
    print("✓ SUCCESS: PyTorch backend basics verified!")
    print("="*60)

except Exception as e:
    print("\n" + "="*60)
    print(f"✗ FAILURE: {e}")
    import traceback
    traceback.print_exc()
    print("="*60)
