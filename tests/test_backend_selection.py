"""
Test backend selection and availability checking.
"""
import torch
from neural_me.physics.milne_eddington import MilneEddington, get_available_backends

# Check available backends
print("=" * 60)
print("Backend Availability Check")
print("=" * 60)
backends = get_available_backends()
for name, available in backends.items():
    status = "✓ Available" if available else "✗ Not available"
    print(f"{name:10s}: {status}")
print()

# Test PyTorch backend (default)
print("=" * 60)
print("Testing PyTorch Backend (Default)")
print("=" * 60)
model_torch = MilneEddington(nlinea=1)
print(f"Model type: {type(model_torch).__name__}")
print(f"Model class: {model_torch.__class__.__module__}.{model_torch.__class__.__name__}")
print()

# Test explicit PyTorch backend
print("=" * 60)
print("Testing PyTorch Backend (Explicit)")
print("=" * 60)
model_torch_explicit = MilneEddington(nlinea=1, backend='torch')
print(f"Model type: {type(model_torch_explicit).__name__}")
print()

# Test C++ backend (if available)
print("=" * 60)
print("Testing C++ Backend")
print("=" * 60)
if backends['cpp']:
    try:
        # Create wavelength axis
        wl = torch.linspace(-1.0, 1.0, 50)
        
        model_cpp = MilneEddington(
            nlinea=1, 
            backend='cpp',
            lambda_axis=wl,
            mu=1.0
        )
        print(f"Model type: {type(model_cpp).__name__}")
        print(f"Model class: {model_cpp.__class__.__module__}.{model_cpp.__class__.__name__}")
        print("✓ C++ backend initialized successfully")
    except Exception as e:
        print(f"✗ C++ backend failed: {e}")
else:
    print("✗ C++ backend not available")
    print("  Run: python setup_cpp.py build_ext --inplace")
print()

# Test invalid backend
print("=" * 60)
print("Testing Invalid Backend (Should Fail)")
print("=" * 60)
try:
    model_invalid = MilneEddington(nlinea=1, backend='invalid')
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")
print()

print("=" * 60)
print("Backend Selection Test Complete")
print("=" * 60)
