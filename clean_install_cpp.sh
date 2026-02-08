#!/bin/bash
# Clean C++ Backend Installation Script
# This script removes all C++ build artifacts and rebuilds the backend

set -e  # Exit on error

echo "============================================================"
echo "Cleaning C++ Backend Build Artifacts"
echo "============================================================"

cd /mn/stornext/d20/RoCS/carlosjd/projects/SUNTORCH/NEURALME/neural_me

# Remove build directories
echo "Removing build/ directory..."
rm -rf build/

# Remove generated Cython C++ file
echo "Removing generated pyMilne.cpp..."
rm -f external/pyMilne/pyMilne.cpp

# Remove compiled .so files
echo "Removing .so files..."
rm -f src/neural_me/physics/*.so
rm -f src/neural_me/physics/milne_eddington_cpp/*.so

# Remove copied interface files
echo "Removing copied MilneEddington.py..."
rm -f src/neural_me/physics/MilneEddington.py
rm -f src/neural_me/physics/milne_eddington_cpp/MilneEddington.py

# Remove Python cache
echo "Removing __pycache__..."
find src/neural_me/physics/milne_eddington_cpp -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "============================================================"
echo "Building C++ Backend"
echo "============================================================"
echo ""

# Build the C++ extension
python setup_cpp.py build_ext --inplace

echo ""
echo "============================================================"
echo "Verifying Installation"
echo "============================================================"
echo ""

# Test if backend is available
python -c "from neural_me.physics.milne_eddington_cpp import is_available; print(f'C++ backend available: {is_available()}')"

echo ""
echo "============================================================"
echo "✓ C++ Backend Installation Complete!"
echo "============================================================"
