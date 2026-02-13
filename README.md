# Neural ME

**Differentiable Milne-Eddington inversions of spectropolarimetric data using PyTorch.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE.rst)

## Overview

Neural ME is a modern, GPU-accelerated framework for Milne-Eddington (ME) spectropolarimetric inversions. It provides:

- **Differentiable forward model**: Full PyTorch implementation with automatic differentiation
- **Flexible parameterization**: Cartesian B-field (Blos, BQ, BU) to avoid angular periodicity issues
- **Multi-instrument support**: Stereoscopic inversions combining data from multiple viewpoints
- **Optimization tricks**: Gradient normalization, parameter clamping, WFA initialization
- **Modular design**: Clean separation of physics, optimization, and I/O components

## Key Features

| Feature | Description |
|---------|-------------|
| **CartesianParameterModel** | Optimize (Blos, BQ, BU) → convert to (B, γ, ξ) |
| **StereoscopicParameterModel** | Shared B-field with frame rotation for multi-instrument |
| **MultiMEProblem** | Weighted loss combination from multiple observations |
| **WFA Initialization** | Weak-field approximation for initial B-field estimates |
| **GPU Acceleration** | Full CUDA support via PyTorch |

## Installation

```bash
# Clone the repository
git clone https://github.com/cdiazbas/neural_me.git
cd neural_me

# Fresh install (choose one):
pip install -e .          # Development mode (editable)
# OR
pip install .             # Regular install
```

## Quick Start

```python
import numpy as np
from neural_me import Observation, MEProblem, PixelSolver

# Load Stokes data (ny, nx, 4, n_wavelengths)
stokes_data = np.load('my_stokes_data.npy')
wavelengths = np.linspace(-0.3, 0.3, 50)

# Create observation and problem
obs = Observation(stokes_data, wavelengths)
problem = MEProblem(obs, nlinea=7)  # Fe I 6173

# Run inversion with Cartesian B-field parameterization
solver = PixelSolver(problem, use_cartesian=True)
solver.solve(n_iterations=500, lr=0.01, normalize_gradients=True)

# Get results
atmosphere = solver.get_atmosphere()
print(f"B-field: {atmosphere.B.mean():.1f} G")
```

## Project Structure

```
neural_me/
├── src/neural_me/
│   ├── core/           # Core data structures
│   │   ├── atmosphere.py    # Atmosphere parameters
│   │   ├── observation.py   # Stokes data container
│   │   ├── problem.py       # MEProblem, MultiMEProblem
│   │   └── parameters.py    # ParameterModel classes
│   ├── physics/        # Physical models
│   │   ├── milne_eddington.py   # ME synthesis
│   │   ├── wfa.py               # Weak-field approximation
│   │   ├── lineas.py            # Spectral line database
│   │   └── zeeman.py            # Zeeman splitting
│   ├── optimization/   # Solvers
│   │   └── pixel_solver.py
│   ├── regularization/ # Spatial smoothing
│   │   └── spatial.py
│   └── utils/          # Utilities
│       ├── io.py       # FITS I/O
│       ├── viz.py      # Plotting
│       └── misc.py     # Helpers
└── notebooks/          # Example notebooks
```

## Parameter Models

### CartesianParameterModel (Single Atmosphere)

Optimizes Cartesian B-field components to avoid angular periodicity:

```python
from neural_me import CartesianParameterModel

model = CartesianParameterModel(n_pixels=100, grid_shape=(10, 10))
atmosphere = model()  # Returns Atmosphere with (B, gamma, xi, ...)
```

### StereoscopicParameterModel (Multi-Instrument)

For stereoscopic observations (e.g., Hinode + Solar Orbiter):

```python
from neural_me import StereoscopicParameterModel, MultiMEProblem

# Shared B-field, rotated between frames
stereo = StereoscopicParameterModel(
    n_pixels=100, grid_shape=(10, 10),
    gamma=15.0,  # Viewing angle difference (degrees)
)

# Get atmospheres for both instruments
atm1, atm2 = stereo()
```

## Supported Spectral Lines

| ID | Line | λ₀ (Å) | g_eff |
|----|------|--------|-------|
| 1 | Fe I | 6301.5 | 1.67 |
| 2 | Fe I | 6302.5 | 2.50 |
| 7 | Fe I | 6173.3 | 2.50 |

## Citation

If you use Neural ME in your research, please cite:

```bibtex
@software{neural_me,
  author = {Díaz Baso, C. J.},
  title = {Neural ME: Differentiable Milne-Eddington Inversions},
  year = {2024},
  url = {https://github.com/cdiazbas/neural_me}
}
```

## License

BSD 3-Clause License. See [LICENSE.rst](LICENSE.rst) for details.

## Acknowledgments

Developed at the Rosseland Centre for Solar Physics (RoCS), University of Oslo.

## Development

Contributions are welcome! Please open an issue or submit a pull request on GitHub. Remember to follow the code style and run "pre-commit run --all-files" before committing.