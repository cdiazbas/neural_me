# Neural ME

**Differentiable Milne-Eddington inversions of spectropolarimetric data using PyTorch.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE.rst)

## Overview

Neural ME is a modern, GPU-accelerated framework for Milne-Eddington (ME) spectropolarimetric inversions. It provides:

- **Differentiable forward model**: Full PyTorch and C++ implementations with automatic differentiation
- **Flexible atmosphere parameterization**: Cartesian, spherical-softplus, and neural field (NF) models
- **Multi-instrument stereoscopic inversions**: Shared B-field with frame rotation between viewpoints
- **Instrumental degradation operators**: Spatial PSF, spectral PSF, stray light, coordinate warping
- **Multiple solvers**: Pixel-wise Adam, Levenberg-Marquardt, and stereoscopic solver
- **Modular design**: Clean separation of physics, atmospheres, optimization, and degradation

## Installation

```bash
git clone https://github.com/cdiazbas/neural_me.git
cd neural_me
pip install -e .
```

The C++ ME backend requires compilation (faster, CPU-only):
```bash
python setup_cpp.py build_ext --inplace
```

## Quick Start

### Single-instrument pixel inversion

```python
import numpy as np
from neural_me import Observation, MEProblem, PixelSolver

# Stokes data: (ny, nx, 4, n_wavelengths)
stokes_data = np.load('my_stokes_data.npy')
wavelengths = np.linspace(-0.3, 0.3, 50)  # relative to line centre [Å]

obs = Observation(data=stokes_data, wavelengths=wavelengths)
prob = MEProblem(observation=obs, line=6173, backend='cpp')

solver = PixelSolver(problem=prob)
solver.solve(n_iterations=500)

atm = solver.get_atmosphere()
```

### Stereoscopic inversion (two instruments)

```python
from neural_me import (
    Observation, MEProblem, StereoscopicAtmosphere
)
from neural_me.optimization import StereoSolver, ExtendedPSFOperator

obs1 = Observation(data=hinode_data, wavelengths=wl_hinode)
obs2 = Observation(data=solo_data,   wavelengths=wl_solo)

prob1 = MEProblem(observation=obs1, line=6301, backend='cpp')
prob2 = MEProblem(observation=obs2, line=6173, backend='cpp')

atm = StereoscopicAtmosphere(grid_shape=(ny, nx), gamma=101.0, delta=90.0)

degradation_operators = {
    2: ExtendedPSFOperator(fwhm=2.0, epsilon=0.20, kernel_size=11)
}

solver = StereoSolver(
    atmosphere=atm,
    problem1=prob1,
    problem2=prob2,
    lr=1e-2,
    alpha=0.5,
    degradation_operators=degradation_operators,
)
losses = solver.solve(n_iterations=1000)
```

## Atmosphere Models

| Class | Description |
|-------|-------------|
| `CartesianAtmosphere` | Cartesian B-field (Blos, BQ, BU) — avoids angular periodicity |
| `SphericalSoftplusAtmosphere` | Spherical (B, γ, φ) with softplus positivity on B |
| `NFAtmosphere` | Neural field (implicit MLP) for single-instrument |
| `StereoscopicAtmosphere` | Shared B-field with per-instrument thermodynamics |
| `StereoscopicNFAtmosphere` | Neural field version of the stereoscopic model |

## Solvers

| Class | Description |
|-------|-------------|
| `PixelSolver` | Adam-based pixel-wise inversion |
| `LevMarSolver` | Levenberg-Marquardt inversion |
| `StereoSolver` | Stereoscopic inversion with optional degradation operators |

## Degradation Operators

All inherit from `DegradationOperator` and are passed to `StereoSolver`:

| Class | Description |
|-------|-------------|
| `StrayLightOperator` | Flat stray-light mixing: `(1-ε)·S + ε·⟨S⟩` |
| `ExtendedPSFOperator` | Spatial PSF convolution + flat stray-light (learnable fwhm and ε) |
| `TwoPSFOperator` | Sinjan et al. (2026) two-component PSF model |
| `SpatialPSFOperator` | Spatial 2D convolution (FFT or direct) |
| `SpectralPSFOperator` | Spectral 1D convolution |
| `WarpOperator` | Thin plate spline coordinate warping |

## Supported Spectral Lines

| ID | Element | λ₀ (Å) | g_eff | Instrument |
|----|---------|--------|-------|------------|
| 1 | Fe I | 6301.5012 | 1.67 | Hinode/SP |
| 2 | Fe I | 6302.4936 | 2.50 | Hinode/SP |
| 3 | Si I | 10827.0890 | 1.50 | |
| 4 | Ca I | 10833.4100 | 1.50 | |
| 5 | Ca I | 10838.9910 | 1.50 | |
| 6 | Fe I | 15648.5088 | 3.00 | |
| 7 | Fe I | 6173.3356 | 2.50 | SDO/HMI, SolO/PHI |

Lines are resolved by approximate wavelength: `MEProblem(line=6173)` → Line ID 7.

## Project Structure

```
neural_me/
├── src/neural_me/
│   ├── core/
│   │   ├── atmospheres/        # CartesianAtmosphere, StereoscopicAtmosphere,
│   │   │                       # NFAtmosphere, StereoscopicNFAtmosphere, ...
│   │   ├── observation.py      # Observation (Stokes data container)
│   │   ├── problem.py          # MEProblem
│   │   └── warp.py             # Thin plate spline warping
│   ├── physics/
│   │   ├── milne_eddington.py       # Pure-Python ME forward model
│   │   ├── milne_eddington_torch.py # PyTorch ME forward model
│   │   ├── milne_eddington_cpp/     # C++ ME backend (fast, CPU)
│   │   ├── lineas.py           # Spectral line database
│   │   ├── zeeman.py           # Zeeman splitting
│   │   ├── wfa.py              # Weak-field approximation
│   │   └── cog.py              # Centre-of-gravity
│   ├── optimization/
│   │   ├── pixel_solver.py     # PixelSolver (Adam)
│   │   ├── levmar_solver.py    # LevMarSolver
│   │   ├── stereo_solver.py    # StereoSolver
│   │   └── degradation.py      # All DegradationOperator subclasses
│   ├── instruments/            # PSF loading utilities
│   ├── regularization/
│   │   ├── spatial.py          # Spatial smoothing
│   │   └── temporal.py         # Temporal regularization
│   └── utils/
│       ├── io.py               # FITS I/O
│       ├── viz.py              # Plotting helpers
│       └── misc.py             # Miscellaneous
└── tests/                      # Unit, integration, regression, benchmarks
```

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


## Development

Contributions are welcome! Please open an issue or submit a pull request on GitHub. Remember to follow the code style and run "pre-commit run --all-files" before committing.