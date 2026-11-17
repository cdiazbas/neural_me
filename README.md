# Neural ME

**Differentiable Milne-Eddington inversions of spectropolarimetric data using PyTorch.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Neural ME is a framework for Milne-Eddington (ME) spectropolarimetric inversions, on CPU and GPU. It provides:

- **Differentiable forward model**: a PyTorch ME synthesis and a C++ one (pyMilne) whose gradient is computed as a
  vector-Jacobian product
- **Atmosphere parameterizations**: per pixel (Cartesian, spherical, spherical-softplus) and neural fields (Fourier
  features or a dense multiresolution grid)
- **Stereoscopic inversions**: one magnetic field shared by two vantage points, thermodynamic parameters per view
- **Instrumental degradation in the forward model**: spatial PSF, spectral PSF, stray light, reprojection onto the pixels
  of another instrument
- **Solvers**: pixel-wise Adam, Levenberg-Marquardt, and the stereoscopic solver
- **Penalties**: spatial smoothness and the null space of the reprojection

## Installation

The C++ backend lives in the `external/pyMilne` submodule, so clone with it:

```bash
git clone --recursive https://github.com/cdiazbas/neural_me.git
cd neural_me                          # or, in an existing clone: git submodule update --init
pip install -e .
```

The C++ backend (CPU only, the fast one) needs FFTW3, Eigen3, a C++20 compiler and a CPU with AVX2, e.g. in a conda
environment:

```bash
conda install -c conda-forge fftw eigen
python setup_cpp.py build_ext --inplace
```

Without it, `backend='torch'` uses the PyTorch synthesis.

## Quick start

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

### Stereoscopic inversion: the setup of the paper

The core of the reference inversion of Díaz Baso et al. (in prep.), *Hinode*/SP (Fe I 6302 Å) and SO/PHI-HRT
(Fe I 6173 Å): one `MEProblem` per instrument with its spectral PSF, a shared neural field with the grid encoding, the
PSF of *Hinode*/SP, and the reprojection, PSF core and stray light of SO/PHI-HRT. The arrays `d_hinode`, `d_phi`,
`psf_hinode` and `warp_valid` come from the co-registered data.

```python
import numpy as np
import torch
from neural_me import Observation, MEProblem, StereoscopicNFAtmosphere
from neural_me.optimization import (StereoSolver, SparseWarpOperator, SpatialPSFOperator,
                                    ExtendedPSFOperator, ComposeOperator)
from neural_me.regularization.spatial import smoothness_loss
from neural_me.instruments import get_resource_path

# Hinode/SP on the model grid, SO/PHI-HRT in its native pixels (zero outside its valid pixels)
obs1 = Observation(data=d_hinode, wavelengths=wl_hinode)
obs2 = Observation(data=d_phi, wavelengths=wl_phi)

# Spectral PSF (offset in mA, transmission), applied on a fine grid of spectral_step mA
profile = lambda name: np.loadtxt(get_resource_path(name))
problem1 = MEProblem(observation=obs1, line=6302, weights=[1, 10, 10, 10], loss_type="l1", backend="cpp",
                     spectralPSF=profile("HINODE_SP_ins_prof.psf"), spectral_step=10)
problem2 = MEProblem(observation=obs2, line=6173, weights=[1, 10, 10, 10], loss_type="l1", backend="cpp",
                     spectralPSF=profile("gaussian_phihrt.psf"), spectral_step=15)

# One field shared by both views, thermodynamics per view, on a multiresolution grid
# (encoding="grid"; the default "fourier" is the slower network of Paper I)
atmosphere = StereoscopicNFAtmosphere(grid_shape=(512, 512), gamma=101.2, beta=0.8, delta=90.0,
                                      encoding="grid", source="free", bounds="clamp")

# Hinode/SP PSF; for SO/PHI-HRT, reprojection onto its pixels, PSF core and stray light (fixed)
warp = SparseWarpOperator.from_file("phi_native_warp.npz")
degradation = {
    1: SpatialPSFOperator(kernel=torch.from_numpy(psf_hinode), method="fft", padding_mode="reflect"),
    2: ComposeOperator(warp, ExtendedPSFOperator(fwhm=2.5, epsilon=0.1, mean_source="post_psf",
                                                 tune_fwhm=False, tune_epsilon=False,
                                                 mask=torch.from_numpy(warp_valid))),
}

# Penalties enter as a function of the atmosphere, e.g. the spatial smoothness of the field
def penalty(atm):
    X = atm.to_stack()
    return 1e-2 * sum(smoothness_loss(X[..., c] / 500.0, penalty="l2") for c in range(3)) / X[..., 0].numel()

solver = StereoSolver(atmosphere, problem1, problem2, lr=1e-2, alpha=0.5,
                      degradation_operators=degradation, penalty=penalty)
losses = solver.solve(n_iterations=500, warmup=50)
params = atmosphere.to_numpy_stack()     # (ny, nx, 15): field and both thermodynamic sets
```

The penalty on the part of the maps that the reprojection does not see (Appendix C of the paper) is
`neural_me.regularization.nullspace.NullSpacePenalty`, added to the same `penalty` function.

## Atmosphere models

From `neural_me` (the per-pixel `SphericalAtmosphere` from `neural_me.core.atmospheres`):

| Class | Description |
|-------|-------------|
| `CartesianAtmosphere` | Cartesian B-field (Blos, BQ, BU), no angular periodicity |
| `SphericalAtmosphere` | Spherical (B, γ, φ) |
| `SphericalSoftplusAtmosphere` | Spherical (B, γ, φ) with softplus positivity on B |
| `NFAtmosphere` | Neural field, single instrument |
| `StereoscopicAtmosphere` | Per pixel: shared B-field, thermodynamics per instrument |
| `StereoscopicNFAtmosphere` | Neural field version of the stereoscopic model (`encoding="grid"` or `"fourier"`) |

## Solvers

From `neural_me` (`PixelSolver`) and `neural_me.optimization`:

| Class | Description |
|-------|-------------|
| `PixelSolver` | Adam, pixel by pixel |
| `LevMarSolver` | Levenberg-Marquardt per pixel; `field="lqlu"` takes the step in (Blos, Lq, Lu), which does not stall in strong transverse fields |
| `StereoSolver` | Stereoscopic inversion with degradation operators and penalties |

## Degradation operators

From `neural_me.optimization`; all inherit from `DegradationOperator` and are passed to `StereoSolver` per view:

| Class | Description |
|-------|-------------|
| `SpatialPSFOperator` | Spatial 2D convolution (FFT or direct) |
| `ExtendedPSFOperator` | Spatial PSF + flat stray light (fwhm and ε fixed or learnable) |
| `StrayLightOperator` | Flat stray light: `(1-ε)·S + ε·⟨S⟩` |
| `TwoPSFOperator` | Two-component PSF of Sinjan et al. (2026) |
| `SparseWarpOperator` | Reprojection onto the pixels of another instrument (sparse matrix) |
| `ComposeOperator` | Chains operators, e.g. reprojection then PSF |
| `WarpOperator` | Thin plate spline coordinate warping |
| `SpectralPSFOperator` | Spectral 1D convolution |

With `backend="cpp"`, the spectral PSF is better given to `MEProblem(spectralPSF=..., spectral_step=...)`: the synthesis
runs on a fine grid and a fixed matrix applies the profile, so the gradient stays a vector-Jacobian product. That path
applies the profile as a correlation, as pyMilne does; `SpectralPSFOperator` applies it as a convolution, which differs
for an asymmetric profile.

## Supported spectral lines

| ID | Element | λ₀ (Å) | g_eff | Instrument |
|----|---------|--------|-------|------------|
| 1 | Fe I | 6301.5012 | 1.67 | Hinode/SP |
| 2 | Fe I | 6302.4936 | 2.50 | Hinode/SP |
| 3 | Si I | 10827.0890 | 1.50 | |
| 4 | Ca I | 10833.4100 | 1.50 | |
| 5 | Ca I | 10838.9910 | 1.50 | |
| 6 | Fe I | 15648.5088 | 3.00 | |
| 7 | Fe I | 6173.3356 | 2.50 | SDO/HMI, SolO/PHI |

Lines are resolved by approximate wavelength: `MEProblem(line=6173)` → line ID 7.

## Project structure

```
neural_me/
├── external/pyMilne/           # C++ ME solver (submodule)
├── src/neural_me/
│   ├── core/
│   │   ├── atmospheres/        # per-pixel, neural-field and stereoscopic atmospheres
│   │   ├── nf.py               # neural fields: Fourier features, dense multiresolution grid (GridField)
│   │   ├── observation.py      # Observation (Stokes data container)
│   │   ├── problem.py          # MEProblem
│   │   ├── geometry.py         # local (heliographic) frame
│   │   ├── coregistration.py   # co-registration helpers (WCS)
│   │   ├── velocity.py         # velocity vector from two lines of sight
│   │   └── warp.py             # thin plate spline warping
│   ├── physics/
│   │   ├── milne_eddington.py       # ME forward model
│   │   ├── milne_eddington_torch.py # PyTorch ME forward model
│   │   ├── milne_eddington_cpp/     # C++ ME backend (built from external/pyMilne)
│   │   ├── lineas.py           # spectral line database
│   │   ├── zeeman.py           # Zeeman splitting
│   │   ├── wfa.py              # weak-field approximation
│   │   └── cog.py              # centre of gravity
│   ├── optimization/
│   │   ├── pixel_solver.py     # PixelSolver (Adam)
│   │   ├── levmar_solver.py    # LevMarSolver
│   │   ├── stereo_solver.py    # StereoSolver
│   │   ├── loss.py             # data terms
│   │   └── degradation.py      # DegradationOperator subclasses
│   ├── instruments/            # PSF and instrumental-profile resources
│   ├── regularization/
│   │   ├── spatial.py          # spatial smoothness
│   │   ├── nullspace.py        # null-space penalty of a degradation
│   │   ├── priors.py           # priors on the parameters
│   │   └── temporal.py         # temporal regularization
│   └── utils/                  # FITS I/O, plotting, miscellaneous
└── tests/                      # unit, integration, regression, benchmarks
```

## Citation

If you use Neural ME in your research, please cite the papers that describe it:

```bibtex
@article{DiazBaso2025_neuralwfa,
  author  = {{D{\'\i}az Baso}, C. J. and {Asensio Ramos}, A. and {de la Cruz Rodr{\'\i}guez}, J. and
             {da Silva Santos}, J. M. and {Rouppe van der Voort}, L.},
  title   = {Exploring spectropolarimetric inversions using neural fields: Solar chromospheric magnetic field
             under the weak-field approximation},
  journal = {Astronomy \& Astrophysics},
  year    = {2025},
  volume  = {693},
  pages   = {A170},
  doi     = {10.1051/0004-6361/202452172}
}
```

and Díaz Baso et al., *Stereoscopic Milne-Eddington inversions of SO/PHI-HRT and Hinode/SP observations using neural
fields* (in prep.). The C++ synthesis is that of pyMilne (de la Cruz Rodríguez 2019, A&A 631, A153).

## License

MIT License. See [LICENSE](LICENSE).

## Development

Contributions are welcome! Please open an issue or submit a pull request on GitHub. Run `pre-commit run --all-files`
before committing.
