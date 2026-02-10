"""
Neural ME: Milne-Eddington Spectropolarimetric Inversion Library.

A PyTorch-based framework for efficient and differentiable
Milne-Eddington inversions of spectropolarimetric data.
"""

# Core components
from neural_me.core import (
    Atmosphere,
    Observation,
    MEProblem,
    MultiMEProblem,
    CartesianAtmosphere,
    StereoscopicAtmosphere,
    SphericalSoftplusAtmosphere,

)

# Physics components
from neural_me.physics.milne_eddington import MilneEddington

# Optimization
from neural_me.optimization.pixel_solver import PixelSolver

__version__ = "0.1.0"

# Auto-apply plot parameters on import (graceful fallback)
try:
    from neural_me.utils.viz import set_plot_params
    set_plot_params()
except ImportError:
    pass  # matplotlib not available, skip styling

__all__ = [
    # Core
    'Atmosphere',
    'Observation', 
    'MEProblem',
    'MultiMEProblem',
    'CartesianAtmosphere',
    'StereoscopicAtmosphere',
    'SphericalSoftplusAtmosphere',
    # Physics
    'MilneEddington',
    # Optimization
    'PixelSolver',
]
