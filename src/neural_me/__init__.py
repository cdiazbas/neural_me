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
    ParameterModel,
    CartesianParameterModel,
    StereoscopicParameterModel,
)

# Physics components
from neural_me.physics.milne_eddington import MilneEddington

# Optimization
from neural_me.optimization.pixel_solver import PixelSolver

__version__ = "0.1.0"

__all__ = [
    # Core
    'Atmosphere',
    'Observation', 
    'MEProblem',
    'MultiMEProblem',
    'ParameterModel',
    'CartesianParameterModel',
    'StereoscopicParameterModel',
    # Physics
    'MilneEddington',
    # Optimization
    'PixelSolver',
]
