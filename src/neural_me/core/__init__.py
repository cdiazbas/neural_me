"""
Neural ME Core Package.

Core components for Milne-Eddington inversions.
"""

from neural_me.core.atmosphere import Atmosphere
from neural_me.core.observation import Observation
from neural_me.core.problem import MEProblem, MultiMEProblem
from neural_me.core.parameters import (
    ParameterModel,
    CartesianParameterModel,
    StereoscopicParameterModel,
)

__all__ = [
    'Atmosphere',
    'Observation',
    'MEProblem',
    'MultiMEProblem',
    'ParameterModel',
    'CartesianParameterModel',
    'StereoscopicParameterModel',
]
