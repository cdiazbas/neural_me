"""
Neural ME Core Package.

Core components for Milne-Eddington inversions.
"""

from neural_me.core.observation import Observation
from neural_me.core.problem import MEProblem, MultiMEProblem
from neural_me.core.atmospheres import (
    Atmosphere,
    SlicedAtmosphere,
    CartesianAtmosphere,
    SphericalAtmosphere,
    SphericalSoftplusAtmosphere,
    StereoscopicAtmosphere,
)
from neural_me.core.atmospheres.base import (
    DEFAULT_NORMALIZATION,
    DEFAULT_PARAMETERS,
)

__all__ = [
    "Observation",
    "MEProblem",
    "MultiMEProblem",
    "Atmosphere",
    "SlicedAtmosphere",
    "CartesianAtmosphere",
    "SphericalAtmosphere",
    "SphericalSoftplusAtmosphere",
    "StereoscopicAtmosphere",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_PARAMETERS",
]
