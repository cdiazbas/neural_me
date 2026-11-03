"""
Neural ME Core Package.

Core components for Milne-Eddington inversions.
"""

from neural_me.core.observation import Observation
from neural_me.core.problem import MEProblem
from neural_me.core.geometry import (
    heliocentric_angles,
    solar_north_axis,
    wcs_roll_angle,
    local_vertical_matrix,
    local_vertical_matrix_from_map,
    apply_local_vertical,
)
from neural_me.core.atmospheres import (
    Atmosphere,
    SlicedAtmosphere,
    CartesianAtmosphere,
    SphericalAtmosphere,
    SphericalSoftplusAtmosphere,
    StereoscopicAtmosphere,
    NFAtmosphere,
    StereoscopicNFAtmosphere,
    LocalVerticalMixin,
)
from neural_me.core.atmospheres.base import (
    DEFAULT_NORMALIZATION,
    DEFAULT_PARAMETERS,
)
from neural_me.core.warp import (
    align_images,
    differentiable_tps_warp,
    extrapolate_points,
    get_points_src,
)

__all__ = [
    "Observation",
    "MEProblem",
    "Atmosphere",
    "SlicedAtmosphere",
    "CartesianAtmosphere",
    "SphericalAtmosphere",
    "SphericalSoftplusAtmosphere",
    "StereoscopicAtmosphere",
    "NFAtmosphere",
    "StereoscopicNFAtmosphere",
    "LocalVerticalMixin",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_PARAMETERS",
    "heliocentric_angles",
    "solar_north_axis",
    "wcs_roll_angle",
    "local_vertical_matrix",
    "local_vertical_matrix_from_map",
    "apply_local_vertical",
]

