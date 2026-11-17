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
    instrument_sampling_grid,
)
from neural_me.core.velocity import (
    view_los_vectors,
    unseen_direction,
    zero_at_umbral_core,
    field_aligned_velocity,
    solve_velocity,
    fit_velocity_field,
)
from neural_me.core.coregistration import (
    pointing_map,
    instrument_map,
    model_grid_wcs,
    sampling_grid_from_wcs,
    sampling_grid_from_images,
    save_sampling_grid,
    reproject_to_model_grid,
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
    "instrument_sampling_grid",
    "pointing_map",
    "instrument_map",
    "model_grid_wcs",
    "sampling_grid_from_wcs",
    "sampling_grid_from_images",
    "save_sampling_grid",
    "reproject_to_model_grid",
    "view_los_vectors",
    "unseen_direction",
    "zero_at_umbral_core",
    "field_aligned_velocity",
    "solve_velocity",
    "fit_velocity_field",
]

