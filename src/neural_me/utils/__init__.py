"""
Neural ME Utilities Package.

Provides I/O, visualization, and miscellaneous helper functions.
"""

# Core utilities (always available)
from neural_me.utils.io import readfits, writefits
from neural_me.utils.misc import (
    get_free_gpu,
    format_scientific,
    even_shape,
    AttributeDict,
    print_summary,
)


# Visualization utilities (lazy import to avoid matplotlib issues in some environments)
def __getattr__(name):
    """
    Lazy import for viz module to handle matplotlib availability.
    """
    viz_exports = [
        "set_plot_params",
        "set_params",
        "LEGACY_PARAMS",
        "torch2numpy",
        "add_colorbar",
        "plot_loss",
        "plot_training_progress",
        "plot_wfa_results",
        "plot_stokes_profiles",
        "plot_me_results",
        "plot_chi2_maps",
        "plot_uncertainties",
    ]
    if name in viz_exports:
        from neural_me.utils import viz

        return getattr(viz, name)
    raise AttributeError(f"module 'neural_me.utils' has no attribute '{name}'")


__all__ = [
    # I/O
    "readfits",
    "writefits",
    # Misc
    "get_free_gpu",
    "format_scientific",
    "even_shape",
    "AttributeDict",
    "print_summary",
    # Viz (lazy loaded)
    "set_plot_params",
    "set_params",
    "LEGACY_PARAMS",
    "torch2numpy",
    "add_colorbar",
    "plot_loss",
    "plot_training_progress",
    "plot_wfa_results",
    "plot_stokes_profiles",
    "plot_me_results",
    "plot_chi2_maps",
    "plot_uncertainties",
]
