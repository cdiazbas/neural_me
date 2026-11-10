"""Priors on single physical parameters."""

import math
from typing import Optional

import torch


def log_prior(x: torch.Tensor, center: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean of (log x - log center)^2: a scale-free pull of a positive parameter
    towards ``center``.

    Used on eta0 of the 6173 line, whose line strength is degenerate with the
    Doppler width (eta0 * ddop^2 is what the profile fixes): without a prior the
    fitted eta0 depends on the start. Centeno et al. (2014, Sol. Phys. 289,
    3531, Sect. 2.4) add a linear penalty 0.002 (eta0 - 5)^2 in the HMI
    inversion; the log form weighs a factor 2 the same at any eta0, which suits
    a parameter spanning 10-600. With the weight 1e-3 and center 50 it removed
    the start dependence of PHI-alone fits (analysis step 12).

    Args:
        x: positive tensor (e.g. ``atmosphere.eta0``).
        center: the value the prior pulls towards (> 0).
        mask: optional boolean tensor of the shape of ``x``; the mean runs over it.
    """
    if center <= 0:
        raise ValueError("center must be > 0")
    d = (torch.log(x) - math.log(center)) ** 2
    return d[mask].mean() if mask is not None else d.mean()
