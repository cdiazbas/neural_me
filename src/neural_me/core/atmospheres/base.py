"""
Base Atmosphere Module.

Defines the `Atmosphere` abstract base class and `SlicedAtmosphere`.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


# --- Standardized Default Values (Single Source of Truth) ---
DEFAULT_NORMALIZATION = {
    # Magnetic field components (Gauss)
    "B": 500.0,      # Total magnetic field strength
    "Blos": 500.0,   # Line-of-sight component
    "BQ": 500.0,     # Stokes Q component
    "BU": 500.0,     # Stokes U component
    # Magnetic field angles (radians)
    "gamma": 1.0,    # Inclination angle [0, π]
    "xi": 1.0,       # Azimuth angle [0, 2π]
    # Thermodynamic parameters
    "vlos": 2.0,     # Line-of-sight velocity (km/s)
    "eta0": 100.0,   # Line-to-continuum absorption ratio
    "a": 0.2,        # Damping parameter
    "ddop": 1.0,     # Doppler width (Angstrom)
    # Source function parameters
    "S0": 1.0,       # Source function constant
    "S1": 1.0,       # Source function gradient
}

DEFAULT_PARAMETERS = {
    # Magnetic field
    "Blos": 500.0,
    "BQ":   500.0,
    "BU":   500.0,
    "B":    500.0,
    "gamma": 1.0,
    "xi":    1.0,
    # Thermodynamic
    "vlos":  0.0,
    "eta0":  50.0,
    "a":     0.2,
    "ddop":  0.02,
    # Source function
    "S0": 0.5,
    "S1": 0.5,
}

# --- Physical Clamp Bounds (Single Source of Truth) ---
# All values are in PHYSICAL units (same as the parameter they bound).
# Subclasses derive their normalized clamp values by dividing by their norm dict.
#
# Format: { parameter_name: (min_physical, max_physical) }
#   None means no bound on that side (open interval).
DEFAULT_CLAMP_BOUNDS = {
    # Magnetic field [Gauss]
    "B":    (0.0,    3500.0),  # Total field: must be non-negative
    "Blos": (-3500.0, 3500.0), # LOS component: signed
    "BQ":   (-3500.0, 3500.0), # Q component: signed
    "BU":   (-3500.0, 3500.0), # U component: signed
    # Magnetic field angles [radians]
    "gamma": (0.0, math.pi),   # Inclination: [0, π]
    "xi":    (0.0, math.pi),   # Azimuth half-period: [0, π]
    # Thermodynamic
    "vlos":  (-5.0,  5.0),   # LOS velocity [km/s]
    "eta0":  (0.01,  600.0),   # Line-to-continuum ratio
    "a":     (0.001,   1.0),   # Voigt damping parameter
    "ddop":  (0.01,    0.2),   # Doppler width [Å]
    # Source function
    "S0":    (0.01,   1.5),    # Source function constant
    "S1":    (0.01,   1.5),    # Source function gradient
}


def half_atan2_azimuth(BU: torch.Tensor, BQ: torch.Tensor) -> torch.Tensor:
    """Azimuth ``0.5 * atan2(BU, BQ)`` with a finite gradient at zero transverse field.

    Where ``BQ = BU = 0`` the azimuth is undefined; it is returned as 0 with a
    zero gradient instead of the ~1e17 gradient of ``atan2(BU, BQ + eps)``.
    Elsewhere the value is that of ``0.5 * atan2(BU, BQ + 1e-15)``.
    """
    zero = (BQ * BQ + BU * BU) < 1e-24
    BU_safe = torch.where(zero, torch.zeros_like(BU), BU)
    BQ_safe = torch.where(zero, torch.ones_like(BQ), BQ + 1e-15)
    return 0.5 * torch.atan2(BU_safe, BQ_safe)


def _expand_stereo_thermo_dict(source: Dict[str, Any]) -> Dict[str, Any]:
    """Duplicate shared thermodynamic values for the two stereoscopic views."""
    return {
        f"thermo{idx}_{name}": source[name]
        for idx in (1, 2)
        for name in ("vlos", "eta0", "a", "ddop", "S0", "S1")
    }


DEFAULT_STEREO_NORMALIZATION = {
    **DEFAULT_NORMALIZATION,
    "Bn": DEFAULT_NORMALIZATION["Blos"],
    "Bw": DEFAULT_NORMALIZATION["Blos"],
    "global_Blos": DEFAULT_NORMALIZATION["Blos"],
    "global_Bn": DEFAULT_NORMALIZATION["Blos"],
    "global_Bw": DEFAULT_NORMALIZATION["Blos"],
    **_expand_stereo_thermo_dict(DEFAULT_NORMALIZATION),
}


DEFAULT_STEREO_PARAMETERS = {
    "global_Blos": 100.0,
    "global_Bn": 100.0,
    "global_Bw": 100.0,
    **_expand_stereo_thermo_dict(DEFAULT_PARAMETERS),
}


DEFAULT_STEREO_CLAMP_BOUNDS = {
    "global_Blos": DEFAULT_CLAMP_BOUNDS["Blos"],
    "global_Bn": DEFAULT_CLAMP_BOUNDS["Blos"],
    "global_Bw": DEFAULT_CLAMP_BOUNDS["Blos"],
    **_expand_stereo_thermo_dict(DEFAULT_CLAMP_BOUNDS),
}


class Atmosphere(nn.Module, ABC):
    """
    Abstract base class for all Atmosphere models.

    An Atmosphere defines the physical state of the solar plasma for ME
    Synthesis. It is an nn.Module, so it can hold learnable parameters.

    Physical parameter bounds are centralized in `DEFAULT_CLAMP_BOUNDS`
    (module-level dict). Subclasses should call `_clamp_normalized()` or
    `_clamp_physical()` inside their `clamp_parameters()` implementations
    rather than hardcoding numeric limits.
    """

    DEFAULT_NORMALIZATION = DEFAULT_NORMALIZATION
    DEFAULT_PARAMETERS    = DEFAULT_PARAMETERS
    DEFAULT_CLAMP_BOUNDS  = DEFAULT_CLAMP_BOUNDS
    DEFAULT_STEREO_NORMALIZATION = DEFAULT_STEREO_NORMALIZATION
    DEFAULT_STEREO_PARAMETERS    = DEFAULT_STEREO_PARAMETERS
    DEFAULT_STEREO_CLAMP_BOUNDS  = DEFAULT_STEREO_CLAMP_BOUNDS

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------ #
    #  Clamp helpers — use these in subclass clamp_parameters()
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inv_softplus_scalar(x: float, eps: float = 1e-7) -> float:
        """
        Numerically stable inverse of ``F.softplus`` for a **scalar** value.

        Returns ``y`` such that ``softplus(y) ≈ x``.

        Uses the identity: ``softplus⁻¹(x) = x + log(1 − exp(−x))``,
        which is equivalent to ``log(exp(x) − 1)`` but avoids overflow for
        large ``x`` by using ``log(-expm1(-x))``.

        Args:
            x:   The positive value to invert. Values ≤ 0 are clamped to ``eps``.
            eps: Minimum value before applying the formula (prevents log(0)).

        Returns:
            Scalar float: the pre-softplus raw value corresponding to ``x``.

        Example::

            hi_raw = Atmosphere._inv_softplus_scalar(hi_phys / norm)
            self._eta0_raw.clamp_(max=hi_raw)
        """
        x = max(x, eps)
        return x + math.log(-math.expm1(-x))

    def _clamp_bounds(self, name: str, norm: Optional[float] = None
                      ) -> Tuple[Optional[float], Optional[float]]:
        """
        Return the ``(min, max)`` clamp bounds for parameter *name* scaled to
        the internal storage space.

        For **linearly normalized** parameters (``stored = physical / norm``),
        pass ``norm`` and apply the returned tuple directly::

            lo, hi = self._clamp_bounds("vlos", self.norm["vlos"])
            self._vlos.clamp_(lo, hi)

        For **softplus pre-activation** parameters, use this to get the
        normalized bounds, then pass each through
        :meth:`_inv_softplus_scalar` to reach raw space::

            lo_norm, hi_norm = self._clamp_bounds("eta0", self.norm["eta0"])
            hi_raw = self._inv_softplus_scalar(hi_norm)
            self._eta0_raw.clamp_(max=hi_raw)   # lower bound not needed (softplus > 0)

        Args:
            name: Parameter name — must be a key in :attr:`DEFAULT_CLAMP_BOUNDS`.
            norm: Normalization factor. Returns ``(lo_phys/norm, hi_phys/norm)``.
                  Pass ``None`` or ``1.0`` to get raw physical bounds.

        Returns:
            ``(lo, hi)`` as floats.
        """
        lo_phys, hi_phys = self.DEFAULT_CLAMP_BOUNDS[name]
        if norm is None or norm == 1.0:
            return lo_phys, hi_phys
        return lo_phys / norm, hi_phys / norm

    # ------------------------------------------------------------------ #
    #  Abstract Physical Properties (Must be implemented by subclasses)
    # ------------------------------------------------------------------ #
    # All properties must return tensors in PHYSICAL units.

    @property
    @abstractmethod
    def B(self) -> torch.Tensor:
        """
        Magnetic Field Strength [Gauss]
        """

    @property
    @abstractmethod
    def gamma(self) -> torch.Tensor:
        """
        Inclination [rad]
        """

    @property
    @abstractmethod
    def xi(self) -> torch.Tensor:
        """
        Azimuth [rad]
        """

    @property
    @abstractmethod
    def vlos(self) -> torch.Tensor:
        """
        Line-of-Sight Velocity [km/s]
        """

    @property
    @abstractmethod
    def eta0(self) -> torch.Tensor:
        """
        Line-to-Continuum Absorption Ratio []
        """

    @property
    @abstractmethod
    def a(self) -> torch.Tensor:
        """
        Damping Parameter []
        """

    @property
    @abstractmethod
    def ddop(self) -> torch.Tensor:
        """
        Doppler Width [Angstrom]
        """

    @property
    @abstractmethod
    def S0(self) -> torch.Tensor:
        """
        Source Function Constant []
        """

    @property
    @abstractmethod
    def S1(self) -> torch.Tensor:
        """
        Source Function Gradient []
        """

    # --- Common Methods ---

    def to_stack(self) -> torch.Tensor:
        """
        Returns the stack of ALL physical parameters.

        Shape: (..., 9)
        Order: [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
        """
        return torch.stack(
            [
                self.B,
                self.gamma,
                self.xi,
                self.vlos,
                self.eta0,
                self.a,
                self.ddop,
                self.S0,
                self.S1,
            ],
            dim=-1,
        )

    def to_numpy_stack(self) -> np.ndarray:
        """
        Returns the stack of physical parameters as a numpy array. Moves to CPU
        and detaches graph.

        Wraps azimuth (index 2) to [0, pi].
        """
        stack = self.to_stack().detach().cpu().numpy()

        # Wrap azimuth robustly to [0, π) for any number of periods
        stack[..., 2] = np.remainder(stack[..., 2], np.pi)

        return stack

    def __getitem__(self, indices) -> "SlicedAtmosphere":
        """
        Returns a sliced view of this Atmosphere.

        This enables efficient batch processing (e.g., loss computation
        on a subset) without copying the underlying parameters or
        breaking the computation graph.
        """
        return SlicedAtmosphere(self, indices)

    def set_parameters(self, params: Dict[str, Any]):
        """
        Optional method to set parameters from external values.

        Subclasses should implement this if they support initialization
        from data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_parameters."
        )

    def clamp_parameters(self):
        """
        Optional method to enforce physical constraints.

        Subclasses likely need to override this.
        """


class SlicedAtmosphere(Atmosphere):
    """
    A lightweight view into another Atmosphere, filtered by indices.

    This class delegates all property access to the parent Atmosphere,
    applying the stored indices to the flattened pixel axis, the layout of
    ``Observation`` data, so a 2-D grid is indexed by pixel, not by row.
    """

    def _take(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(-1)[self.indices]

    def __init__(self, parent: Atmosphere, indices):
        super().__init__()
        # We don't register parent as a submodule to avoid recursion/duplication issues
        # if this view is stored (though typically it's ephemeral).
        # Actually, for an ephemeral object, standard assignment is fine.
        self.parent = parent
        self.indices = indices

    @property
    def B(self) -> torch.Tensor:
        return self._take(self.parent.B)

    @property
    def gamma(self) -> torch.Tensor:
        return self._take(self.parent.gamma)

    @property
    def xi(self) -> torch.Tensor:
        return self._take(self.parent.xi)

    @property
    def vlos(self) -> torch.Tensor:
        return self._take(self.parent.vlos)

    @property
    def eta0(self) -> torch.Tensor:
        return self._take(self.parent.eta0)

    @property
    def a(self) -> torch.Tensor:
        return self._take(self.parent.a)

    @property
    def ddop(self) -> torch.Tensor:
        return self._take(self.parent.ddop)

    @property
    def S0(self) -> torch.Tensor:
        return self._take(self.parent.S0)

    @property
    def S1(self) -> torch.Tensor:
        return self._take(self.parent.S1)
