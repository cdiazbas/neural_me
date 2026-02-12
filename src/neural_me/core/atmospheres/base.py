"""
Base Atmosphere Module.

Defines the `Atmosphere` abstract base class and `SlicedAtmosphere`.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


# --- Standardized Default Values (Single Source of Truth) ---
DEFAULT_NORMALIZATION = {
    # Magnetic field components (Gauss)
    "B": 500.0,  # Total magnetic field strength
    "Blos": 500.0,  # Line-of-sight component
    "BQ": 500.0,  # Stokes Q component
    "BU": 500.0,  # Stokes U component
    # Magnetic field angles (radians)
    "gamma": 1.0,  # Inclination angle [0, π]
    "xi": 1.0,  # Azimuth angle [0, 2π]
    # Thermodynamic parameters
    "vlos": 2.0,  # Line-of-sight velocity (km/s)
    "eta0": 100.0,  # Line-to-continuum absorption ratio
    "a": 0.2,  # Damping parameter
    "ddop": 1.0,  # Doppler width (Angstrom)
    # Source function parameters
    "S0": 1.0,  # Source function constant
    "S1": 1.0,  # Source function gradient
}

DEFAULT_PARAMETERS = {
    # Magnetic field
    "Blos": 500.0,
    "BQ": 500.0,
    "BU": 500.0,
    "B": 500.0,
    "gamma": 1.0,
    "xi": 1.0,
    # Thermodynamic
    "vlos": 0.0,
    "eta0": 50.0,
    "a": 0.2,
    "ddop": 0.02,
    # Source function
    "S0": 0.5,
    "S1": 0.5,
}


class Atmosphere(nn.Module, ABC):
    """
    Abstract base class for all Atmosphere models.

    An Atmosphere defines the physical state of the solar plasma for ME
    Synthesis. It is an nn.Module, so it can hold learnable parameters.
    """

    DEFAULT_NORMALIZATION = DEFAULT_NORMALIZATION
    DEFAULT_PARAMETERS = DEFAULT_PARAMETERS

    def __init__(self):
        super().__init__()

    # --- Abstract Physical Properties (Must be implemented by subclasses) ---
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

        # Wrap azimuth to [0, pi]
        phiB = stack[..., 2]
        phiB[phiB < 0] += np.pi
        phiB[phiB > np.pi] -= np.pi
        stack[..., 2] = phiB

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
    applying the stored indices.
    """

    def __init__(self, parent: Atmosphere, indices):
        super().__init__()
        # We don't register parent as a submodule to avoid recursion/duplication issues
        # if this view is stored (though typically it's ephemeral).
        # Actually, for an ephemeral object, standard assignment is fine.
        self.parent = parent
        self.indices = indices

    @property
    def B(self) -> torch.Tensor:
        return self.parent.B[self.indices]

    @property
    def gamma(self) -> torch.Tensor:
        return self.parent.gamma[self.indices]

    @property
    def xi(self) -> torch.Tensor:
        return self.parent.xi[self.indices]

    @property
    def vlos(self) -> torch.Tensor:
        return self.parent.vlos[self.indices]

    @property
    def eta0(self) -> torch.Tensor:
        return self.parent.eta0[self.indices]

    @property
    def a(self) -> torch.Tensor:
        return self.parent.a[self.indices]

    @property
    def ddop(self) -> torch.Tensor:
        return self.parent.ddop[self.indices]

    @property
    def S0(self) -> torch.Tensor:
        return self.parent.S0[self.indices]

    @property
    def S1(self) -> torch.Tensor:
        return self.parent.S1[self.indices]
