"""
Spherical Softplus Atmosphere implementation.

Uses Softplus activation to enforce non-negative constraints on physical
parameters, providing smoother gradients than hard clamping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

from .base import Atmosphere


class SphericalSoftplusAtmosphere(Atmosphere):
    """
    Atmosphere parameterized in Spherical coordinates with Softplus
    constraints.

    Parameters:
        - B_raw: -> softplus -> B
        - gamma: -> gamma_raw -> sigmoid * pi
        - xi: -> xi
        - vlos: -> vlos
        - eta0_raw: -> softplus -> eta0
        - a_raw: -> softplus -> a
        - ddop_raw: -> softplus -> ddop
        - S0_raw: -> softplus -> S0
        - S1_raw: -> softplus -> S1
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        norm: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
        initial_values: Optional[Dict[str, Any]] = None,
        requires_grad: bool = True,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")

        # Normalization
        self.norm = self.DEFAULT_NORMALIZATION.copy()
        if norm:
            self.norm.update(norm)

        self.n_elements = int(np.prod(self.grid_shape))
        self.n_pixels_shape = self.grid_shape

        # Register raw parameters
        self._B_raw = self._create_parameter(0.0, requires_grad)
        self._gamma_raw = self._create_parameter(0.0, requires_grad)
        self._xi = self._create_parameter(0.0, requires_grad)
        self._vlos = self._create_parameter(0.0, requires_grad)
        self._eta0_raw = self._create_parameter(0.0, requires_grad)
        self._a_raw = self._create_parameter(0.0, requires_grad)
        self._ddop_raw = self._create_parameter(0.0, requires_grad)
        self._S0_raw = self._create_parameter(0.0, requires_grad)
        self._S1_raw = self._create_parameter(0.0, requires_grad)

        # Initialize
        self._init_parameters(initial_values)

    def _create_parameter(self, value: float, requires_grad: bool) -> nn.Parameter:
        # Create tensor of shape grid_shape
        t = torch.full(
            self.n_pixels_shape, value, device=self.device, dtype=torch.float32
        )
        return nn.Parameter(t, requires_grad=requires_grad)

    def _to_tensor(self, x: Union[float, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Helper to convert input to tensor on device.
        """
        if isinstance(x, (float, int)):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device).to(dtype=torch.float32)
        return x

    def _inv_softplus(
        self, x: Union[float, np.ndarray, torch.Tensor], epsilon=1e-6
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Inverse softplus: log(exp(x) - 1)."""
        # Ensure x is large enough to avoid log(neg)
        if isinstance(x, (int, float)):
            x = max(x, epsilon)
            return np.log(np.exp(x) - 1.0)
        elif isinstance(x, np.ndarray):
            x = np.maximum(x, epsilon)
            return np.log(np.exp(x) - 1.0)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, min=epsilon)
            return torch.log(torch.exp(x) - 1.0)
        return x

    def _inv_sigmoid(
        self, x: Union[float, np.ndarray, torch.Tensor], epsilon=1e-6
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Inverse sigmoid (logit): log(x / (1 - x))."""
        # Clip x to [epsilon, 1-epsilon] to avoid infinity
        if isinstance(x, (int, float)):
            x = np.clip(x, epsilon, 1.0 - epsilon)
            return np.log(x / (1.0 - x))
        elif isinstance(x, np.ndarray):
            x = np.clip(x, epsilon, 1.0 - epsilon)
            return np.log(x / (1.0 - x))
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, min=epsilon, max=1.0 - epsilon)
            return torch.log(x / (1.0 - x))
        return x

    def _init_parameters(self, initial_values: Optional[Dict[str, Any]]):
        """
        Initialize parameters, converting physical values to raw storage.
        """

    def _init_parameters(self, initial_values: Optional[Dict[str, Any]]):
        """
        Initialize parameters, converting physical values to raw storage.
        """
        defaults = self.DEFAULT_PARAMETERS.copy()

        if initial_values:
            for k, v in initial_values.items():
                if k in defaults:
                    defaults[k] = v

        with torch.no_grad():
            # Softplus parameters: store inv_softplus(val / norm)
            self._B_raw.copy_(
                self._to_tensor(self._inv_softplus(defaults["B"] / self.norm["B"]))
            )
            self._eta0_raw.copy_(
                self._to_tensor(
                    self._inv_softplus(defaults["eta0"] / self.norm["eta0"])
                )
            )
            self._a_raw.copy_(
                self._to_tensor(self._inv_softplus(defaults["a"] / self.norm["a"]))
            )
            self._ddop_raw.copy_(
                self._to_tensor(
                    self._inv_softplus(defaults["ddop"] / self.norm["ddop"])
                )
            )
            self._S0_raw.copy_(
                self._to_tensor(self._inv_softplus(defaults["S0"] / self.norm["S0"]))
            )
            self._S1_raw.copy_(
                self._to_tensor(self._inv_softplus(defaults["S1"] / self.norm["S1"]))
            )

            # Linear parameters
            # gamma is now sigmoid-constrained to [0, pi]
            # normalized gamma range is [0, pi] / norm['gamma'] (usually norm=1 -> [0, pi])
            # We map (val/norm) -> [0, 1] relative to pi, then inv_sigmoid
            gamma_val = defaults["gamma"] / self.norm["gamma"]
            gamma_fraction = gamma_val / np.pi  # Expected to be in [0, 1]
            self._gamma_raw.copy_(self._to_tensor(self._inv_sigmoid(gamma_fraction)))

            self._xi.copy_(self._to_tensor(defaults["xi"] / self.norm["xi"]))
            self._vlos.copy_(self._to_tensor(defaults["vlos"] / self.norm["vlos"]))

    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameters from physical values.
        """
        with torch.no_grad():
            for k, v in params.items():
                t = self._to_tensor(v)

                # Handle softplus params
                if k == "B":
                    self._B_raw.copy_(self._inv_softplus(t / self.norm["B"]))
                elif k == "eta0":
                    self._eta0_raw.copy_(self._inv_softplus(t / self.norm["eta0"]))
                elif k == "a":
                    self._a_raw.copy_(self._inv_softplus(t / self.norm["a"]))
                elif k == "ddop":
                    self._ddop_raw.copy_(self._inv_softplus(t / self.norm["ddop"]))
                elif k == "S0":
                    self._S0_raw.copy_(self._inv_softplus(t / self.norm["S0"]))
                elif k == "S1":
                    self._S1_raw.copy_(self._inv_softplus(t / self.norm["S1"]))

                # Handle linear params
                # gamma is special: sigmoid
                elif k == "gamma":
                    val = t / self.norm["gamma"]
                    self._gamma_raw.copy_(self._inv_sigmoid(val / np.pi))

                elif k == "xi":
                    self._xi.copy_(t / self.norm["xi"])
                elif k == "vlos":
                    self._vlos.copy_(t / self.norm["vlos"])

    def clamp_parameters(self):
        """
        Softplus handles the lower bound (0).

        We might want upper bounds.
        """
        pass  # Generally not needed for softplus, but maybe for stability?

    # --- Properties ---
    @property
    def B(self):
        return F.softplus(self._B_raw) * self.norm["B"]

    @property
    def gamma(self):
        # gamma in [0, pi]
        return (torch.sigmoid(self._gamma_raw) * np.pi) * self.norm["gamma"]

    @property
    def xi(self):
        return self._xi * self.norm["xi"]

    @property
    def vlos(self):
        return self._vlos * self.norm["vlos"]

    @property
    def eta0(self):
        return F.softplus(self._eta0_raw) * self.norm["eta0"]

    @property
    def a(self):
        return F.softplus(self._a_raw) * self.norm["a"]

    @property
    def ddop(self):
        return F.softplus(self._ddop_raw) * self.norm["ddop"]

    @property
    def S0(self):
        return F.softplus(self._S0_raw) * self.norm["S0"]

    @property
    def S1(self):
        return F.softplus(self._S1_raw) * self.norm["S1"]
