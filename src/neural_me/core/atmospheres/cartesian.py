"""
Cartesian Atmosphere Implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .base import Atmosphere, half_atan2_azimuth


class CartesianAtmosphere(Atmosphere):
    """
    Atmosphere parameterized by Cartesian Magnetic Field components (Blos, BQ,
    BU).

    This avoids periodicity issues with angular variables (gamma, xi) during optimization.

    Internal State:
        Normalized parameters: _Blos, _BQ, _BU, _vlos, _eta0, ...
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        mask: Optional[torch.Tensor] = None,
        norm: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
        init_type: str = "default",
    ):
        """
        Args:
            grid_shape: Shape of the spatial grid (e.g., (Ny, Nx)).
            mask: Optional boolean mask. If provided, parameters are only allocated
                  for True pixels (flattened). Useful for active regions.
            norm: Custom normalization factors.
            device: Torch device.
            init_type: Initialization strategy ('default', 'zeros'). The default 'default' uses the same physical defaults as the base Atmosphere, but normalized. 'zeros' initializes all parameters to zero (after normalization). The default values are defined in the base Atmosphere class.
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")

        # Determine number of pixels/elements
        if mask is not None:
            self.n_elements = mask.sum().item()
            self.is_masked = True
            self.register_buffer("mask", mask)
        else:
            self.n_elements = int(np.prod(grid_shape))
            self.is_masked = False

        # Normalization
        self.norm = self.DEFAULT_NORMALIZATION.copy()
        if norm:
            self.norm.update(norm)

        # Defaults (Physical Units)
        # Defaults (Inherited)
        self.defaults = self.DEFAULT_PARAMETERS.copy()

        # Allocate Parameters
        self._init_params(init_type)
        print(f"CartesianAtmosphere initialized | Grid: {self.grid_shape} | Masked: {self.is_masked}")

    def _init_params(self, init_type: str):
        """
        Allocate and initialize all learnable parameters.

        Parameters are stored in **normalized** form (physical / norm) to keep
        gradient magnitudes comparable across parameters with different physical
        scales.  The normalization factors live in ``self.norm``.

        Args:
            init_type: ``'default'`` — use the physical defaults from
                       :attr:`DEFAULT_PARAMETERS` divided by their norms.
                       ``'zeros'`` — initialize all normalized params to 0.
        """
        def make_param(val):
            return nn.Parameter(torch.ones(self.n_elements, device=self.device) * val)

        d = self.defaults.copy()
        if init_type == "zeros":
            for k in d:
                d[k] = 0.0

        self._Blos = make_param(d["Blos"] / self.norm["Blos"])
        self._BQ   = make_param(d["BQ"]   / self.norm["BQ"])
        self._BU   = make_param(d["BU"]   / self.norm["BU"])
        self._vlos = make_param(d["vlos"] / self.norm["vlos"])
        self._eta0 = make_param(d["eta0"] / self.norm["eta0"])
        self._a    = make_param(d["a"]    / self.norm["a"])
        self._ddop = make_param(d["ddop"] / self.norm["ddop"])
        self._S0   = make_param(d["S0"]   / self.norm["S0"])
        self._S1   = make_param(d["S1"]   / self.norm["S1"])

    def _expand(self, tensor):
        """
        Reshape a flat parameter tensor to ``self.grid_shape``.

        For masked atmospheres the flat layout is preserved (reshaping would
        require zero-filling the masked pixels, which is deferred to callers).
        """
        if self.is_masked:
            return tensor
        return tensor.view(*self.grid_shape)

    # --- Properties ---

    @property
    def B(self) -> torch.Tensor:
        return self._expand(self._field()[3])

    def _field(self):
        """Physical ``(Blos, BQ, BU, B)``; one place for the shared B."""
        Blos = self._Blos * self.norm["Blos"]
        BQ   = self._BQ   * self.norm["BQ"]
        BU   = self._BU   * self.norm["BU"]
        # Add eps inside sqrt to keep gradient finite when B→0
        B = torch.sqrt(Blos**2 + BQ**2 + BU**2 + 1e-8)
        return Blos, BQ, BU, B

    @staticmethod
    def _gamma_from(Blos: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Clamp ratio away from ±1 to keep acos gradient finite
        ratio = torch.clamp(Blos / B, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(ratio)

    @property
    def gamma(self) -> torch.Tensor:
        Blos, _, _, B = self._field()
        return self._expand(self._gamma_from(Blos, B))

    def to_stack(self) -> torch.Tensor:
        """Stack of the 9 physical parameters, computing B once."""
        Blos, BQ, BU, B = self._field()
        return torch.stack(
            [
                self._expand(B),
                self._expand(self._gamma_from(Blos, B)),
                self._expand(half_atan2_azimuth(BU, BQ)),
                self.vlos, self.eta0, self.a, self.ddop, self.S0, self.S1,
            ],
            dim=-1,
        )

    @property
    def xi(self) -> torch.Tensor:
        BQ = self._BQ * self.norm["BQ"]
        BU = self._BU * self.norm["BU"]
        # xi = 0.5 * atan2(BU, BQ) → result in [-π/2, π/2]
        return self._expand(half_atan2_azimuth(BU, BQ))

    @property
    def vlos(self) -> torch.Tensor:
        return self._expand(self._vlos * self.norm["vlos"])

    @property
    def eta0(self) -> torch.Tensor:
        return self._expand(self._eta0 * self.norm["eta0"])

    @property
    def a(self) -> torch.Tensor:
        return self._expand(self._a * self.norm["a"])

    @property
    def ddop(self) -> torch.Tensor:
        return self._expand(self._ddop * self.norm["ddop"])

    @property
    def S0(self) -> torch.Tensor:
        return self._expand(self._S0 * self.norm["S0"])

    @property
    def S1(self) -> torch.Tensor:
        return self._expand(self._S1 * self.norm["S1"])

    # --- Methods ---

    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameters from physical values.
        """
        with torch.no_grad():
            for k, v in params.items():
                # Value 'v' is physical
                if isinstance(v, (np.ndarray, float, int)):
                    t = torch.as_tensor(
                        v, dtype=torch.float32, device=self.device
                    ).flatten()
                elif isinstance(v, torch.Tensor):
                    t = v.to(self.device).flatten()
                else:
                    continue

                # Assign to correct internal parameter
                # Cartesian inputs
                if k == "Blos":
                    self._Blos.copy_(t / self.norm["Blos"])
                elif k == "BQ":
                    self._BQ.copy_(t / self.norm["BQ"])
                elif k == "BU":
                    self._BU.copy_(t / self.norm["BU"])
                # Thermo inputs
                elif k == "vlos":
                    self._vlos.copy_(t / self.norm["vlos"])
                elif k == "eta0":
                    self._eta0.copy_(t / self.norm["eta0"])
                elif k == "a":
                    self._a.copy_(t / self.norm["a"])
                elif k == "ddop":
                    self._ddop.copy_(t / self.norm["ddop"])
                elif k == "S0":
                    self._S0.copy_(t / self.norm["S0"])
                elif k == "S1":
                    self._S1.copy_(t / self.norm["S1"])

    def clamp_parameters(self):
        """
        Enforce physical parameter bounds on the normalized internal tensors.

        Bounds are derived from :attr:`DEFAULT_CLAMP_BOUNDS` divided by the
        per-parameter normalization factor, so the clamp is applied in
        **normalized** space (i.e. on the raw ``nn.Parameter`` values).
        """
        with torch.no_grad():
            # Cartesian B components — signed, symmetric bounds
            lo, hi = self._clamp_bounds("Blos", self.norm["Blos"])
            self._Blos.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("BQ", self.norm["BQ"])
            self._BQ.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("BU", self.norm["BU"])
            self._BU.clamp_(lo, hi)

            lo, hi = self._clamp_bounds("vlos", self.norm["vlos"])
            self._vlos.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("eta0", self.norm["eta0"])
            self._eta0.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("a", self.norm["a"])
            self._a.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("ddop", self.norm["ddop"])
            self._ddop.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("S0", self.norm["S0"])
            self._S0.clamp_(lo, hi)
            lo, hi = self._clamp_bounds("S1", self.norm["S1"])
            self._S1.clamp_(lo, hi)

