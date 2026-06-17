"""
Spherical Atmosphere Implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .base import Atmosphere


class SphericalAtmosphere(Atmosphere):
    """
    Atmosphere parameterized directly by Spherical coordinates (B, gamma, xi).
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        mask: Optional[torch.Tensor] = None,
        norm: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
        init_type: str = "default",
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")

        if mask is not None:
            self.n_elements = mask.sum().item()
            self.is_masked = True
            self.register_buffer("mask", mask)
        else:
            self.n_elements = int(np.prod(grid_shape))
            self.is_masked = False

        self.norm = self.DEFAULT_NORMALIZATION.copy()
        if norm:
            self.norm.update(norm)

        self.defaults = self.DEFAULT_PARAMETERS.copy()

        self._init_params(init_type)

    def _init_params(self, init_type: str):
        """
        Allocate and initialize all learnable parameters in normalized space.

        Args:
            init_type: ``'default'`` uses :attr:`DEFAULT_PARAMETERS` divided by
                       norm. ``'zeros'`` initializes all normalized params to 0.
        """
        def make_param(val):
            return nn.Parameter(torch.ones(self.n_elements, device=self.device) * val)

        d = self.defaults.copy()
        if init_type == "zeros":
            for k in d:
                d[k] = 0.0

        self._B    = make_param(d["B"]    / self.norm["B"])
        self._gamma= make_param(d["gamma"]/ self.norm["gamma"])
        self._xi   = make_param(d["xi"]   / self.norm["xi"])
        self._vlos = make_param(d["vlos"] / self.norm["vlos"])
        self._eta0 = make_param(d["eta0"] / self.norm["eta0"])
        self._a    = make_param(d["a"]    / self.norm["a"])
        self._ddop = make_param(d["ddop"] / self.norm["ddop"])
        self._S0   = make_param(d["S0"]   / self.norm["S0"])
        self._S1   = make_param(d["S1"]   / self.norm["S1"])

    def _expand(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape flat tensor to ``self.grid_shape`` (no-op when masked)."""
        if self.is_masked:
            return tensor
        return tensor.view(*self.grid_shape)

    @property
    def B(self):
        return self._expand(self._B * self.norm["B"])

    @property
    def gamma(self):
        return self._expand(self._gamma * self.norm["gamma"])

    @property
    def xi(self):
        return self._expand(self._xi * self.norm["xi"])

    @property
    def vlos(self):
        return self._expand(self._vlos * self.norm["vlos"])

    @property
    def eta0(self):
        return self._expand(self._eta0 * self.norm["eta0"])

    @property
    def a(self):
        return self._expand(self._a * self.norm["a"])

    @property
    def ddop(self):
        return self._expand(self._ddop * self.norm["ddop"])

    @property
    def S0(self):
        return self._expand(self._S0 * self.norm["S0"])

    @property
    def S1(self):
        return self._expand(self._S1 * self.norm["S1"])

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from a dict of physical values."""
        with torch.no_grad():
            for k, v in params.items():
                if isinstance(v, (np.ndarray, float, int)):
                    t = torch.as_tensor(
                        v, dtype=torch.float32, device=self.device
                    ).flatten()
                elif isinstance(v, torch.Tensor):
                    t = v.to(self.device).flatten()
                else:
                    continue

                if k == "B":
                    self._B.copy_(t / self.norm["B"])
                elif k == "gamma":
                    self._gamma.copy_(t / self.norm["gamma"])
                elif k == "xi":
                    self._xi.copy_(t / self.norm["xi"])
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

        Parameters are stored as ``_param = physical / norm``, so the clamp
        bounds are divided by the same norm factor before applying.
        All physical bounds are defined in :attr:`DEFAULT_CLAMP_BOUNDS`.
        """
        with torch.no_grad():
            lo, hi = self._clamp_bounds("B", self.norm["B"])
            self._B.clamp_(lo, hi)

            lo, hi = self._clamp_bounds("gamma", self.norm["gamma"])
            self._gamma.clamp_(lo, hi)

            # xi is periodic — wrap, don't clamp
            # (handled by to_numpy_stack via np.remainder)

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

