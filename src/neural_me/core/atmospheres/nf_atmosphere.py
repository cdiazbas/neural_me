"""
Single-View NF (Neural Field) Atmosphere.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn

from .base import Atmosphere, half_atan2_azimuth
from .nf_heads import ThermoHead
from ..nf import MultiMLP, to_coordinates


class NFAtmosphere(Atmosphere):
    """
    Neural Field (NF) Atmosphere for single-view observations.
    
    Instead of optimizing pixel grids directly, this model houses a neural network (MultiMLP)
    that predicts the physical (or Cartesian) parameters from normalized spatial coordinates (y, x).
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        mask: Optional[torch.Tensor] = None,
        norm: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
        num_resnet_blocks: int = 3,
        num_layers_per_block: int = 2,
        dim_hidden: int = 64,
        activation: nn.Module = nn.GELU(),
        fourier_features: bool = True,
        m_freqs: int = 512,
        sigma: Union[float, List[float], List[List[float]]] = 100.0,
        tune_beta: bool = False,
        init: Optional[Dict[str, float]] = None,
        bounds: Optional[str] = None,
    ):
        """
        ``init``: starting values of the thermodynamic parameters (any of vlos,
        eta0, a, ddop, S0), reached when the network outputs 0, e.g. the medians
        of a pixel-wise inversion. ``bounds="clamp"``: range of vlos, eta0, a and
        ddop from DEFAULT_CLAMP_BOUNDS instead of the original ones. See
        :mod:`neural_me.core.atmospheres.nf_heads`.
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")

        # Normalization
        self.norm = self.DEFAULT_NORMALIZATION.copy()
        if norm:
            self.norm.update(norm)
        self.thermo_head = ThermoHead(self.norm, self.DEFAULT_CLAMP_BOUNDS, init=init, bounds=bounds)

        # Handle mask
        if mask is not None:
            self.is_masked = True
            self.register_buffer("mask", mask)
            self.n_elements = mask.sum().item()
        else:
            self.is_masked = False
            self.n_elements = int(np.prod(grid_shape))

        # Generate coordinates
        coords_all = to_coordinates(grid_shape, device=self.device)
        if self.is_masked:
            # Mask is assumed flat or matches grid_shape
            self.register_buffer("coords", coords_all[mask.flatten()])
        else:
            self.register_buffer("coords", coords_all)

        # Define NF (8 parameters predicted: Blos, BQ, BU, vlos, eta0, a, ddop, S0)
        self.NF = MultiMLP(
            dim_in=len(grid_shape),
            dim_out=8,
            num_resnet_blocks=num_resnet_blocks,
            num_layers_per_block=num_layers_per_block,
            dim_hidden=dim_hidden,
            activation=activation,
            fourier_features=fourier_features,
            m_freqs=m_freqs,
            sigma=sigma,
            tune_beta=tune_beta
        ).to(self.device)

        print(f"NFAtmosphere initialized | Grid: {self.grid_shape} | Masked: {self.is_masked} | Blocks: {num_resnet_blocks}, Hidden: {dim_hidden}")

    def _expand(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape a flat parameter tensor to self.grid_shape if not masked.
        """
        if self.is_masked:
            return tensor
        return tensor.view(*self.grid_shape)

    def _evaluate_nf(self) -> Dict[str, torch.Tensor]:
        """
        Evaluate the neural network on the coordinates and apply parameter mapping.
        """
        out = self.NF(self.coords)
        
        # 1. Cartesian Magnetic Field Components
        # Scale factors matching typical values and legacy ME_INR_model2
        Blos = 5.0 * out[:, 0] * self.norm["Blos"]
        BQ   = 5.0 * out[:, 1] * self.norm["BQ"]
        BU   = 5.0 * out[:, 2] * self.norm["BU"]
        
        # Derive spherical parameters
        B2 = Blos**2 + BQ**2 + BU**2 + 1e-8
        B = torch.sqrt(B2)
        ratio = torch.clamp(Blos / B, -1.0 + 1e-6, 1.0 - 1e-6)
        gamma = torch.acos(ratio)
        xi = half_atan2_azimuth(BU, BQ)

        # 2. Thermodynamics and source function (bounded maps, see nf_heads)
        t = self.thermo_head(out[:, 3], out[:, 4], out[:, 5], out[:, 6], out[:, 7])
        vlos, eta0, a, ddop, S0, S1 = t["vlos"], t["eta0"], t["a"], t["ddop"], t["S0"], t["S1"]

        return {
            "B": B, "gamma": gamma, "xi": xi,
            "vlos": vlos, "eta0": eta0, "a": a, "ddop": ddop,
            "S0": S0, "S1": S1
        }

    @property
    def B(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["B"])

    @property
    def gamma(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["gamma"])

    @property
    def xi(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["xi"])

    @property
    def vlos(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["vlos"])

    @property
    def eta0(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["eta0"])

    @property
    def a(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["a"])

    @property
    def ddop(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["ddop"])

    @property
    def S0(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["S0"])

    @property
    def S1(self) -> torch.Tensor:
        return self._expand(self._evaluate_nf()["S1"])

    def to_stack(self) -> torch.Tensor:
        """Stack of the 9 physical parameters from a single network evaluation."""
        d = self._evaluate_nf()
        return torch.stack([self._expand(d[k]) for k in ("B", "gamma", "xi", "vlos", "eta0", "a", "ddop", "S0", "S1")], dim=-1)

    def clamp_parameters(self):
        """
        No-op since parameter mapping already constrains values to physical bounds.
        """
        pass
