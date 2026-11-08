"""
Single-View NF (Neural Field) Atmosphere.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn

from .base import Atmosphere, half_atan2_azimuth
from .nf_heads import ThermoHead, n_outputs
from ..nf import GridField, MultiMLP, to_coordinates


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
        encoding: str = "fourier",
        grid_kwargs: Optional[Dict[str, Any]] = None,
        source: str = "free",
        continuum: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        ``init``: starting values of the thermodynamic parameters (any of vlos,
        eta0, a, ddop, S0), reached when the network outputs 0, e.g. the medians
        of a pixel-wise inversion. ``bounds="clamp"``: range of vlos, eta0, a and
        ddop from DEFAULT_CLAMP_BOUNDS instead of the original ones. See
        :mod:`neural_me.core.atmospheres.nf_heads`.

        ``encoding="fourier"`` (default): MultiMLP with Fourier features, the
        network arguments above. ``encoding="grid"``: :class:`GridField`
        (multi-resolution grids + small head, ~50x cheaper per evaluation);
        ``grid_kwargs`` go to GridField, and the network arguments above are
        ignored except ``activation``. By default the grid field has two
        groups: B, vlos and S0 up to the map resolution, and eta0, a, ddop up
        to half of it (the sigma 100 / 40 split of the Fourier default).

        ``source``: parametrisation of the source function (see nf_heads).
        "free" (default): S0 and S1 are independent network outputs (9
        outputs). "continuum": S0 + S1 = ``continuum``, the observed continuum
        map (grid_shape, same units as the data); the network fits only
        S0 / (S0 + S1), i.e. the data are normalised to their own continuum
        internally, one parameter less (8 outputs).
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")

        # Normalization
        self.norm = self.DEFAULT_NORMALIZATION.copy()
        if norm:
            self.norm.update(norm)
        self.thermo_head = ThermoHead(self.norm, self.DEFAULT_CLAMP_BOUNDS, init=init, bounds=bounds,
                                      source=source)
        self.source = source
        n_out = 3 + n_outputs(source)

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

        if (source == "continuum") != (continuum is not None):
            raise ValueError('continuum map is needed with source="continuum", and only then')
        if continuum is not None:
            ic = torch.as_tensor(np.asarray(continuum), dtype=torch.float32, device=self.device)
            if tuple(ic.shape) != tuple(grid_shape):
                raise ValueError(f"continuum has shape {tuple(ic.shape)}, expected {tuple(grid_shape)}")
            ic = ic.reshape(-1)
            self.register_buffer("continuum", ic[mask.flatten()] if self.is_masked else ic)

        # Define NF (8 parameters predicted: Blos, BQ, BU, vlos, eta0, a, ddop, S0)
        if encoding not in ("fourier", "grid"):
            raise ValueError(f"encoding must be 'fourier' or 'grid', got {encoding!r}")
        self.encoding = encoding
        if encoding == "grid":
            n = max(grid_shape)
            fast = (0, 1, 2, 3, 7) + ((8,) if n_out == 9 else ())
            kw = dict(groups=[(fast, n), ((4, 5, 6), n // 2)],
                      activation=activation, zero_init=tuple(range(3, n_out)))
            kw.update(grid_kwargs or {})
            self.NF = GridField(grid_shape, dim_out=n_out, **kw).to(self.device)
            print(f"NFAtmosphere initialized | Grid: {self.grid_shape} | Masked: {self.is_masked} | "
                  f"encoding: grid, {sum(p.numel() for p in self.NF.parameters())} params")
            return
        if n_out == 9 and isinstance(sigma, list) and len(sigma) == 8:
            sigma = sigma + [sigma[7]]           # S1 gets the S0 scale
        self.NF = MultiMLP(
            dim_in=len(grid_shape),
            dim_out=n_out,
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
        if self.encoding == "grid":
            out = self.NF()
            if self.is_masked:
                out = out[self.mask.flatten()]
        else:
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
        t = self.thermo_head(out[:, 3], out[:, 4], out[:, 5], out[:, 6], out[:, 7],
                             o_S1=out[:, 8] if self.source == "free" else None,
                             ic=getattr(self, "continuum", None))
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
