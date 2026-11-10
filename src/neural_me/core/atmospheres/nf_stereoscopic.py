"""
Stereoscopic NF (Neural Field) Atmosphere.
"""

import math
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn

from .base import Atmosphere
from .local_vertical import LocalVerticalMixin
from .step_cache import StepCacheMixin
from .nf_heads import ThermoHead, n_outputs
from ..nf import GridField, MultiMLP, to_coordinates


class StereoscopicNFAtmosphere(StepCacheMixin, LocalVerticalMixin, nn.Module):
    """
    Unified model for Stereoscopic inversions using Neural Fields.
    
    Houses a MultiMLP predicting 13 parameter channels:
    - 0, 1, 2: Global Cartesian Magnetic Field (Blos, Bn, Bw)
    - 3, 4, 5, 6, 7: Thermo 1 (vlos, eta0, a, ddop, S0)
    - 8, 9, 10, 11, 12: Thermo 2 (vlos, eta0, a, ddop, S0)
    """

    DEFAULT_NORMALIZATION = Atmosphere.DEFAULT_STEREO_NORMALIZATION
    DEFAULT_PARAMETERS = Atmosphere.DEFAULT_STEREO_PARAMETERS
    DEFAULT_CLAMP_BOUNDS = Atmosphere.DEFAULT_STEREO_CLAMP_BOUNDS

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        gamma: float = 101.0,
        beta: float = 0.0,
        delta: float = 90.0,
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
        source: str = "free",
        continuum: Optional[Tuple[Any, Any]] = None,
        encoding: str = "fourier",
        grid_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        ``init``: starting values of the thermodynamic parameters per view
        (keys ``thermo1_<p>`` / ``thermo2_<p>`` for p in vlos, eta0, a, ddop,
        S0), reached when the network outputs 0. ``bounds="clamp"``: ranges of
        vlos, eta0, a, ddop, S0, S1 from DEFAULT_STEREO_CLAMP_BOUNDS. See
        :mod:`neural_me.core.atmospheres.nf_heads`.

        ``source="free"`` (default): S0 and S1 of each view are independent
        network outputs (15 outputs: 3 global B + 6 per view). ``"continuum"``:
        S0 + S1 of view i is the observed continuum map ``continuum[i - 1]``
        (grid_shape each) and the network fits S0 / (S0 + S1) (13 outputs).
        A 13-entry ``sigma`` list (the old layout) gets S1 the sigma of S0.

        ``encoding="fourier"`` (default): MultiMLP with Fourier features.
        ``encoding="grid"``: :class:`GridField` (multi-resolution grids, ~50x
        cheaper; ``grid_kwargs`` go to it). Default groups: the global field
        and vlos, S0, S1 of both views up to the map resolution, eta0, a, ddop
        of both views up to half of it (the sigma 100 / 40 split).
        """
        super().__init__()
        self.grid_shape = grid_shape
        self._init_device = device or torch.device("cpu")

        # Geometry constants: non-persistent buffers, so they follow .to()
        # without changing the state_dict of existing checkpoints.
        for name, deg in (("gamma_rad", gamma), ("beta_rad", beta), ("delta_rad", delta)):
            self.register_buffer(
                name,
                torch.tensor(np.radians(deg), device=self.device, dtype=torch.float32),
                persistent=False,
            )

        # Normalization
        self._normalization = self.DEFAULT_NORMALIZATION.copy()

        if norm:
            self._normalization.update(norm)
        init = dict(init or {})
        allowed = {f"thermo{i}_{n}" for i in (1, 2) for n in ("vlos", "eta0", "a", "ddop", "S0", "S1")}
        if set(init) - allowed:
            raise ValueError(f"unknown init keys {sorted(set(init) - allowed)}")
        self._thermo_heads = {
            i: ThermoHead(self._normalization, self.DEFAULT_CLAMP_BOUNDS, bounds=bounds, prefix=f"thermo{i}_",
                          init={k: v for k, v in init.items() if k.startswith(f"thermo{i}_")}, source=source)
            for i in (1, 2)
        }
        self.source = source
        self._n_view = n_outputs(source)             # outputs per view
        if (source == "continuum") != (continuum is not None):
            raise ValueError('continuum maps (view 1, view 2) are needed with source="continuum", and only then')
        if continuum is not None:
            for i, ic in zip((1, 2), continuum):
                ic = torch.as_tensor(np.asarray(ic), dtype=torch.float32, device=self.device)
                if tuple(ic.shape) != tuple(grid_shape):
                    raise ValueError(f"continuum {i} has shape {tuple(ic.shape)}, expected {tuple(grid_shape)}")
                self.register_buffer(f"continuum{i}", ic.reshape(-1))
        if source == "free" and isinstance(sigma, list) and len(sigma) == 13:
            sigma = sigma[:8] + [sigma[7]] + sigma[8:] + [sigma[12]]
        self._defaults = self.DEFAULT_PARAMETERS.copy()
        self.n_elements = int(np.prod(grid_shape))
        coords_all = to_coordinates(grid_shape, device=self.device)
        self.register_buffer("coords", coords_all)

        # Define NF (3 global B channels + thermodynamics of each view)
        n_out = 3 + 2 * self._n_view
        if encoding not in ("fourier", "grid"):
            raise ValueError(f"encoding must be 'fourier' or 'grid', got {encoding!r}")
        self.encoding = encoding
        if encoding == "grid":
            if len(grid_shape) != 2:
                raise ValueError("encoding='grid' needs a 2-D grid")
            n = max(grid_shape)
            slow = tuple(3 + v * self._n_view + k for v in (0, 1) for k in (1, 2, 3))   # eta0, a, ddop
            fast = tuple(c for c in range(n_out) if c not in slow)
            kw = dict(groups=[(fast, n), (slow, n // 2)], activation=activation,
                      zero_init=tuple(range(3, n_out)))
            kw.update(grid_kwargs or {})
            self.NF = GridField(grid_shape, dim_out=n_out, **kw).to(self.device)
            print(f"StereoscopicNFAtmosphere initialized | Grid: {self.grid_shape} | γ={gamma:.1f}°, β={beta:.1f}°, "
                  f"δ={delta:.1f}° | encoding: grid, source: {source}, "
                  f"{sum(p.numel() for p in self.NF.parameters())} params")
            return
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

        print(f"StereoscopicNFAtmosphere initialized | Grid: {self.grid_shape} | γ={gamma:.1f}°, β={beta:.1f}°, δ={delta:.1f}° | Blocks: {num_resnet_blocks}, Hidden: {dim_hidden}")

    @property
    def device(self) -> torch.device:
        """Device of the parameters (follows ``.to()``)."""
        p = next(self.parameters(), None)
        return p.device if p is not None else self._init_device

    @property
    def normalization(self):
        return self._normalization

    def _expand(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(*self.grid_shape)

    def _net(self) -> torch.Tensor:
        """Network outputs on the pixel grid, (n_pixels, n_outputs); one evaluation per open step_cache()."""
        return self._cached(lambda: self.NF() if getattr(self, "encoding", "fourier") == "grid" else self.NF(self.coords))

    def _global_B(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hook for LocalVerticalMixin: global field in the reference LOS frame."""
        out = self._net()
        return (
            self._expand(5.0 * out[:, 0] * self._normalization["global_Blos"]),
            self._expand(5.0 * out[:, 1] * self._normalization["global_Bn"]),
            self._expand(5.0 * out[:, 2] * self._normalization["global_Bw"]),
        )

    def _evaluate_for_view(self, index: int, out: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate the neural network on the coordinates and map parameters for the given view.

        ``out`` is a network output already computed for ``self.coords``; it
        lets several views share one evaluation.
        """
        if out is None:
            out = self._net()
        
        # 1. Global Cartesian Magnetic Field Components
        Blos = 5.0 * out[:, 0] * self._normalization["global_Blos"]
        Bn   = 5.0 * out[:, 1] * self._normalization["global_Bn"]
        Bw   = 5.0 * out[:, 2] * self._normalization["global_Bw"]

        # Magnetic view transformation
        if index == 1:
            B2 = Blos**2 + Bn**2 + Bw**2 + 1e-8
            Btot = torch.sqrt(B2)
            inc = torch.acos(torch.clamp(Blos / Btot, -1.0 + 1e-6, 1.0 - 1e-6))
            azi = torch.atan2(Bn, Bw)
        else:
            cos_g = torch.cos(self.gamma_rad)
            sin_g = torch.sin(self.gamma_rad)
            cos_b = torch.cos(self.beta_rad)
            sin_b = torch.sin(self.beta_rad)

            # First rotation by longitudinal separation gamma around Bn (North/Y axis):
            Blos_prime = Blos * cos_g + Bw * sin_g
            Bn_prime = Bn
            Bw_prime = -Blos * sin_g + Bw * cos_g

            # Second rotation by elevation tilt beta around Bw_prime (West/Z axis):
            Blos_new = Blos_prime * cos_b + Bn_prime * sin_b
            Bn_new = -Blos_prime * sin_b + Bn_prime * cos_b
            Bw_new = Bw_prime

            B2 = Blos_new**2 + Bn_new**2 + Bw_new**2 + 1e-8
            Btot = torch.sqrt(B2)
            inc = torch.acos(torch.clamp(Blos_new / Btot, -1.0 + 1e-6, 1.0 - 1e-6))
            azi = torch.atan2(Bn_new, Bw_new) + self.delta_rad

        # 2. Thermodynamics
        offset = 3 if index == 1 else 3 + self._n_view
        t = self._thermo_heads[index](
            *(out[:, offset + k] for k in range(5)),
            o_S1=out[:, offset + 5] if self.source == "free" else None,
            ic=getattr(self, f"continuum{index}", None))
        vlos, eta0, a, ddop, S0, S1 = t["vlos"], t["eta0"], t["a"], t["ddop"], t["S0"], t["S1"]

        return {
            "B": self._expand(Btot),
            "gamma": self._expand(inc),
            "xi": self._expand(azi),
            "vlos": self._expand(vlos),
            "eta0": self._expand(eta0),
            "a": self._expand(a),
            "ddop": self._expand(ddop),
            "S0": self._expand(S0),
            "S1": self._expand(S1)
        }

    def get_atmosphere(self, instrument_index: int) -> "NFViewAtmosphere":
        """
        Returns an Atmosphere view for the specified instrument index (1 or 2).
        """
        if instrument_index not in (1, 2):
            raise ValueError("instrument_index must be 1 or 2")
        return NFViewAtmosphere(self, instrument_index)

    def clamp_parameters(self):
        """
        Clamps atmosphere parameters. Activations handle physical constraints.
        """
        pass

    def to_stack(self) -> torch.Tensor:
        """
        Returns the stack of ALL physical parameters in stereoscopic configuration.

        Shape: (..., 15)
        """
        out = self._net()
        Blos = self._expand(5.0 * out[:, 0] * self._normalization["global_Blos"])
        Bn   = self._expand(5.0 * out[:, 1] * self._normalization["global_Bn"])
        Bw   = self._expand(5.0 * out[:, 2] * self._normalization["global_Bw"])
        
        thermo = ("vlos", "eta0", "a", "ddop", "S0", "S1")
        view1 = self._evaluate_for_view(1, out=out)
        view2 = self._evaluate_for_view(2, out=out)

        return torch.stack(
            [Blos, Bn, Bw]
            + [view1[k] for k in thermo]
            + [view2[k] for k in thermo],
            dim=-1,
        )

    def to_numpy_stack(self) -> np.ndarray:
        """
        Returns the stack of physical parameters as a numpy array. Moves to CPU
        and detaches graph.
        """
        return self.to_stack().detach().cpu().numpy()



class NFViewAtmosphere(Atmosphere):
    """
    Atmosphere view presenting standard Atmosphere interfaces for StereoscopicNFAtmosphere.
    """

    def __init__(self, parent: StereoscopicNFAtmosphere, index: int):
        super().__init__()
        self.parent = parent
        self.index = index
        self.grid_shape = parent.grid_shape

    @property
    def device(self) -> torch.device:
        return self.parent.device

    @property
    def B(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["B"]

    @property
    def gamma(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["gamma"]

    @property
    def xi(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["xi"]

    @property
    def vlos(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["vlos"]

    @property
    def eta0(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["eta0"]

    @property
    def a(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["a"]

    @property
    def ddop(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["ddop"]

    @property
    def S0(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["S0"]

    @property
    def S1(self) -> torch.Tensor:
        return self.parent._evaluate_for_view(self.index)["S1"]

    def to_stack(self) -> torch.Tensor:
        """Stack of the 9 physical parameters from a single network evaluation."""
        d = self.parent._evaluate_for_view(self.index)
        return torch.stack([d[k] for k in ("B", "gamma", "xi", "vlos", "eta0", "a", "ddop", "S0", "S1")], dim=-1)

    def clamp_parameters(self):
        pass
