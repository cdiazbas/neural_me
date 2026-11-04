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
from .nf_heads import ThermoHead
from ..nf import MultiMLP, to_coordinates


class StereoscopicNFAtmosphere(LocalVerticalMixin, nn.Module):
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
    ):
        """
        ``init``: starting values of the thermodynamic parameters per view
        (keys ``thermo1_<p>`` / ``thermo2_<p>`` for p in vlos, eta0, a, ddop,
        S0), reached when the network outputs 0. ``bounds="clamp"``: ranges of
        vlos, eta0, a, ddop from DEFAULT_STEREO_CLAMP_BOUNDS. See
        :mod:`neural_me.core.atmospheres.nf_heads`.
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
        allowed = {f"thermo{i}_{n}" for i in (1, 2) for n in ("vlos", "eta0", "a", "ddop", "S0")}
        if set(init) - allowed:
            raise ValueError(f"unknown init keys {sorted(set(init) - allowed)}")
        self._thermo_heads = {
            i: ThermoHead(self._normalization, self.DEFAULT_CLAMP_BOUNDS, bounds=bounds, prefix=f"thermo{i}_",
                          init={k: v for k, v in init.items() if k.startswith(f"thermo{i}_")})
            for i in (1, 2)
        }
        self._defaults = self.DEFAULT_PARAMETERS.copy()
        self.n_elements = int(np.prod(grid_shape))
        coords_all = to_coordinates(grid_shape, device=self.device)
        self.register_buffer("coords", coords_all)

        # Define NF (13 parameter channels)
        self.NF = MultiMLP(
            dim_in=len(grid_shape),
            dim_out=13,
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

    def _global_B(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hook for LocalVerticalMixin: global field in the reference LOS frame."""
        out = self.NF(self.coords)
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
            out = self.NF(self.coords)
        
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
        offset = 3 if index == 1 else 8
        prefix = f"thermo{1 if index == 1 else 2}_"
        
        t = self._thermo_heads[index](*(out[:, offset + k] for k in range(5)))
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
        out = self.NF(self.coords)
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
