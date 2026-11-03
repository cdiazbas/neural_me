"""
Stereoscopic Atmosphere Implementation (Simplified & Optimized).

This module contains the `StereoscopicAtmosphere` class which holds ALL parameters for
a multi-view inversion (Global Magnetic Field + Instrument-Specific Thermodynamics).

Refactored to define thermodynamic parameters directly, avoiding unused magnetic parameters
from nested SphericalSoftplusAtmosphere instances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Any, Dict

from .base import Atmosphere
from .local_vertical import LocalVerticalMixin


class StereoscopicAtmosphere(LocalVerticalMixin, nn.Module):
    """
    Unified model for Stereoscopic inversions.

    Parameters stored (Total ~15 per pixel):
    1. Global Magnetic Field (Cartesian): Blos, Bn, Bw (3)
    2. Instrument 1 Thermodynamics: vlos, eta0, a, ddop, S0, S1 (6)
    3. Instrument 2 Thermodynamics: vlos, eta0, a, ddop, S0, S1 (6)
    4. Calibration: cali (1+)
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
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            grid_shape: (ny, nx)
            gamma: Longitudinal separation angle in DEGREES.
            beta: Elevation tilt angle in DEGREES.
            delta: Azimuth rotation in DEGREES.
            device: Torch device.
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")
        self._normalization = self.DEFAULT_NORMALIZATION.copy()
        self._defaults = self.DEFAULT_PARAMETERS.copy()

        # Geometry constants
        self.gamma_rad = torch.tensor(
            np.radians(gamma), device=self.device, dtype=torch.float32
        )
        self.beta_rad = torch.tensor(
            np.radians(beta), device=self.device, dtype=torch.float32
        )
        self.delta_rad = torch.tensor(
            np.radians(delta), device=self.device, dtype=torch.float32
        )

        # --- 1. Global Magnetic Field (Cartesian) ---
        # Initialized with noise to avoid 0-gradient singularity
        global_noise_std = 10.0
        self.global_Blos_raw = nn.Parameter(
            (torch.randn(grid_shape, device=self.device) * global_noise_std
             + self._defaults["global_Blos"]) / self._normalization["global_Blos"]
        )
        self.global_Bn_raw = nn.Parameter(
            (torch.randn(grid_shape, device=self.device) * global_noise_std
             + self._defaults["global_Bn"]) / self._normalization["global_Bn"]
        )
        self.global_Bw_raw = nn.Parameter(
            (torch.randn(grid_shape, device=self.device) * global_noise_std
             + self._defaults["global_Bw"]) / self._normalization["global_Bw"]
        )

        # --- 2. Instrument 1 Thermodynamics (Hinode) ---
        # Stored as RAW parameters (pre-activation)
        self.thermo1_vlos = nn.Parameter(
            torch.full(
                grid_shape,
                self._defaults["thermo1_vlos"] / self._normalization["vlos"],
                device=self.device,
            )
        )
        self.thermo1_eta0_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo1_eta0"] / self._normalization["eta0"]
                ),
                device=self.device,
            )
        )
        self.thermo1_a_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo1_a"] / self._normalization["a"]
                ),
                device=self.device,
            )
        )
        self.thermo1_ddop_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo1_ddop"] / self._normalization["ddop"]
                ),
                device=self.device,
            )
        )
        self.thermo1_S0_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo1_S0"] / self._normalization["S0"]
                ),
                device=self.device,
            )
        )
        self.thermo1_S1_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo1_S1"] / self._normalization["S1"]
                ),
                device=self.device,
            )
        )

        # --- 3. Instrument 2 Thermodynamics (SolO) ---
        self.thermo2_vlos = nn.Parameter(
            torch.full(
                grid_shape,
                self._defaults["thermo2_vlos"] / self._normalization["vlos"],
                device=self.device,
            )
        )
        self.thermo2_eta0_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo2_eta0"] / self._normalization["eta0"]
                ),
                device=self.device,
            )
        )
        self.thermo2_a_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo2_a"] / self._normalization["a"]
                ),
                device=self.device,
            )
        )
        self.thermo2_ddop_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo2_ddop"] / self._normalization["ddop"]
                ),
                device=self.device,
            )
        )
        self.thermo2_S0_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo2_S0"] / self._normalization["S0"]
                ),
                device=self.device,
            )
        )
        self.thermo2_S1_raw = nn.Parameter(
            torch.full(
                grid_shape,
                self._inverse_softplus(
                    self._defaults["thermo2_S1"] / self._normalization["S1"]
                ),
                device=self.device,
            )
        )

        print(f"StereoscopicAtmosphere initialized | Grid: {self.grid_shape} | γ={gamma:.1f}°, β={beta:.1f}°, δ={delta:.1f}°")

    @property
    def normalization(self):
        return self._normalization

    @property
    def global_Blos(self) -> torch.Tensor:
        return self.global_Blos_raw * self._normalization["global_Blos"]

    @property
    def global_Bn(self) -> torch.Tensor:
        return self.global_Bn_raw * self._normalization["global_Bn"]

    @property
    def global_Bw(self) -> torch.Tensor:
        return self.global_Bw_raw * self._normalization["global_Bw"]

    def _global_B(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hook for LocalVerticalMixin: global field in the reference LOS frame."""
        return self.global_Blos, self.global_Bn, self.global_Bw

    # ------------------------------------------------------------------ #
    #  Clamp helpers (mirrors Atmosphere base class API)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inv_softplus_scalar(x: float, eps: float = 1e-7) -> float:
        """Numerically stable inverse softplus for a scalar. See Atmosphere._inv_softplus_scalar."""
        return Atmosphere._inv_softplus_scalar(x, eps)

    def _clamp_bounds(self, name: str, norm: float = 1.0):
        """Return (lo, hi) clamp bounds in *normalized* space for parameter *name*.

        Delegates to :data:`DEFAULT_CLAMP_BOUNDS` from base.
        """
        lo_phys, hi_phys = self.DEFAULT_CLAMP_BOUNDS[name]
        if norm == 1.0:
            return lo_phys, hi_phys
        return lo_phys / norm, hi_phys / norm

    def initialize(self, method: str = "default"):
        """
        Initialize parameters.
        """
        if method == "default":
            # Re-initialize with random noise / default values similar to __init__
            with torch.no_grad():
                self.global_Blos_raw.normal_(0, 10 / self._normalization["global_Blos"]).add_(self._defaults["global_Blos"] / self._normalization["global_Blos"])
                self.global_Bn_raw.normal_(0, 10 / self._normalization["global_Bn"]).add_(self._defaults["global_Bn"] / self._normalization["global_Bn"])
                self.global_Bw_raw.normal_(0, 10 / self._normalization["global_Bw"]).add_(self._defaults["global_Bw"] / self._normalization["global_Bw"])
                self.thermo1_vlos.fill_(
                    self._defaults["thermo1_vlos"] / self._normalization["vlos"]
                )
                self.thermo2_vlos.fill_(
                    self._defaults["thermo2_vlos"] / self._normalization["vlos"]
                )
                self.thermo1_eta0_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo1_eta0"] / self._normalization["eta0"]
                    )
                )
                self.thermo2_eta0_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo2_eta0"] / self._normalization["eta0"]
                    )
                )
                self.thermo1_a_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo1_a"] / self._normalization["a"]
                    )
                )
                self.thermo2_a_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo2_a"] / self._normalization["a"]
                    )
                )
                self.thermo1_ddop_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo1_ddop"] / self._normalization["ddop"]
                    )
                )
                self.thermo2_ddop_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo2_ddop"] / self._normalization["ddop"]
                    )
                )
                self.thermo1_S0_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo1_S0"] / self._normalization["S0"]
                    )
                )
                self.thermo2_S0_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo2_S0"] / self._normalization["S0"]
                    )
                )
                self.thermo1_S1_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo1_S1"] / self._normalization["S1"]
                    )
                )
                self.thermo2_S1_raw.fill_(
                    self._inverse_softplus(
                        self._defaults["thermo2_S1"] / self._normalization["S1"]
                    )
                )

    def _inverse_softplus(self, x, epsilon=1e-7):
        """
        Numerically stable inverse of F.softplus: y = x + log(-expm1(-x)).
        """
        x_tensor = torch.as_tensor(x)
        x_clamped = torch.clamp(x_tensor, min=epsilon)
        return x_clamped + torch.log(-torch.expm1(-x_clamped))

    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameters from a dictionary of physical values.
        """

        def set_param(target_raw, value_phys, norm_val, is_softplus=True):
            if value_phys is None:
                return
            val_tensor = torch.as_tensor(
                value_phys, device=self.device, dtype=torch.float32
            )
            if val_tensor.shape != target_raw.shape:
                val_tensor = val_tensor.reshape(target_raw.shape)

            if is_softplus:
                target_raw.data.copy_(self._inverse_softplus(val_tensor / norm_val))
            else:
                target_raw.data.copy_(val_tensor / norm_val)

        # 1. Global B (Stored in normalized units)
        if "global_Blos" in params:
            self.global_Blos_raw.data.copy_(
                torch.as_tensor(params["global_Blos"], device=self.device) / self._normalization["global_Blos"]
            )
        if "global_Bn" in params:
            self.global_Bn_raw.data.copy_(
                torch.as_tensor(params["global_Bn"], device=self.device) / self._normalization["global_Bn"]
            )
        if "global_Bw" in params:
            self.global_Bw_raw.data.copy_(
                torch.as_tensor(params["global_Bw"], device=self.device) / self._normalization["global_Bw"]
            )

        # 2. Thermo1
        set_param(
            self.thermo1_vlos, params.get("thermo1_vlos"), self._normalization["vlos"], is_softplus=False
        )
        set_param(self.thermo1_eta0_raw, params.get("thermo1_eta0"), self._normalization["eta0"], is_softplus=True)
        set_param(self.thermo1_a_raw, params.get("thermo1_a"), self._normalization["a"], is_softplus=True)
        set_param(self.thermo1_ddop_raw, params.get("thermo1_ddop"), self._normalization["ddop"], is_softplus=True)
        set_param(self.thermo1_S0_raw, params.get("thermo1_S0"), self._normalization["S0"], is_softplus=True)
        set_param(self.thermo1_S1_raw, params.get("thermo1_S1"), self._normalization["S1"], is_softplus=True)

        # 3. Thermo2
        set_param(self.thermo2_vlos, params.get("thermo2_vlos"), self._normalization["vlos"], is_softplus=False)
        set_param(self.thermo2_eta0_raw, params.get("thermo2_eta0"), self._normalization["eta0"], is_softplus=True)
        set_param(self.thermo2_a_raw, params.get("thermo2_a"), self._normalization["a"], is_softplus=True)
        set_param(self.thermo2_ddop_raw, params.get("thermo2_ddop"), self._normalization["ddop"], is_softplus=True)
        set_param(self.thermo2_S0_raw, params.get("thermo2_S0"), self._normalization["S0"], is_softplus=True)
        set_param(self.thermo2_S1_raw, params.get("thermo2_S1"), self._normalization["S1"], is_softplus=True)

        # 4. (epsilon removed — owned by degradation operators)

    def clamp_parameters(self):
        """
        Enforce physical parameter bounds on all internal tensors.

        Two normalization chains are used:

        **Chain A — linear normalized** (``thermo_vlos``):
            ``stored = physical / norm``
            → clamp to ``(lo_phys / norm, hi_phys / norm)``

        **Chain B — softplus pre-activation** (all ``_raw`` parameters):
            ``physical = softplus(raw) * norm``
            → only upper bound enforced: ``raw.clamp_(max=inv_softplus(hi/norm))``
            (softplus already guarantees positivity; lower physical bound is satisfied
            by construction)

        All physical bounds come from :attr:`DEFAULT_CLAMP_BOUNDS`.
        """
        norm = self._normalization

        def storage_norm(name):
            # Thermo parameters are stored, set and read with the unprefixed
            # normalisation ("eta0"), so clamp with the same one; the bounds
            # stay per view ("thermo1_eta0").
            return norm[name.split("_", 1)[1] if name.startswith("thermo") else name]

        def clamp_softplus_upper(tensor_raw, name):
            """Clamp raw softplus parameter to its upper physical bound."""
            _, hi_norm = self._clamp_bounds(name, storage_norm(name))
            tensor_raw.data.clamp_(max=self._inv_softplus_scalar(hi_norm))

        def clamp_linear(tensor_norm, name):
            """Clamp linearly normalized parameter to physical bounds / norm."""
            lo, hi = self._clamp_bounds(name, storage_norm(name))
            tensor_norm.data.clamp_(lo, hi)
        # --- Thermodynamic softplus parameters (Chain B) ---
        clamp_softplus_upper(self.thermo1_eta0_raw, "thermo1_eta0")
        clamp_softplus_upper(self.thermo2_eta0_raw, "thermo2_eta0")
        clamp_softplus_upper(self.thermo1_a_raw, "thermo1_a")
        clamp_softplus_upper(self.thermo2_a_raw, "thermo2_a")
        clamp_softplus_upper(self.thermo1_ddop_raw, "thermo1_ddop")
        clamp_softplus_upper(self.thermo2_ddop_raw, "thermo2_ddop")
        clamp_softplus_upper(self.thermo1_S0_raw, "thermo1_S0")
        clamp_softplus_upper(self.thermo2_S0_raw, "thermo2_S0")
        clamp_softplus_upper(self.thermo1_S1_raw, "thermo1_S1")
        clamp_softplus_upper(self.thermo2_S1_raw, "thermo2_S1")

        # --- vlos: linear normalized (Chain A) ---
        clamp_linear(self.thermo1_vlos, "thermo1_vlos")
        clamp_linear(self.thermo2_vlos, "thermo2_vlos")

        # --- Global Magnetic Field (Cartesian) ---
        lo, hi = self._clamp_bounds("global_Blos", self._normalization["global_Blos"])
        self.global_Blos_raw.data.clamp_(lo, hi)
        lo, hi = self._clamp_bounds("global_Bn", self._normalization["global_Bn"])
        self.global_Bn_raw.data.clamp_(lo, hi)
        lo, hi = self._clamp_bounds("global_Bw", self._normalization["global_Bw"])
        self.global_Bw_raw.data.clamp_(lo, hi)


    def get_atmosphere(self, instrument_index: int) -> "ViewAtmosphere":
        """
        Returns an Atmosphere object for the specified instrument.

        Index 1: Reference View (Global Frame).
        Index 2: Secondary View (Rotated Frame).
        """
        if instrument_index not in (1, 2):
            raise ValueError("instrument_index must be 1 or 2")
        return ViewAtmosphere(self, instrument_index)

    def set_global_B(self, Blos: np.ndarray, Bn: np.ndarray, Bw: np.ndarray):
        with torch.no_grad():
            self.global_Blos_raw.copy_(torch.from_numpy(Blos).to(self.device).float() / self._normalization["global_Blos"])
            self.global_Bn_raw.copy_(torch.from_numpy(Bn).to(self.device).float() / self._normalization["global_Bn"])
            self.global_Bw_raw.copy_(torch.from_numpy(Bw).to(self.device).float() / self._normalization["global_Bw"])

    def to_stack(self) -> torch.Tensor:
        """
        Returns the stack of ALL physical parameters in stereoscopic configuration.

        Shape: (..., 15)
        Order: [global_Blos, global_Bn, global_Bw, thermo1_vlos, ..., thermo1_S1, thermo2_vlos, ..., thermo2_S1]
        """
        thermo1 = self.get_atmosphere(1)
        thermo2 = self.get_atmosphere(2)
        return torch.stack(
            [
                self.global_Blos,
                self.global_Bn,
                self.global_Bw,
                thermo1.vlos,
                thermo1.eta0,
                thermo1.a,
                thermo1.ddop,
                thermo1.S0,
                thermo1.S1,
                thermo2.vlos,
                thermo2.eta0,
                thermo2.a,
                thermo2.ddop,
                thermo2.S0,
                thermo2.S1,
            ],
            dim=-1,
        )

    def to_numpy_stack(self) -> np.ndarray:
        """
        Returns the stack of physical parameters as a numpy array. Moves to CPU
        and detaches graph.
        """
        return self.to_stack().detach().cpu().numpy()


class ViewAtmosphere(Atmosphere):
    """
    A lightweight internal class that presents a standard Atmosphere interface
    to the MEProblem, pulling data from the parent StereoscopicAtmosphere.
    """

    def __init__(self, parent: StereoscopicAtmosphere, index: int):
        super().__init__()
        self.parent = parent
        self.index = index
        self.device = parent.device
        self.grid_shape = parent.grid_shape

    # --- Magnetic Properties ---

    def _compute_magnetic(self):
        Blos = self.parent.global_Blos
        Bn = self.parent.global_Bn
        Bw = self.parent.global_Bw

        if self.index == 1:
            # Identity transformation
            B2 = Blos**2 + Bn**2 + Bw**2 + 1e-8
            Btot = torch.sqrt(B2)
            # inc = acos(Blos/Btot); clamp ratio away from ±1 to keep acos gradient finite
            inc = torch.acos(torch.clamp(Blos / Btot, -1.0 + 1e-6, 1.0 - 1e-6))
            azi = torch.atan2(Bn, Bw)
            return Btot, inc, azi

        elif self.index == 2:
            # Rotated transformation
            gamma = self.parent.gamma_rad
            beta = self.parent.beta_rad
            delta = self.parent.delta_rad

            cos_g = torch.cos(gamma)
            sin_g = torch.sin(gamma)
            cos_b = torch.cos(beta)
            sin_b = torch.sin(beta)

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
            inc = torch.acos(
                torch.clamp(Blos_new / Btot, -1.0 + 1e-6, 1.0 - 1e-6)
            )
            azi = torch.atan2(Bn_new, Bw_new) + delta

            return Btot, inc, azi

    @property
    def B(self):
        return self._compute_magnetic()[0]

    @property
    def gamma(self):
        return self._compute_magnetic()[1]

    @property
    def xi(self):
        return self._compute_magnetic()[2]

    # --- Thermodynamic Properties ---
    # Implement Softplus constraints here, mirroring SphericalSoftplusAtmosphere
    # CRITICAL: Apply normalization from parent to return PHYSICAL units.

    @property
    def vlos(self):
        # No constraint
        raw = self.parent.thermo1_vlos if self.index == 1 else self.parent.thermo2_vlos
        return raw * self.parent.normalization["vlos"]

    @property
    def eta0(self):
        raw = (
            self.parent.thermo1_eta0_raw
            if self.index == 1
            else self.parent.thermo2_eta0_raw
        )
        return F.softplus(raw) * self.parent.normalization["eta0"]

    @property
    def a(self):
        raw = (
            self.parent.thermo1_a_raw if self.index == 1 else self.parent.thermo2_a_raw
        )
        return F.softplus(raw) * self.parent.normalization["a"]

    @property
    def ddop(self):
        raw = (
            self.parent.thermo1_ddop_raw
            if self.index == 1
            else self.parent.thermo2_ddop_raw
        )
        return F.softplus(raw) * self.parent.normalization["ddop"]

    @property
    def S0(self):
        raw = (
            self.parent.thermo1_S0_raw
            if self.index == 1
            else self.parent.thermo2_S0_raw
        )
        return F.softplus(raw) * self.parent.normalization["S0"]

    @property
    def S1(self):
        raw = (
            self.parent.thermo1_S1_raw
            if self.index == 1
            else self.parent.thermo2_S1_raw
        )
        return F.softplus(raw) * self.parent.normalization["S1"]
