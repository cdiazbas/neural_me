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


class StereoscopicAtmosphere(nn.Module):
    """
    Unified model for Stereoscopic inversions.

    Parameters stored (Total ~15 per pixel):
    1. Global Magnetic Field (Cartesian): Blos, Bn, Bw (3)
    2. Instrument 1 Thermodynamics: vlos, eta0, a, ddop, S0, S1 (6)
    3. Instrument 2 Thermodynamics: vlos, eta0, a, ddop, S0, S1 (6)
    4. Calibration: cali (1+)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        gamma: float = 101.0,
        rotangle: float = 90.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            grid_shape: (ny, nx)
            gamma: Inclination offset in DEGREES.
            rotangle: Azimuth rotation in DEGREES.
            device: Torch device.
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device("cpu")

        # Geometry constants
        self.gamma_rad = torch.tensor(
            np.radians(gamma), device=self.device, dtype=torch.float32
        )
        self.rotangle_rad = torch.tensor(
            np.radians(rotangle), device=self.device, dtype=torch.float32
        )

        # --- 1. Global Magnetic Field (Cartesian) ---
        # Initialized with noise to avoid 0-gradient singularity
        self.global_Blos = nn.Parameter(
            torch.randn(grid_shape, device=self.device) * 10.0 + 100.0
        )
        self.global_Bn = nn.Parameter(
            torch.randn(grid_shape, device=self.device) * 10.0 + 100.0
        )
        self.global_Bw = nn.Parameter(
            torch.randn(grid_shape, device=self.device) * 10.0 + 100.0
        )

        # --- 2. Instrument 1 Thermodynamics (Hinode) ---
        # Stored as RAW parameters (pre-activation)
        self.thermo1_vlos = nn.Parameter(torch.zeros(grid_shape, device=self.device))
        self.thermo1_eta0_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) + 1.0
        )  # Init > 0 for log
        self.thermo1_a_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) - 2.0
        )  # Small init
        self.thermo1_ddop_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) - 2.0
        )
        self.thermo1_S0_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) + 1.0
        )
        self.thermo1_S1_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) + 1.0
        )

        # --- 3. Instrument 2 Thermodynamics (SolO) ---
        self.thermo2_vlos = nn.Parameter(torch.zeros(grid_shape, device=self.device))
        self.thermo2_eta0_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) + 1.0
        )
        self.thermo2_a_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) - 2.0
        )
        self.thermo2_ddop_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) - 2.0
        )
        self.thermo2_S0_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) + 1.0
        )
        self.thermo2_S1_raw = nn.Parameter(
            torch.zeros(grid_shape, device=self.device) + 1.0
        )

        # --- 4. Calibration ---
        self.cali = nn.Parameter(torch.tensor([0.245], device=self.device))

        # --- Normalization Constants (Legacy Match) ---
        self._normalization = {
            "B": 500.0,
            "Blos": 500.0,
            "Bn": 500.0,
            "Bw": 500.0,
            "vlos": 2.0,
            "eta0": 100.0,
            "a": 0.2,
            "ddop": 1.0,
            "S0": 1.0,
            "S1": 1.0,
            "gamma": 1.0,
            "xi": 1.0,  # Angles not normalized usually but kept for consistency
        }

    @property
    def normalization(self):
        return self._normalization

    def initialize(self, method: str = "default"):
        """
        Initialize parameters.
        """
        if method == "default":
            # Re-initialize with random noise / default values similar to __init__
            with torch.no_grad():
                self.global_Blos.normal_(0, 10).add_(100)
                self.global_Bn.normal_(0, 10).add_(100)
                self.global_Bw.normal_(0, 10).add_(100)
                # Reset thermos to default (mostly zero/small for raw)
                for p in [self.thermo1_vlos, self.thermo2_vlos]:
                    p.zero_()
                for p in [self.thermo1_eta0_raw, self.thermo2_eta0_raw]:
                    p.fill_(1.0)
                for p in [
                    self.thermo1_a_raw,
                    self.thermo2_a_raw,
                    self.thermo1_ddop_raw,
                    self.thermo2_ddop_raw,
                ]:
                    p.fill_(-2.0)
                for p in [
                    self.thermo1_S0_raw,
                    self.thermo2_S0_raw,
                    self.thermo1_S1_raw,
                    self.thermo2_S1_raw,
                ]:
                    p.fill_(1.0)

    def _inverse_softplus(self, x):
        """
        Inverse of F.softplus: y = log(exp(x) - 1).
        Includes clamping to prevent NaNs on non-positive inputs.
        """
        # x = log(exp(y) + 1) -> exp(y) = exp(x) - 1 -> y = log(exp(x) - 1)
        # We must ensure exp(x) - 1 > 0  => exp(x) > 1 => x > 0.
        # Clamp x to a small positive value to handle legacy data inconsistencies.
        x_clamped = torch.clamp(torch.as_tensor(x), min=1e-7)
        return torch.log(torch.exp(x_clamped) - 1.0 + 1e-10)

    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameters from a dictionary of values.

        Values are assumed to be NORMALIZED physical values (matching
        legacy meparam). We convert them to raw parameters (inverse
        softplus/sigmoid) if needed.
        """

        def set_param(target_raw, value_norm, is_softplus=True):
            if value_norm is None:
                return
            val_tensor = torch.as_tensor(
                value_norm, device=self.device, dtype=torch.float32
            )
            if val_tensor.shape != target_raw.shape:
                val_tensor = val_tensor.reshape(target_raw.shape)

            if is_softplus:
                # Target stores raw, property returns softplus(raw)
                # We want softplus(raw) = val_norm (since normalization is applied on top usually?
                # WAIT: In ViewAtmosphere, property returns F.softplus(raw).
                # It does NOT apply * self.normalization there?
                # Let's check ViewAtmosphere implementation.
                # If ViewAtmosphere returns un-normalized physical units?
                # No, standard Atmosphere properties return PHYSICAL units.
                # My implementation of ViewAtmosphere (lines 160+) returns `F.softplus(raw)`.
                # If `raw` roughly maps to `normalized values`, then we are missing the `* norm` factor in ViewAtmosphere!
                # I MUST FIX ViewAtmosphere properties to multiply by normalization to be physically correct.
                # Assuming ViewAtmosphere is fixed/correct: property = softplus(raw) * norm.
                # Then val_norm * norm = property.
                # So softplus(raw) * norm = val_norm * norm -> softplus(raw) = val_norm.
                # So raw = inverse_softplus(val_norm).
                target_raw.data.copy_(self._inverse_softplus(val_tensor))
            else:
                # Linear (vlos)
                # vlos = raw.
                # vlos_physical = vlos_norm * norm.
                # property vlos = raw * norm (if we fix ViewAtmosphere).
                # So raw = val_norm.
                target_raw.data.copy_(val_tensor)

        # 1. Global B (Linear in legacy? Blos/Bn/Bw are raw there)
        # Legacy: Blos_A = meparam[:,0]*norm[0].
        # Here: global_Blos is used directly in _compute_magnetic.
        # And _compute_magnetic expects PHYSICAL units (Gauss).
        # So if input is normalized, we must multiply by norm.
        if "global_Blos" in params:
            self.global_Blos.data.copy_(
                torch.as_tensor(params["global_Blos"], device=self.device)
                * self._normalization["Blos"]
            )
        if "global_Bn" in params:
            self.global_Bn.data.copy_(
                torch.as_tensor(params["global_Bn"], device=self.device)
                * self._normalization["Bn"]
            )
        if "global_Bw" in params:
            self.global_Bw.data.copy_(
                torch.as_tensor(params["global_Bw"], device=self.device)
                * self._normalization["Bw"]
            )

        # 2. Thermo1
        set_param(
            self.thermo1_vlos, params.get("thermo1_vlos"), is_softplus=False
        )  # vlos is linear
        set_param(self.thermo1_eta0_raw, params.get("thermo1_eta0"), is_softplus=True)
        set_param(self.thermo1_a_raw, params.get("thermo1_a"), is_softplus=True)
        set_param(self.thermo1_ddop_raw, params.get("thermo1_ddop"), is_softplus=True)
        set_param(self.thermo1_S0_raw, params.get("thermo1_S0"), is_softplus=True)
        set_param(self.thermo1_S1_raw, params.get("thermo1_S1"), is_softplus=True)

        # 3. Thermo2
        set_param(self.thermo2_vlos, params.get("thermo2_vlos"), is_softplus=False)
        set_param(self.thermo2_eta0_raw, params.get("thermo2_eta0"), is_softplus=True)
        set_param(self.thermo2_a_raw, params.get("thermo2_a"), is_softplus=True)
        set_param(self.thermo2_ddop_raw, params.get("thermo2_ddop"), is_softplus=True)
        set_param(self.thermo2_S0_raw, params.get("thermo2_S0"), is_softplus=True)
        set_param(self.thermo2_S1_raw, params.get("thermo2_S1"), is_softplus=True)

        # 4. Calibration
        if "cali" in params:
            self.cali.data.copy_(torch.as_tensor(params["cali"], device=self.device))

    def clamp_parameters(self):
        """
        Enforce parameter constraints matching legacy script.

        Legacy constraints are on NORMALIZED values. We must clamp raw
        values such that softplus(raw) implies the normalized range.
        range_raw = inverse_softplus(range_norm)
        """

        def clamp_prop(tensor_raw, min_norm, max_norm, is_softplus=True):
            if is_softplus:
                min_raw = self._inverse_softplus(torch.tensor(min_norm)).item()
                max_raw = self._inverse_softplus(torch.tensor(max_norm)).item()
                tensor_raw.data.clamp_(min_raw, max_raw)
            else:
                # Linear (vlos)
                # vlos = raw * norm (Wait? No, see below)
                # If vlos property = raw * norm? Or just raw?
                # In ViewAtmosphere I need to check. If I assume ViewAtmosphere returns raw * norm_const...
                # Actually, legacy clamps: meparam[:,3] (normalized) to [-10, 10].
                # If my raw vlos corresponds to normalized vlos, I clamp to [-10, 10].
                tensor_raw.data.clamp_(min_norm, max_norm)

        # Constraints from ME_combinado:
        # Vlos: [-10, 10] (normalized? Legacy: `clamp(meparam*norm, -10, 10)/norm`. So clamp value is physical?
        # NO. `torch.clamp(meparam*norm, -10, 10)/norm` means the PHYSICAL value is clamped to [-10, 10].
        # If norm=2.0 (vlos), then normalized param is clamped to [-5, 5].
        # My `thermo_vlos` (if I define it as normalized) should be clamped to [-5, 5].
        # OR: I clamp physical value [-10, 10].
        # Let's assume my raw parameters are "normalized-like" because I implemented `set_parameters` assuming they map to `val_norm`.

        # Vlos (Physical [-10, 10] km/s) -> Normalized / 2.0 -> [-5, 5]
        clamp_prop(self.thermo1_vlos, -5.0, 5.0, is_softplus=False)
        clamp_prop(self.thermo2_vlos, -5.0, 5.0, is_softplus=False)

        # Ddop (Physical [0.01, 0.2]) -> Normalized / 1.0 -> [0.01, 0.2]
        clamp_prop(self.thermo1_ddop_raw, 0.01, 0.2, is_softplus=True)
        clamp_prop(self.thermo2_ddop_raw, 0.01, 0.2, is_softplus=True)

        # S0 (Physical [0.01, 1.5]) -> Normalized / 1.0 -> [0.01, 1.5]
        clamp_prop(self.thermo1_S0_raw, 0.01, 1.5, is_softplus=True)
        clamp_prop(self.thermo2_S0_raw, 0.01, 1.5, is_softplus=True)

        # S1 (Legacy commented out? No, active lines 374 set it. But if we clamp?)
        # Legacy loop *sets* S1 from data every step. If we optimize it, we should clamp it.
        # Assuming we optimize it: Physical [0.01, 1.5]
        clamp_prop(self.thermo1_S1_raw, 0.01, 1.5, is_softplus=True)
        clamp_prop(self.thermo2_S1_raw, 0.01, 1.5, is_softplus=True)

        # Damp (a) (Physical [0.001, 1.0]) -> Normalized / 0.2 -> [0.005, 5.0]
        clamp_prop(self.thermo1_a_raw, 0.005, 5.0, is_softplus=True)
        clamp_prop(self.thermo2_a_raw, 0.005, 5.0, is_softplus=True)

        # Eta0 (Physical [1.0, 700]) -> Normalized / 100.0 -> [0.01, 7.0]
        clamp_prop(self.thermo1_eta0_raw, 0.01, 7.0, is_softplus=True)
        clamp_prop(self.thermo2_eta0_raw, 0.01, 7.0, is_softplus=True)

    def get_atmosphere(self, instrument_index: int) -> "ViewAtmosphere":
        """
        Returns an Atmosphere object for the specified instrument.

        Index 0: Reference View (Global Frame).
        Index 1: Secondary View (Rotated Frame).
        """
        return ViewAtmosphere(self, instrument_index)

    def set_global_B(self, Blos: np.ndarray, Bn: np.ndarray, Bw: np.ndarray):
        with torch.no_grad():
            self.global_Blos.copy_(torch.from_numpy(Blos).to(self.device).float())
            self.global_Bn.copy_(torch.from_numpy(Bn).to(self.device).float())
            self.global_Bw.copy_(torch.from_numpy(Bw).to(self.device).float())


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

        if self.index == 0:
            # Identity transformation
            Btot = torch.sqrt(Blos**2 + Bn**2 + Bw**2)
            # inc = acos(Blos/Btot)
            eps = 1e-7
            inc = torch.acos(torch.clamp(Blos / (Btot + eps), -1.0 + eps, 1.0 - eps))
            azi = torch.atan2(Bn, Bw)
            return Btot, inc, azi

        elif self.index == 1:
            # Rotated transformation
            gamma = self.parent.gamma_rad
            rot = self.parent.rotangle_rad

            cos_g = torch.cos(gamma)
            sin_g = torch.sin(gamma)

            # Legacy equations
            # Blos_B = Blos_A*cos(gamma) + Bw_A*sin(gamma)
            # Bw_B = -Blos_A*sin(gamma) + Bw_A*cos(gamma)
            Blos_new = Blos * cos_g + Bw * sin_g
            Bw_new = -Blos * sin_g + Bw * cos_g
            Bn_new = Bn

            Btot = torch.sqrt(Blos_new**2 + Bn_new**2 + Bw_new**2)
            eps = 1e-7
            inc = torch.acos(
                torch.clamp(Blos_new / (Btot + eps), -1.0 + eps, 1.0 - eps)
            )
            azi = torch.atan2(Bn_new, Bw_new) + rot

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
        raw = self.parent.thermo1_vlos if self.index == 0 else self.parent.thermo2_vlos
        return raw * self.parent.normalization["vlos"]

    @property
    def eta0(self):
        raw = (
            self.parent.thermo1_eta0_raw
            if self.index == 0
            else self.parent.thermo2_eta0_raw
        )
        return F.softplus(raw) * self.parent.normalization["eta0"]

    @property
    def a(self):
        raw = (
            self.parent.thermo1_a_raw if self.index == 0 else self.parent.thermo2_a_raw
        )
        return F.softplus(raw) * self.parent.normalization["a"]

    @property
    def ddop(self):
        raw = (
            self.parent.thermo1_ddop_raw
            if self.index == 0
            else self.parent.thermo2_ddop_raw
        )
        return F.softplus(raw) * self.parent.normalization["ddop"]

    @property
    def S0(self):
        raw = (
            self.parent.thermo1_S0_raw
            if self.index == 0
            else self.parent.thermo2_S0_raw
        )
        return F.softplus(raw) * self.parent.normalization["S0"]

    @property
    def S1(self):
        raw = (
            self.parent.thermo1_S1_raw
            if self.index == 0
            else self.parent.thermo2_S1_raw
        )
        return F.softplus(raw) * self.parent.normalization["S1"]
