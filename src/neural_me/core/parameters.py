"""
Parameter Models for Milne-Eddington Inversion.

This module provides abstractions for optimization parameters, decoupling
what we optimize (e.g., Cartesian B-field) from what the physics engine
needs (spherical B-field).

Classes:
    ParameterModel: Abstract base class
    CartesianParameterModel: Optimizes Blos, BQ, BU (Cartesian) -> B, gamma, xi
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Union
from abc import ABC, abstractmethod

from neural_me.core.atmosphere import Atmosphere


class ParameterModel(nn.Module, ABC):
    """
    Abstract base class for parameter models.
    
    A ParameterModel holds the learnable optimization variables and provides
    a forward() method that returns one or more Atmosphere objects with
    the physical parameters needed by the ME synthesis engine.
    """
    
    # Default normalization factors (from legacy)
    DEFAULT_NORM = {
        'B': 500.0,      # Magnetic field (G)
        'Blos': 500.0, 'BQ': 500.0, 'BU': 500.0,
        'gamma': 1.0,    # Inclination (rad)
        'xi': 1.0,       # Azimuth (rad)
        'vlos': 2.0,     # Velocity (km/s)
        'eta0': 100.0,   # Line-to-continuum
        'a': 0.2,        # Damping
        'ddop': 1.0,     # Doppler width (Å)
        'S0': 1.0,       # Source function constant
        'S1': 1.0        # Source function gradient
    }
    
    @abstractmethod
    def forward(self) -> Union[Atmosphere, Tuple[Atmosphere, ...]]:
        """Returns one or more Atmosphere objects."""
        pass
    
    @abstractmethod
    def clamp_parameters(self):
        """Apply physical constraints to parameters."""
        pass
    
    def get_raw_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns dict of raw (normalized) parameter tensors."""
        return {name: param.data for name, param in self.named_parameters()}


class CartesianParameterModel(ParameterModel):
    """
    Optimizes Cartesian magnetic field components (Blos, BQ, BU).
    
    Converts to spherical (B, gamma, xi) for Atmosphere output.
    This parameterization avoids periodicity issues with angular variables.
    
    Optimization Parameters (9 total):
      - Blos, BQ, BU: Cartesian B-field components (normalized)
      - vlos: Line-of-sight velocity (normalized)
      - eta0: Line-to-continuum absorption ratio (normalized)
      - a: Damping parameter (normalized)
      - ddop: Doppler width (normalized)
      - S0, S1: Source function parameters (normalized)
    """
    
    def __init__(
        self,
        n_pixels: int,
        grid_shape: Tuple[int, int],
        norm: Optional[Dict[str, float]] = None,
        device: torch.device = None,
        initial_values: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize CartesianParameterModel.
        
        Args:
            n_pixels: Number of pixels (Ny * Nx)
            grid_shape: Spatial grid shape (Ny, Nx)
            norm: Normalization factors (defaults to DEFAULT_NORM)
            device: Torch device
            initial_values: Optional dict of initial values (physical units)
                           Keys: 'Blos', 'BQ', 'BU', 'vlos', etc.
        """
        super().__init__()
        self.n_pixels = n_pixels
        self.grid_shape = grid_shape
        self.device = device or torch.device('cpu')
        
        # Normalization factors
        self.norm = self.DEFAULT_NORM.copy()
        if norm:
            self.norm.update(norm)
        
        # Store as buffer for easy access
        self.register_buffer('_norm_tensor', torch.tensor([
            self.norm['Blos'], self.norm['BQ'], self.norm['BU'],
            self.norm['vlos'], self.norm['eta0'], self.norm['a'],
            self.norm['ddop'], self.norm['S0'], self.norm['S1']
        ], dtype=torch.float32, device=self.device))
        
        # Initialize parameters
        self._init_parameters(initial_values)
    
    def _init_parameters(self, initial_values: Optional[Dict[str, np.ndarray]] = None):
        """Initialize learnable parameters."""
        n = self.n_pixels
        dev = self.device
        
        # Defaults (all ones for B-field as requested, sensible values for others)
        defaults = {
            'Blos': 1.0,   # 500 G (normalized)
            'BQ': 1.0,     # 500 G
            'BU': 1.0,     # 500 G
            'vlos': 0.0,   # 0 km/s
            'eta0': 0.5,   # 50 (normalized by 100)
            'a': 0.5,      # 0.1 (normalized by 0.2)
            'ddop': 0.04,  # 0.04 Å
            'S0': 0.5,     # 0.5
            'S1': 0.5,     # 0.5
        }
        
        # Override with initial values if provided (convert to normalized)
        if initial_values:
            for key in defaults:
                if key in initial_values:
                    # Convert from physical to normalized
                    phys_val = initial_values[key]
                    if isinstance(phys_val, np.ndarray):
                        phys_val = phys_val.flatten()
                    defaults[key] = phys_val / self.norm[key]
        
        # Create parameters
        def _make_param(key):
            val = defaults[key]
            if isinstance(val, (int, float)):
                return nn.Parameter(torch.ones(n, device=dev) * val)
            else:
                return nn.Parameter(torch.from_numpy(val.astype(np.float32)).to(dev))
        
        self.Blos = _make_param('Blos')
        self.BQ = _make_param('BQ')
        self.BU = _make_param('BU')
        self.vlos = _make_param('vlos')
        self.eta0 = _make_param('eta0')
        self.a = _make_param('a')
        self.ddop = _make_param('ddop')
        self.S0 = _make_param('S0')
        self.S1 = _make_param('S1')
    
    def _cartesian_to_spherical(
        self, 
        Blos: torch.Tensor, 
        BQ: torch.Tensor, 
        BU: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian B-field to spherical (B, gamma, xi).
        
        Blos = B * cos(gamma)
        BQ = B * sin(gamma) * cos(2*xi)
        BU = B * sin(gamma) * sin(2*xi)
        
        Equivalently:
        Btrans = sqrt(BQ^2 + BU^2) = B * sin(gamma)
        B = sqrt(Blos^2 + Btrans^2)
        gamma = acos(Blos / B)
        xi = 0.5 * atan2(BU, BQ)
        """
        eps = 1e-15
        Btrans = torch.sqrt(BQ**2 + BU**2)
        B = torch.sqrt(Blos**2 + Btrans**2)
        
        # gamma in [0, pi]
        gamma = torch.acos(torch.clamp(Blos / (B + eps), -1 + eps, 1 - eps))
        
        # xi in [-pi/2, pi/2] from atan2
        xi = 0.5 * torch.atan2(BU, BQ + eps)
        
        return B, gamma, xi
    
    def forward(self) -> Atmosphere:
        """
        Compute physical parameters and return Atmosphere.
        
        Returns:
            Atmosphere object with physical parameters
        """
        # De-normalize to physical units
        Blos_phys = self.Blos * self.norm['Blos']
        BQ_phys = self.BQ * self.norm['BQ']
        BU_phys = self.BU * self.norm['BU']
        
        # Convert to spherical
        B, gamma, xi = self._cartesian_to_spherical(Blos_phys, BQ_phys, BU_phys)
        
        # Stack all 9 physical parameters
        # Order: [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
        params = torch.stack([
            B,
            gamma,
            xi,
            self.vlos * self.norm['vlos'],
            self.eta0 * self.norm['eta0'],
            self.a * self.norm['a'],
            self.ddop * self.norm['ddop'],
            self.S0 * self.norm['S0'],
            self.S1 * self.norm['S1']
        ], dim=-1)
        
        return Atmosphere(params, grid_shape=self.grid_shape)
    
    def clamp_parameters(self):
        """
        Apply physical constraints after optimizer step.
        
        Call this after optimizer.step() to keep parameters in valid ranges.
        """
        with torch.no_grad():
            # vlos: [-5, +5] km/s -> normalized by 2.0 -> [-2.5, 2.5]
            self.vlos.clamp_(-2.5, 2.5)
            
            # eta0: [1, 600] -> normalized by 100 -> [0.01, 6.0]
            self.eta0.clamp_(0.01, 6.0)
            
            # a (damp): [0.001, 1.0] -> normalized by 0.2 -> [0.005, 5.0]
            self.a.clamp_(0.005, 5.0)
            
            # ddop: [0.01, 0.2] Å (already in physical units if norm=1)
            self.ddop.clamp_(0.01, 0.2)
            
            # S0, S1: [0.01, 1.5]
            self.S0.clamp_(0.01, 1.5)
            self.S1.clamp_(0.01, 1.5)
    
    def get_cartesian_field(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Blos, BQ, BU in physical units."""
        return (
            self.Blos * self.norm['Blos'],
            self.BQ * self.norm['BQ'],
            self.BU * self.norm['BU']
        )
    
    def set_from_wfa(
        self, 
        Blos: np.ndarray, 
        BtQ: np.ndarray, 
        BtU: np.ndarray,
        S0: Optional[np.ndarray] = None,
        S1: Optional[np.ndarray] = None
    ):
        """
        Set magnetic field parameters from WFA estimates.
        
        Args:
            Blos: LOS field from WFA (Ny, Nx) in Gauss
            BtQ, BtU: Transverse field components from WFA (Ny, Nx) in Gauss
            S0, S1: Optional source function estimates
        """
        with torch.no_grad():
            self.Blos.data = torch.from_numpy(
                (Blos.flatten() / self.norm['Blos']).astype(np.float32)
            ).to(self.device)
            self.BQ.data = torch.from_numpy(
                (BtQ.flatten() / self.norm['BQ']).astype(np.float32)
            ).to(self.device)
            self.BU.data = torch.from_numpy(
                (BtU.flatten() / self.norm['BU']).astype(np.float32)
            ).to(self.device)
            
            if S0 is not None:
                self.S0.data = torch.from_numpy(
                    (S0.flatten() / self.norm['S0']).astype(np.float32)
                ).to(self.device)
            if S1 is not None:
                self.S1.data = torch.from_numpy(
                    (S1.flatten() / self.norm['S1']).astype(np.float32)
                ).to(self.device)


class StereoscopicParameterModel(ParameterModel):
    """
    Parameter model for coupled multi-instrument inversions.
    
    For stereoscopic observations (e.g., Hinode + SolO viewing the same region
    from different angles), this model:
    1. Shares a single magnetic field vector defined in Frame A (reference frame)
    2. Rotates this vector to Frame B using a specified rotation angle (gamma)
    3. Maintains independent thermodynamic parameters for each instrument
    
    This enables self-consistent inversions that constrain the 3D magnetic field
    using observations from multiple viewpoints.
    
    Optimization Parameters (15 total):
      Shared (3): Blos_A, Bn_A, Bw_A (magnetic field in Frame A)
      Instrument 1 (6): vlos1, eta01, a1, ddop1, S01, S11
      Instrument 2 (6): vlos2, eta02, a2, ddop2, S02, S12
      
    Optional: Calibration factor for stray light correction
    """
    
    def __init__(
        self,
        n_pixels: int,
        grid_shape: Tuple[int, int],
        gamma: float = 0.0,
        rotangle: float = 0.0,
        norm: Optional[Dict[str, float]] = None,
        device: torch.device = None,
        initial_values: Optional[Dict[str, np.ndarray]] = None,
        learn_calibration: bool = True,
        cali_init: float = 0.2
    ):
        """
        Initialize StereoscopicParameterModel.
        
        Args:
            n_pixels: Number of pixels (Ny * Nx)
            grid_shape: Spatial grid shape (Ny, Nx)
            gamma: Rotation angle from Frame A to Frame B (degrees)
            rotangle: Additional azimuth rotation for Frame B (degrees)
            norm: Normalization factors (defaults to DEFAULT_NORM values)
            device: Torch device
            initial_values: Optional dict of initial values (physical units)
            learn_calibration: If True, learn stray light calibration factor
            cali_init: Initial calibration factor
        """
        super().__init__()
        self.n_pixels = n_pixels
        self.grid_shape = grid_shape
        self.device = device or torch.device('cpu')
        
        # Convert angles to radians
        self.gamma = torch.tensor(np.radians(gamma), device=self.device)
        self.rotangle = torch.tensor(np.radians(rotangle), device=self.device)
        
        # Normalization: [Blos, Bn, Bw, vlos1, eta01, a1, ddop1, S01, S11, 
        #                                vlos2, eta02, a2, ddop2, S02, S12]
        self.norm = [500.0, 500.0, 500.0,   # B-field
                     2.0, 100.0, 0.2, 1.0, 1.0, 1.0,   # Inst 1 thermo
                     2.0, 100.0, 0.2, 1.0, 1.0, 1.0]   # Inst 2 thermo
        
        # Initialize parameters
        self._init_parameters(initial_values)
        
        # Calibration factor (for stray light correction on Instrument 2)
        if learn_calibration:
            self.cali = nn.Parameter(torch.tensor([cali_init], device=self.device))
        else:
            self.register_buffer('cali', torch.tensor([cali_init], device=self.device))
        self.learn_calibration = learn_calibration
    
    def _init_parameters(self, initial_values: Optional[Dict[str, np.ndarray]] = None):
        """Initialize learnable parameters."""
        n = self.n_pixels
        dev = self.device
        
        # Defaults (ones for B-field, sensible values for thermo)
        defaults = {
            # Shared magnetic field (Frame A)
            'Blos_A': 1.0, 'Bn_A': 1.0, 'Bw_A': 1.0,
            # Instrument 1 thermodynamic
            'vlos1': 0.0, 'eta01': 0.5, 'a1': 0.5, 'ddop1': 0.04, 'S01': 0.5, 'S11': 0.5,
            # Instrument 2 thermodynamic
            'vlos2': 0.0, 'eta02': 0.5, 'a2': 0.5, 'ddop2': 0.04, 'S02': 0.5, 'S12': 0.5,
        }
        
        # Override with initial values if provided
        if initial_values:
            for key in defaults:
                if key in initial_values:
                    phys_val = initial_values[key]
                    if isinstance(phys_val, np.ndarray):
                        phys_val = phys_val.flatten()
                    # Get norm index
                    norm_idx = list(defaults.keys()).index(key)
                    defaults[key] = phys_val / self.norm[norm_idx]
        
        def _make_param(key):
            val = defaults[key]
            if isinstance(val, (int, float)):
                return nn.Parameter(torch.ones(n, device=dev) * val)
            else:
                return nn.Parameter(torch.from_numpy(val.astype(np.float32)).to(dev))
        
        # Shared magnetic field (Frame A) - using Blos, Bn (north), Bw (west)
        self.Blos_A = _make_param('Blos_A')
        self.Bn_A = _make_param('Bn_A')
        self.Bw_A = _make_param('Bw_A')
        
        # Instrument 1 thermodynamic parameters
        self.vlos1 = _make_param('vlos1')
        self.eta01 = _make_param('eta01')
        self.a1 = _make_param('a1')
        self.ddop1 = _make_param('ddop1')
        self.S01 = _make_param('S01')
        self.S11 = _make_param('S11')
        
        # Instrument 2 thermodynamic parameters
        self.vlos2 = _make_param('vlos2')
        self.eta02 = _make_param('eta02')
        self.a2 = _make_param('a2')
        self.ddop2 = _make_param('ddop2')
        self.S02 = _make_param('S02')
        self.S12 = _make_param('S12')
    
    def _cartesian_to_spherical(
        self,
        Blos: torch.Tensor,
        Bn: torch.Tensor,
        Bw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert Cartesian (Blos, Bn, Bw) to spherical (B, inclination, azimuth)."""
        eps = 1e-7
        Btrans = torch.sqrt(Bn**2 + Bw**2)
        B = torch.sqrt(Blos**2 + Btrans**2)
        
        inclination = torch.acos(torch.clamp(Blos / (B + eps), -1 + eps, 1 - eps))
        azimuth = torch.atan2(Bn, Bw + eps)
        
        return B, inclination, azimuth
    
    def _rotate_frame(
        self,
        Blos_A: torch.Tensor,
        Bn_A: torch.Tensor,
        Bw_A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rotate magnetic field vector from Frame A to Frame B.
        
        The transformation applies a rotation around the Bn axis by angle gamma.
        This models the different viewing geometry between two instruments.
        """
        cos_g = torch.cos(self.gamma)
        sin_g = torch.sin(self.gamma)
        
        Blos_B = Blos_A * cos_g + Bw_A * sin_g
        Bn_B = Bn_A  # Unchanged
        Bw_B = -Blos_A * sin_g + Bw_A * cos_g
        
        return Blos_B, Bn_B, Bw_B
    
    def forward(self) -> Tuple[Atmosphere, Atmosphere]:
        """
        Compute physical parameters and return two Atmosphere objects.
        
        Returns:
            Tuple of (Atmosphere_A, Atmosphere_B) for the two instruments
        """
        # De-normalize magnetic field (Frame A)
        Blos_A = self.Blos_A * self.norm[0]
        Bn_A = self.Bn_A * self.norm[1]
        Bw_A = self.Bw_A * self.norm[2]
        
        # === Frame A (Instrument 1) ===
        B_A, inc_A, azi_A = self._cartesian_to_spherical(Blos_A, Bn_A, Bw_A)
        
        params1 = torch.stack([
            B_A,
            inc_A,
            azi_A,
            self.vlos1 * self.norm[3],
            self.eta01 * self.norm[4],
            self.a1 * self.norm[5],
            self.ddop1 * self.norm[6],
            self.S01 * self.norm[7],
            self.S11 * self.norm[8]
        ], dim=-1)
        
        # === Frame B (Instrument 2) ===
        Blos_B, Bn_B, Bw_B = self._rotate_frame(Blos_A, Bn_A, Bw_A)
        B_B, inc_B, azi_B = self._cartesian_to_spherical(Blos_B, Bn_B, Bw_B)
        azi_B = azi_B + self.rotangle  # Apply additional azimuth rotation
        
        params2 = torch.stack([
            B_B,
            inc_B,
            azi_B,
            self.vlos2 * self.norm[9],
            self.eta02 * self.norm[10],
            self.a2 * self.norm[11],
            self.ddop2 * self.norm[12],
            self.S02 * self.norm[13],
            self.S12 * self.norm[14]
        ], dim=-1)
        
        atm1 = Atmosphere(params1, grid_shape=self.grid_shape)
        atm2 = Atmosphere(params2, grid_shape=self.grid_shape)
        
        return atm1, atm2
    
    def clamp_parameters(self):
        """Apply physical constraints after optimizer step."""
        with torch.no_grad():
            # Velocity: [-5, +5] km/s -> normalized by 2.0
            self.vlos1.clamp_(-2.5, 2.5)
            self.vlos2.clamp_(-2.5, 2.5)
            
            # eta0: [1, 600] -> normalized by 100
            self.eta01.clamp_(0.01, 6.0)
            self.eta02.clamp_(0.01, 6.0)
            
            # Damping: [0.001, 1.0] -> normalized by 0.2
            self.a1.clamp_(0.005, 5.0)
            self.a2.clamp_(0.005, 5.0)
            
            # Doppler width: [0.01, 0.2] Å
            self.ddop1.clamp_(0.01, 0.2)
            self.ddop2.clamp_(0.01, 0.2)
            
            # Source function: [0.01, 1.5]
            self.S01.clamp_(0.01, 1.5)
            self.S02.clamp_(0.01, 1.5)
            self.S11.clamp_(0.01, 1.5)
            self.S12.clamp_(0.01, 1.5)
            
            # Calibration: [0, 1]
            if self.learn_calibration:
                self.cali.clamp_(0.0, 1.0)
    
    def get_shared_field(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return shared magnetic field (Blos_A, Bn_A, Bw_A) in physical units."""
        return (
            self.Blos_A * self.norm[0],
            self.Bn_A * self.norm[1],
            self.Bw_A * self.norm[2]
        )
    
    def get_calibration(self) -> float:
        """Return current calibration factor."""
        return self.cali.item()

