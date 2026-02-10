"""
Cartesian Atmosphere Implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .base import Atmosphere
from .base import Atmosphere

class CartesianAtmosphere(Atmosphere):
    """
    Atmosphere parameterized by Cartesian Magnetic Field components (Blos, BQ, BU).
    
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
        init_type: str = 'default'
    ):
        """
        Args:
            grid_shape: Shape of the spatial grid (e.g., (Ny, Nx)).
            mask: Optional boolean mask. If provided, parameters are only allocated
                  for True pixels (flattened). Useful for active regions.
            norm: Custom normalization factors.
            device: Torch device.
            init_type: Initialization strategy ('default', 'zeros').
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.device = device or torch.device('cpu')
        
        # Determine number of pixels/elements
        if mask is not None:
            self.n_elements = mask.sum().item()
            self.is_masked = True
            self.register_buffer('mask', mask)
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
        
    def _init_params(self, init_type):
        # Helper to create parameter
        def make_param(val):
            return nn.Parameter(torch.ones(self.n_elements, device=self.device) * val)
            
        # Calculate initial normalized values
        d = self.defaults.copy()
        if init_type == 'zeros':
            for k in d: d[k] = 0.0
            
        self._Blos = make_param(d['Blos'] / self.norm['Blos'])
        self._BQ   = make_param(d['BQ']   / self.norm['BQ'])
        self._BU   = make_param(d['BU']   / self.norm['BU'])
        self._vlos = make_param(d['vlos'] / self.norm['vlos'])
        self._eta0 = make_param(d['eta0'] / self.norm['eta0'])
        self._a    = make_param(d['a']    / self.norm['a'])
        self._ddop = make_param(d['ddop'] / self.norm['ddop'])
        self._S0   = make_param(d['S0']   / self.norm['S0'])
        self._S1   = make_param(d['S1']   / self.norm['S1'])
        
    def _expand(self, tensor):
        """Reshapes flat parameter tensor to grid_shape if not masked."""
        if self.is_masked:
            # If masked, we can't easily reshape to grid_shape without filling
            # For now, return flat, MEProblem must handle (it does usually handle flat)
            return tensor
        else:
            return tensor.view(*self.grid_shape)

    # --- Properties ---

    @property
    def B(self) -> torch.Tensor:
        Blos = self._Blos * self.norm['Blos']
        BQ = self._BQ * self.norm['BQ']
        BU = self._BU * self.norm['BU']
        return self._expand(torch.sqrt(Blos**2 + BQ**2 + BU**2))

    @property
    def gamma(self) -> torch.Tensor:
        Blos = self._Blos * self.norm['Blos']
        # B is computed again here (autodiff handles graph)
        # We could cache B, but it's cheap enough
        B = self.B.flatten() if not self.is_masked else self.B
        eps = 1e-8
        return self._expand(torch.acos(torch.clamp(Blos / (B + eps), -1+eps, 1-eps)))

    @property
    def xi(self) -> torch.Tensor:
        BQ = self._BQ * self.norm['BQ']
        BU = self._BU * self.norm['BU']
        # xi = 0.5 * atan2(BU, BQ) -> result in [-pi/2, pi/2]
        return self._expand(0.5 * torch.atan2(BU, BQ + 1e-15))

    @property
    def vlos(self) -> torch.Tensor:
        return self._expand(self._vlos * self.norm['vlos'])

    @property
    def eta0(self) -> torch.Tensor:
        return self._expand(self._eta0 * self.norm['eta0'])

    @property
    def a(self) -> torch.Tensor:
        return self._expand(self._a * self.norm['a'])

    @property
    def ddop(self) -> torch.Tensor:
        return self._expand(self._ddop * self.norm['ddop'])

    @property
    def S0(self) -> torch.Tensor:
        return self._expand(self._S0 * self.norm['S0'])

    @property
    def S1(self) -> torch.Tensor:
        return self._expand(self._S1 * self.norm['S1'])
        
    # --- Methods ---
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from physical values."""
        with torch.no_grad():
            for k, v in params.items():
                # Value 'v' is physical
                if isinstance(v, (np.ndarray, float, int)):
                    t = torch.as_tensor(v, dtype=torch.float32, device=self.device).flatten()
                elif isinstance(v, torch.Tensor):
                    t = v.to(self.device).flatten()
                else:
                    continue
                
                # Assign to correct internal parameter
                # Cartesian inputs
                if k == 'Blos': self._Blos.copy_(t / self.norm['Blos'])
                elif k == 'BQ': self._BQ.copy_(t / self.norm['BQ'])
                elif k == 'BU': self._BU.copy_(t / self.norm['BU'])
                # Thermo inputs
                elif k == 'vlos': self._vlos.copy_(t / self.norm['vlos'])
                elif k == 'eta0': self._eta0.copy_(t / self.norm['eta0'])
                elif k == 'a':    self._a.copy_(t / self.norm['a'])
                elif k == 'ddop': self._ddop.copy_(t / self.norm['ddop'])
                elif k == 'S0':   self._S0.copy_(t / self.norm['S0'])
                elif k == 'S1':   self._S1.copy_(t / self.norm['S1'])
                
    def clamp_parameters(self):
        with torch.no_grad():
            # Physical range limits -> Normalized
            # vlos [-5, 5] -> norm 2 -> [-2.5, 2.5]
            self._vlos.clamp_(-2.5, 2.5)
            # eta0 [0.01, 600] -> norm 100 -> [0.0001, 6.0]
            self._eta0.clamp_(0.0001, 6.0)
            # a [0.001, 1.0] -> norm 0.2 -> [0.005, 5.0]
            self._a.clamp_(0.005, 5.0)
            # ddop [0.01, 0.2] -> norm 1.0 (usually) -> [0.01, 0.2]
            self._ddop.clamp_(0.01, 0.2)
            # Source functions > 0
            self._S0.clamp_(min=0.01)
            self._S1.clamp_(min=0.01)
