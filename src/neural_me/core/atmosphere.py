
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Union

class Atmosphere:
    """
    Represents the solar atmosphere parameters for Milne-Eddington inversion.
    
    Acts as the central data structure for the model state, handling:
    1. Storage of normalized parameters.
    2. Conversion to physical units.
    3. Reshaping and device management.
    
    The 9 parameters are:
    0: B (Magnetic Field Strength) [kG]
    1: Gamma (Inclination) [rad]
    2: Xi (Azimuth) [rad]
    3: Vlos (Line-of-Sight Velocity) [km/s]
    4: Eta0 (Line-to-Continuum Absorption Ratio) []
    5: a (Damping) [] 
    6: Ddop (Doppler Width) [Angstrom]
    7: S0 (Source Function Constant) []
    8: S1 (Source Function Gradient) []
    """
    
    # Parameter Indices
    IDX_B = 0
    IDX_GAMMA = 1
    IDX_XI = 2
    IDX_VLOS = 3
    IDX_ETA0 = 4
    IDX_A = 5
    IDX_DDOP = 6
    IDX_S0 = 7
    IDX_S1 = 8
    
    def __init__(
        self,
        params: torch.Tensor,
        parameter_weights: Optional[Dict[str, float]] = None,
        parameter_biases: Optional[Dict[str, float]] = None,
        grid_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize from NORMALIZED parameters.
        
        Args:
            params: Tensor of shape (..., 9). Normalized parameters.
            parameter_weights: Scaling factors for denormalization (Physical = Norm * Scale + Bias).
            parameter_biases: Offsets for denormalization.
            grid_shape: Spatial grid shape (e.g., (Ny, Nx) or (Nt, Ny, Nx)).
        """
        self._params_norm = params
        self.grid_shape = grid_shape
        
        # Default Weights (Scaling factors for normalization)
        # By default, no scaling (all 1.0). Actual normalization is defined in MEProblem.
        self.weights = {
            'B': 1.0,
            'gamma': 1.0,
            'xi': 1.0,
            'vlos': 1.0,
            'eta0': 1.0,
            'a': 1.0,
            'ddop': 1.0,
            'S0': 1.0,
            'S1': 1.0
        }
        if parameter_weights:
            self.weights.update(parameter_weights)
            
        self.biases = {
            'B': 0.0,
            'gamma': 0.0,
            'xi': 0.0,
            'vlos': 0.0,
            'eta0': 0.0,
            'a': 0.0,
            'ddop': 0.0,
            'S0': 0.0,
            'S1': 0.0
        }
        if parameter_biases:
            self.biases.update(parameter_biases)
            
    @property
    def device(self):
        return self._params_norm.device
        
    @property
    def shape(self):
        return self._params_norm.shape[:-1]

    def _denorm(self, idx, name):
        val = self._params_norm[..., idx]
        w = self.weights.get(name, 1.0)
        b = self.biases.get(name, 0.0)
        return val * w + b

    # --- Physical Quantities ---
    
    @property
    def B(self) -> torch.Tensor:
        """Magnetic Field Strength [kG]"""
        return self._denorm(self.IDX_B, 'B')

    @property
    def gamma(self) -> torch.Tensor:
        """Inclination [rad]"""
        return self._denorm(self.IDX_GAMMA, 'gamma')

    @property
    def xi(self) -> torch.Tensor:
        """Azimuth [rad]"""
        return self._denorm(self.IDX_XI, 'xi')

    @property
    def vlos(self) -> torch.Tensor:
        """Line-of-Sight Velocity [km/s]"""
        return self._denorm(self.IDX_VLOS, 'vlos')

    @property
    def eta0(self) -> torch.Tensor:
        """Line-to-Continuum Ratio []"""
        return self._denorm(self.IDX_ETA0, 'eta0')

    @property
    def a(self) -> torch.Tensor:
        """Damping []"""
        return self._denorm(self.IDX_A, 'a')

    @property
    def ddop(self) -> torch.Tensor:
        """Doppler Width [Angstrom]"""
        return self._denorm(self.IDX_DDOP, 'ddop')

    @property
    def S0(self) -> torch.Tensor:
        """Source Function Constant []"""
        return self._denorm(self.IDX_S0, 'S0')

    @property
    def S1(self) -> torch.Tensor:
        """Source Function Gradient []"""
        return self._denorm(self.IDX_S1, 'S1')

    # --- Utilities ---
    
    def to_stack(self) -> torch.Tensor:
        """Returns the stack of PHYSICAL parameters [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]"""
        return torch.stack([
            self.B, self.gamma, self.xi, self.vlos, self.eta0, 
            self.a, self.ddop, self.S0, self.S1
        ], dim=-1)

    def to_numpy_stack(self) -> np.ndarray:
        """
        Returns physical parameters as a numpy array, reshaped to grid_shape if available.
        Shape: (Ny, Nx, 9) or (Nt, Ny, Nx, 9)
        Includes standard post-processing: B > 0, Xi in [0, pi].
        """
        stack = self.to_stack().detach().cpu().numpy()
        
        # Reshape if grid is defined
        if self.grid_shape is not None:
            # stack is (N_pixels, 9) -> (..., 9)
            stack = stack.reshape(*self.grid_shape, 9)
            
        # Post-processing
        # 1. B (Index 0) should be positive
        stack[..., 0] = np.abs(stack[..., 0])
        
        # 2. Xi / PhiB (Index 2) should be in [0, pi]
        # ME model depends on 2*xi, so corrections of +/- pi are valid.
        phiB = stack[..., 2]
        phiB[phiB < 0] += np.pi
        phiB[phiB > np.pi] -= np.pi
        stack[..., 2] = phiB
        
        return stack

    def to(self, device):
        return Atmosphere(
            self._params_norm.to(device),
            self.weights,
            self.biases,
            self.grid_shape
        )
        
    def detach(self):
        return Atmosphere(
            self._params_norm.detach(),
            self.weights,
            self.biases,
            self.grid_shape
        )
        
    def requires_grad_(self, requires_grad=True):
        self._params_norm.requires_grad_(requires_grad)
        return self
