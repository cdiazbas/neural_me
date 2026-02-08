
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict

from neural_me.core.observation import Observation
from neural_me.core.atmosphere import Atmosphere
from neural_me.physics.milne_eddington import MilneEddington
from neural_me.regularization.spatial import smoothness_loss

class MEProblem:
    """
    Defines the Milne-Eddington (ME) inversion problem.
    
    1. Holds the Observation (Data).
    2. Initializes the Forward Model (Physics).
    3. Computes the Loss (Data vs Model).
    """
    
    def __init__(
        self,
        observation: Observation,
        nlinea: int = 4, 
        weights: List[float] = [1.0, 1.0, 1.0, 1.0], 
        loss_type: str = 'mse',
        regularization_weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            observation: The observed data.
            nlinea: Index for parameter line configuration.
            weights: Loss weights for [Stokes I, Q, U, V].
            loss_type: 'mse' or 'l1'.
            regularization_weights: Dict mapping parameter names (or 'all') to spatial smoothing weights.
                                    Example: {'all': 1e-3} or {'B': 1e-2, 'vlos': 1e-3}.
        """
        self.obs = observation
        self.device = device if device else observation.device
        self.loss_type = loss_type.lower()
        self.regularization_weights = regularization_weights
        
        # Ensure obs is on device
        if self.obs.device != self.device:
            self.obs = self.obs.to(self.device)
            
        # Initialize Physics Model
        self.model = MilneEddington(nlinea=nlinea)
        self.model.to(self.device)
        
        # Parameter normalization weights (for scaling normalized [0,1] params to physical units)
        # These values were empirically optimized in neural_me_legacy
        self.normalization = {
            'B': 500.0,      # Magnetic field (G): [0,1] → [0, 500 G]
            'gamma': 1.0,    # Inclination (rad): [0,1] → [0, π]
            'xi': 1.0,       # Azimuth (rad): [0,1] → [0, 2π]
            'vlos': 2.0,     # Velocity (km/s): [0,1] → [0, 2 km/s]
            'eta0': 100.0,   # Line-to-continuum ratio: [0,1] → [0, 100]
            'a': 0.2,        # Damping parameter: [0,1] → [0, 0.2]
            'ddop': 1.0,     # Doppler width (Å): [0,1] → [0, 1 Å]
            'S0': 1.0,       # Source function (continuum): [0,1] → [0, 1]
            'S1': 1.0        # Source function (gradient): [0,1] → [0, 1]
        }
        
        # Loss Weights
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

    def _compute_regularization(self, atmosphere: Atmosphere) -> torch.Tensor:
        """Computes spatial regularization loss."""
        if not self.regularization_weights:
            return torch.tensor(0.0, device=self.device)

        # We need the full grid for spatial regularization
        # If atmosphere is a subset (indices passed to compute_loss), spatial reg might be invalid
        # BUT PixelSolver passes the FULL atmosphere usually.
        # If compute_loss received indices, we probably shouldn't compute spatial reg on the subset 
        # unless it's a contiguous patch. 
        # For now, assume atmosphere has grid_shape populated if we want spatial reg.
        
        if atmosphere.grid_shape is None:
            return torch.tensor(0.0, device=self.device)
            
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # Parameters to regularize
        # Atmosphere has 9 params.
        # We can key them by name.
        param_names = ['B', 'gamma', 'xi', 'vlos', 'eta0', 'a', 'ddop', 'S0', 'S1']
        
        # Get stack: (Ny, Nx, 9)
        stack = atmosphere.to_stack() 
        if stack.dim() != 3: # Expect 2D map + params dim
             # Possibly (Batch, Ny, Nx, 9) ?
             if stack.dim() == 4:
                 pass # OK
             else:
                 return torch.tensor(0.0, device=self.device)

        # Reshape to (C, Ny, Nx) or (Batch, C, Ny, Nx) for smoothness_loss
        # stack is (..., Ny, Nx, 9) -> permute to (..., 9, Ny, Nx)
        stack_perm = stack.permute(0, 3, 1, 2) if stack.dim() == 4 else stack.permute(2, 0, 1)
        
        # Global weight
        if 'all' in self.regularization_weights:
            w = self.regularization_weights['all']
            reg_loss += w * smoothness_loss(stack_perm, penalty=self.loss_type)
            
        # Specific weights
        for i, name in enumerate(param_names):
            if name in self.regularization_weights:
                w = self.regularization_weights[name]
                # Slice specific channel
                param_map = stack_perm[..., i, :, :]
                reg_loss += w * smoothness_loss(param_map, penalty=self.loss_type)
                
        return reg_loss

    def compute_forward_model(self, atmosphere: Atmosphere) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the forward physics model on the given atmosphere.
        Returns: (I, Q, U, V) tensors of shape (..., n_lambda)
        """
        # Get physical parameters stack [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
        params = atmosphere.to_stack() # Shape (..., 9)
        
        # Broadcast wavelengths to match batch size
        batch_size = params.shape[0] if params.dim() > 1 else 1
        wl = self.obs.wavelengths.unsqueeze(0).expand(batch_size, -1)
        
        # Check if params is 1D (single pixel) -> make it 2D (1, 9)
        if params.dim() == 1:
            params = params.unsqueeze(0)
            
        I, Q, U, V = self.model(params, wl)
        
        return I, Q, U, V
        
    def compute_loss(
        self,
        atmosphere: Atmosphere,
        indices: Optional[torch.Tensor] = None,
        return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Computes weighted loss (MSE or L1 based on self.loss_type).
        """
        # 1. Forward Pass
        if indices is not None:
             params_subset = atmosphere._params_norm[indices]
             atm_subset = Atmosphere(
                 params_subset, 
                 atmosphere.weights, 
                 atmosphere.biases,
                 grid_shape=None 
             )
             I_mod, Q_mod, U_mod, V_mod = self.compute_forward_model(atm_subset)
             
             obs_I = self.obs.stokes_I[indices]
             obs_Q = self.obs.stokes_Q[indices]
             obs_U = self.obs.stokes_U[indices]
             obs_V = self.obs.stokes_V[indices]
        else:
             I_mod, Q_mod, U_mod, V_mod = self.compute_forward_model(atmosphere)
             obs_I = self.obs.stokes_I
             obs_Q = self.obs.stokes_Q
             obs_U = self.obs.stokes_U
             obs_V = self.obs.stokes_V
             
        # 2. Masking (Wavelengths)
        mask = self.obs.active_wav_idx
        
        # 3. Residuals
        res_I = (I_mod - obs_I)[..., mask]
        res_Q = (Q_mod - obs_Q)[..., mask]
        res_U = (U_mod - obs_U)[..., mask]
        res_V = (V_mod - obs_V)[..., mask]
        
        # 4. Error Calculation per Stokes
        if self.loss_type == 'l1':
            err_I = torch.mean(torch.abs(res_I))
            err_Q = torch.mean(torch.abs(res_Q))
            err_U = torch.mean(torch.abs(res_U))
            err_V = torch.mean(torch.abs(res_V))
        else: # 'mse' default
            err_I = torch.mean(res_I**2)
            err_Q = torch.mean(res_Q**2)
            err_U = torch.mean(res_U**2)
            err_V = torch.mean(res_V**2)
        
        # 5. Weighted Sum
        loss = (self.weights[0] * err_I + 
                self.weights[1] * err_Q + 
                self.weights[2] * err_U + 
                self.weights[3] * err_V)
        
        # 6. Regularization
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.regularization_weights and indices is None: 
            # Only compute reg if we are processing the full grid/frame
            # otherwise spatial topology is lost or incomplete
            reg_loss = self._compute_regularization(atmosphere)
            loss += reg_loss
                
        if return_breakdown:
            return loss, {
                'loss_I': err_I.item(),
                'loss_Q': err_Q.item(),
                'loss_U': err_U.item(),
                'loss_V': err_V.item(),
                'loss_reg': reg_loss.item()
            }
            
        return loss
