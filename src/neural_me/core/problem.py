
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict

from neural_me.core.observation import Observation
from neural_me.core.atmosphere import Atmosphere
from neural_me.physics.milne_eddington import MilneEddington

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
        device: Optional[torch.device] = None
    ):
        """
        Args:
            observation: The observed data.
            nlinea: Index for parameter line configuration.
            weights: Loss weights for [Stokes I, Q, U, V].
            loss_type: 'mse' (Mean Squared Error) or 'l1' (Mean Absolute Error).
        """
        self.obs = observation
        self.device = device if device else observation.device
        self.loss_type = loss_type.lower()
        
        # Ensure obs is on device
        if self.obs.device != self.device:
            self.obs = self.obs.to(self.device)
            
        # Initialize Physics Model
        self.model = MilneEddington(nlinea=nlinea)
        self.model.to(self.device)
        
        # Loss Weights
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
    def compute_forward_model(self, atmosphere: Atmosphere) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the forward physics model on the given atmosphere.
        
        Returns:
            (I, Q, U, V) tensors of shape (..., n_lambda)
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
                
        if return_breakdown:
            return loss, {
                'loss_I': err_I.item(),
                'loss_Q': err_Q.item(),
                'loss_U': err_U.item(),
                'loss_V': err_V.item()
            }
            
        return loss
