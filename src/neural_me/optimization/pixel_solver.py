
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from typing import Optional, Dict

from neural_me.core.problem import MEProblem
from neural_me.core.atmosphere import Atmosphere

class PixelSolver:
    """
    Solver for Explicit (Pixel-wise) Milne-Eddington Inversion.
    
    Optimizes atmospheric parameters directly for each pixel (no neural network).
    Useful for:
    1. Validating the forward model and gradients.
    2. Generating "Ground Truth" or reference inversions for training.
    3. Solving small patches where training a NN is overkill.
    """
    
    def __init__(
        self,
        problem: MEProblem,
        device: Optional[torch.device] = None
    ):
        self.problem = problem
        self.device = device if device else problem.device
        
        # Determine number of pixels/samples
        self.n_pixels = self.problem.obs.n_pixels
        self.grid_shape = self.problem.obs.grid_shape
        
        # Parameters: (N_pixels, 9)
        self.params = torch.full(
            (self.n_pixels, 9), 
            0.5, 
            device=self.device, 
            dtype=torch.float32, 
            requires_grad=True
        )
        
    def initialize_parameters(self, method: str = 'default'):
        """
        Initialize parameters intelligently.
        """
        with torch.no_grad():
            if method == 'zeros':
                self.params.fill_(0.0)
            elif method == 'mean':
                self.params.fill_(0.5) 
                self.params.add_(torch.randn_like(self.params) * 0.01)
            elif method == 'default':
                # Legacy initialization values (Physical approximations)

                # B (0.5 kG)
                self.params[:, 0] = 0.5 
                # Gamma (1.0 rad)
                self.params[:, 1] = 1.0
                # Xi (1.0 rad)
                self.params[:, 2] = 1.0
                # Vlos (0.0 km/s)
                self.params[:, 3] = 0.0
                # Eta0 (50.0)
                self.params[:, 4] = 50.0
                # a (0.2)
                self.params[:, 5] = 0.2
                # Ddop (0.02)
                self.params[:, 6] = 0.02
                # S0 (0.5)
                self.params[:, 7] = 0.5
                # S1 (0.5)
                self.params[:, 8] = 0.5
                
                self.params.requires_grad = True

            
    def solve(
        self,
        n_iterations: int = 100,
        lr: float = 1e-2,
        optimizer_cls: torch.optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = None,
        verbose: bool = True
    ):
        """
        Runs the optimization loop.
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            
        optimizer = optimizer_cls([self.params], lr=lr, **optimizer_kwargs)
        
        iterator = trange(n_iterations, leave=True) if verbose else range(n_iterations)
        
        for i in iterator:
            optimizer.zero_grad()
            
            # 1. Create Atmosphere from current parameters 
            # We treat params as the normalized values directly.
            normalized_params = self.params
            
            atmosphere = Atmosphere(
                normalized_params,
                grid_shape=self.grid_shape
            )
            
            # 2. Compute Loss
            loss, breakdown = self.problem.compute_loss(
                atmosphere,
                return_breakdown=True
            )
            
            # 3. Backward
            loss.backward()
            
            # 4. Step
            optimizer.step()
            
            # 5. Logging
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'Chi2': f"{loss.item():.2e}",
                    # Optional: Show breakdown if useful, but user requested focus on total loss
                    #'I': f"{breakdown['loss_I']:.1e}", 
                    #'V': f"{breakdown['loss_V']:.1e}"
                })
                
    def get_atmosphere(self) -> Atmosphere:
        """Returns the optimized Atmosphere."""
        with torch.no_grad():
            normalized_params = self.params
            return Atmosphere(normalized_params, grid_shape=self.grid_shape)
