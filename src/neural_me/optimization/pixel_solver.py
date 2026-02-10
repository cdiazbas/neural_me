
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from typing import Optional, Dict, Union

from neural_me.core.problem import MEProblem
from neural_me.core.atmospheres import Atmosphere, CartesianAtmosphere


class PixelSolver:
    """
    Solver for Explicit (Pixel-wise) Milne-Eddington Inversion.
    
    Optimizes atmospheric parameters directly for each pixel (no neural network).
    Requires a valid Atmosphere object to optimize.
    """
    
    def __init__(
        self,
        problem: MEProblem,
        atmosphere: Optional[Atmosphere] = None,
        use_cartesian: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the PixelSolver.
        
        Args:
            problem: MEProblem instance.
            atmosphere: Atmosphere instance to optimize.
            use_cartesian: If True and atmosphere is None, creates CartesianAtmosphere.
            device: Torch device.
        """
        self.problem = problem
        
        # Use problem's device by default - this respects C++ backend's CPU requirement
        if device is not None and hasattr(problem, 'backend') and problem.backend == 'cpp':
            import warnings
            device_str = str(device) if isinstance(device, torch.device) else device
            if 'cuda' in device_str.lower():
                warnings.warn(
                    f"PixelSolver device overridden to CPU (C++ backend requires CPU, "
                    f"requested: {device})"
                )
        self.device = problem.device  # Always use problem's device for consistency
        
        # Determine number of pixels/samples
        self.n_pixels = self.problem.obs.n_pixels
        self.grid_shape = self.problem.obs.grid_shape
        
        # Initialize parameter model / atmosphere
        self.atmosphere = None

        if atmosphere is not None:
             self.atmosphere = atmosphere
        elif use_cartesian:
            # Unified: Create CartesianAtmosphere directly
            self.atmosphere = CartesianAtmosphere(
                grid_shape=self.grid_shape,
                device=self.device
            )
        else:
            raise ValueError(
                "PixelSolver requires either an 'atmosphere' object or 'use_cartesian=True'. "
                "Legacy raw tensor mode is no longer supported."
            )
        
    def set_from_wfa(self, Blos: np.ndarray, BtQ: np.ndarray, BtU: np.ndarray,
                     S0: Optional[np.ndarray] = None, S1: Optional[np.ndarray] = None):
        """
        Set B-field parameters from WFA estimates.
        """
        if isinstance(self.atmosphere, Atmosphere):
             # Unified Atmosphere
             self.atmosphere.set_parameters({
                 'Blos': Blos, 'BQ': BtQ, 'BU': BtU,
                 'S0': S0, 'S1': S1
             })
             return
        
    def solve(
        self,
        n_iterations: int = 100,
        lr: float = 1e-2,
        optimizer_cls: torch.optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = None,
        verbose: bool = True,
        normalize_gradients: bool = False,
        clamp_parameters: bool = True
    ):
        """
        Runs the optimization loop.
        
        Args:
            n_iterations: Number of optimization iterations
            lr: Learning rate
            optimizer_cls: Optimizer class (default: Adam)
            optimizer_kwargs: Additional optimizer arguments
            verbose: Show progress bar
            normalize_gradients: Deprecated. Gradient normalization is removed.
            clamp_parameters: Apply parameter clamping after each step
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        # Get parameters to optimize
        params_to_optimize = list(self.atmosphere.parameters())
            
        optimizer = optimizer_cls(params_to_optimize, lr=lr, **optimizer_kwargs)
        
        iterator = trange(n_iterations, leave=True) if verbose else range(n_iterations)
        
        for i in iterator:
            optimizer.zero_grad()
            
            # 1. Compute Loss
            # PixelSolver passes the FULL atmosphere usually.
            loss, breakdown = self.problem.compute_loss(
                self.atmosphere,
                return_breakdown=True
            )
            
            # 2. Backward
            loss.backward()
            
            # 3. Optimizer step
            optimizer.step()
            
            # 4. Parameter clamping
            if clamp_parameters:
                self.atmosphere.clamp_parameters()
            
            # 5. Logging
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'Chi2': f"{loss.item():.2e}",
                })
                
    def get_atmosphere(self) -> Atmosphere:
        """Returns the optimized Atmosphere."""
        return self.atmosphere
