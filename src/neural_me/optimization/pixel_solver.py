
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from typing import Optional, Dict, Union

from neural_me.core.problem import MEProblem
from neural_me.core.atmosphere import Atmosphere
from neural_me.core.parameters import ParameterModel, CartesianParameterModel


class PixelSolver:
    """
    Solver for Explicit (Pixel-wise) Milne-Eddington Inversion.
    
    Optimizes atmospheric parameters directly for each pixel (no neural network).
    Supports two modes:
    1. Legacy mode: Raw normalized parameters (original behavior)
    2. ParameterModel mode: Uses CartesianParameterModel for Blos/BQ/BU optimization
    
    Useful for:
    - Validating the forward model and gradients.
    - Generating "Ground Truth" or reference inversions for training.
    - Solving small patches where training a NN is overkill.
    """
    
    def __init__(
        self,
        problem: MEProblem,
        param_model: Optional[ParameterModel] = None,
        use_cartesian: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the PixelSolver.
        
        Args:
            problem: MEProblem instance (contains observation and forward model)
            param_model: Optional ParameterModel instance. If provided, uses it.
            use_cartesian: If True and param_model is None, creates CartesianParameterModel.
            device: Torch device for computation
        """
        self.problem = problem
        self.device = device if device else problem.device
        
        # Determine number of pixels/samples
        self.n_pixels = self.problem.obs.n_pixels
        self.grid_shape = self.problem.obs.grid_shape
        
        # Optimization options
        self.normalize_gradients = False
        self.do_clamping = False
        
        # Initialize parameter model
        if param_model is not None:
            self.param_model = param_model
            self._use_param_model = True
        elif use_cartesian:
            self.param_model = CartesianParameterModel(
                n_pixels=self.n_pixels,
                grid_shape=self.grid_shape,
                device=self.device
            )
            self._use_param_model = True
        else:
            # Legacy mode: raw normalized parameters
            self.param_model = None
            self._use_param_model = False
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
        
        For ParameterModel mode, this is handled by the model itself.
        For legacy mode, initializes normalized parameters.
        """
        if self._use_param_model:
            # ParameterModel handles its own initialization
            return
        
        with torch.no_grad():
            if method == 'zeros':
                self.params.fill_(0.0)
            elif method == 'mean':
                self.params.fill_(0.5) 
                self.params.add_(torch.randn_like(self.params) * 0.01)
            elif method == 'default':
                # Initialize with reasonable normalized values
                self.params[:, 0] = 1.0   # B: 500 G
                self.params[:, 1] = 0.5   # Gamma: π/2
                self.params[:, 2] = 0.5   # Xi: π
                self.params[:, 3] = 0.0   # Vlos: 0 km/s
                self.params[:, 4] = 0.5   # Eta0: 50
                self.params[:, 5] = 0.5   # a: 0.1
                self.params[:, 6] = 0.02  # Ddop: 0.02
                self.params[:, 7] = 0.5   # S0
                self.params[:, 8] = 0.5   # S1
                
                self.params.requires_grad = True
    
    def set_from_wfa(self, Blos: np.ndarray, BtQ: np.ndarray, BtU: np.ndarray,
                     S0: Optional[np.ndarray] = None, S1: Optional[np.ndarray] = None):
        """
        Set B-field parameters from WFA estimates.
        
        Only works in ParameterModel mode.
        """
        if not self._use_param_model:
            raise RuntimeError("set_from_wfa only works with ParameterModel mode (use_cartesian=True)")
        self.param_model.set_from_wfa(Blos, BtQ, BtU, S0, S1)
        
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
            normalize_gradients: Apply gradient normalization trick (legacy behavior)
            clamp_parameters: Apply parameter clamping after each step
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        # Get parameters to optimize
        if self._use_param_model:
            params_to_optimize = list(self.param_model.parameters())
        else:
            params_to_optimize = [self.params]
            
        optimizer = optimizer_cls(params_to_optimize, lr=lr, **optimizer_kwargs)
        
        iterator = trange(n_iterations, leave=True) if verbose else range(n_iterations)
        
        for i in iterator:
            optimizer.zero_grad()
            
            # 1. Create Atmosphere from current parameters 
            if self._use_param_model:
                atmosphere = self.param_model()
            else:
                atmosphere = Atmosphere(
                    self.params,
                    parameter_weights=self.problem.normalization,
                    grid_shape=self.grid_shape
                )
            
            # 2. Compute Loss
            loss, breakdown = self.problem.compute_loss(
                atmosphere,
                return_breakdown=True
            )
            
            # 3. Backward
            loss.backward()
            
            # 4. Gradient normalization trick (optional)
            if normalize_gradients:
                for p in params_to_optimize:
                    if p.grad is not None:
                        grad_mean = torch.mean(torch.abs(p.grad), dim=0, keepdim=True)
                        p.grad = p.grad / (grad_mean + 1e-9)
            
            # 5. Optimizer step
            optimizer.step()
            
            # 6. Parameter clamping (optional)
            if clamp_parameters and self._use_param_model:
                self.param_model.clamp_parameters()
            
            # 7. Logging
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'Chi2': f"{loss.item():.2e}",
                })
                
    def get_atmosphere(self) -> Atmosphere:
        """Returns the optimized Atmosphere."""
        with torch.no_grad():
            if self._use_param_model:
                return self.param_model()
            else:
                return Atmosphere(
                    self.params, 
                    parameter_weights=self.problem.normalization,
                    grid_shape=self.grid_shape
                )
    
    def get_cartesian_field(self):
        """
        Returns (Blos, BQ, BU) in physical units.
        
        Only works in ParameterModel mode.
        """
        if not self._use_param_model:
            raise RuntimeError("get_cartesian_field only works with ParameterModel mode")
        return self.param_model.get_cartesian_field()
