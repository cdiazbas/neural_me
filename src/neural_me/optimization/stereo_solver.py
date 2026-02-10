"""
Stereoscopic Solver Implementation.

Dedicated solver for optimizing StereoscopicAtmosphere using two MEProblems (Instruments).
Includes support for Stray Light correction on the second instrument.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List, Tuple, Callable, Dict

from ..core.atmospheres.stereoscopic import StereoscopicAtmosphere
from ..core import MEProblem
from abc import ABC, abstractmethod

class DegradationOperator(nn.Module, ABC):
    """
    Abstract base class for instrumental degradation (Stray Light, PSF, etc.)
    """
    @abstractmethod
    def forward(self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere) -> Tuple[torch.Tensor, ...]:
        pass

class LegacyActiveOperator(DegradationOperator):
    """
    Implements the logic found in the active lines of ME_combinado_mecpp_caliV.py.
    Legacy: out = (cali)*stokes + (1-cali)*stokes
    Note: If cali is a scalar, this is identity. If cali is a tensor, it broadcasts.
    """
    def forward(self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere) -> Tuple[torch.Tensor, ...]:
        I, Q, U, V = stokes
        cali = atmosphere.cali
        
        # Logic from ME_combinado line 162:
        # outstokes2 = (cali)*stokes2 + (1-self.cali)*stokes2 
        # We apply this to each component.
        
        I_new = cali * I + (1.0 - cali) * I
        Q_new = cali * Q + (1.0 - cali) * Q
        U_new = cali * U + (1.0 - cali) * U
        V_new = cali * V + (1.0 - cali) * V
        
        return I_new, Q_new, U_new, V_new

class StrayLightOperator(DegradationOperator):
    """
    Implements a standard stray light correction:
    out = (1 - cali) * stokes + cali * I_mean
    
    Where I_mean is the spatial average of I (assumed scattered light).
    """
    def forward(self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere) -> Tuple[torch.Tensor, ...]:
        I, Q, U, V = stokes
        cali = atmosphere.cali
        
        # Compute spatial mean (batch/pixels dim is 0)
        # Shape: (N, n_lambda) -> Mean -> (1, n_lambda)
        I_mean = I.mean(dim=0, keepdim=True)
        Q_mean = Q.mean(dim=0, keepdim=True)
        U_mean = U.mean(dim=0, keepdim=True)
        V_mean = V.mean(dim=0, keepdim=True)
        
        I_new = (1.0 - cali) * I + cali * I_mean
        Q_new = (1.0 - cali) * Q + cali * Q_mean
        U_new = (1.0 - cali) * U + cali * U_mean
        V_new = (1.0 - cali) * V + cali * V_mean
        
        return I_new, Q_new, U_new, V_new


class StereoSolver:
    """
    Optimizes a StereoscopicAtmosphere to fit two observations simultaneously.
    """
    
    def __init__(
        self,
        atmosphere: StereoscopicAtmosphere,
        problem1: MEProblem,
        problem2: MEProblem,
        lr: float = 1e-2,
        alpha: float = 0.5,
        degradation_operators: Optional[Dict[int, DegradationOperator]] = None
    ):
        """
        Args:
            atmosphere: The unified model.
            problem1: MEProblem for Instrument 1 (Reference/Hinode).
            problem2: MEProblem for Instrument 2 (Secondary/SolO).
            lr: Learning rate.
            alpha: Weighting factor for loss: (1-alpha)*L1 + alpha*L2.
            degradation_operators: Dict mapping view index (0 or 1) to a DegradationOperator.
        """
        self.atmosphere = atmosphere
        self.problem1 = problem1
        self.problem2 = problem2
        self.alpha = alpha
        self.degradation_operators = degradation_operators or {}
        
        # Register operators as submodules if they have parameters (though mostly they use atm params)
        # If passed in list/dict, we can't easily register unless we are an nn.Module ourselves.
        # StereoSolver is NOT an nn.Module. 
        # However, the optimizer needs to know strictly about parameters.
        # Our operators use `atmosphere.cali` so that is covered by `atmosphere.parameters()`.
        # If an operator had its OWN parameters, we'd need to add them.
        # For now, we assume operators rely on Atmosphere params or are fixed.
        
        params = list(self.atmosphere.parameters())
        # If operators have parameters, add them:
        for op in self.degradation_operators.values():
            params.extend(list(op.parameters()))
            
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
    def solve(self, n_iterations: int = 1000, verbose: bool = True) -> List[float]:
        """
        Runs the optimization loop.
        """
        self.atmosphere.train()
        losses = []
        
        iterator = range(n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Stereo Inversion")
            
        for i in iterator:
            self.optimizer.zero_grad()
            
            # 1. Get Views
            atm1 = self.atmosphere.get_atmosphere(0)
            atm2 = self.atmosphere.get_atmosphere(1)
            
            # 2. Forward Pass (Compute Spectra)
            syn1 = self.problem1.compute_forward_model(atm1)
            syn2 = self.problem2.compute_forward_model(atm2)
            
            # 3. Apply Degradation (Stray Light / Calibration)
            if 0 in self.degradation_operators:
                syn1 = self.degradation_operators[0](syn1, self.atmosphere)
            if 1 in self.degradation_operators:
                syn2 = self.degradation_operators[1](syn2, self.atmosphere)
                
            # 4. Compute Loss
            def calc_loss(syn, obs_obj):
                I, Q, U, V = syn
                # Simple MSE
                # mask = obs_obj.active_wav_idx # Not used currently in simple MSE
                loss = 0.0
                loss += torch.mean((I - obs_obj.stokes_I)**2)
                loss += torch.mean((Q - obs_obj.stokes_Q)**2)
                loss += torch.mean((U - obs_obj.stokes_U)**2)
                loss += torch.mean((V - obs_obj.stokes_V)**2)
                return loss

            loss1 = calc_loss(syn1, self.problem1.obs)
            loss2 = calc_loss(syn2, self.problem2.obs)
            
            # Combined Loss
            loss = (1.0 - self.alpha) * loss1 + self.alpha * loss2
            
            loss.backward()
            
            # 5. Optimization Step
            self.optimizer.step()
            
            # 6. Clamp Parameters (CRITICAL for Legacy Reproduction)
            if hasattr(self.atmosphere, 'clamp_parameters'):
                self.atmosphere.clamp_parameters()
            
            losses.append(loss.item())
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'L': f"{loss.item():.2e}",
                    'L1': f"{loss1.item():.2e}",
                    'L2': f"{loss2.item():.2e}"
                })
                
        return losses
