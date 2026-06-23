"""
Stereoscopic Solver Implementation.

Dedicated solver for optimizing StereoscopicAtmosphere using two
MEProblems (Instruments). Includes support for Stray Light correction on
the second instrument.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Union

from ..core.atmospheres.stereoscopic import StereoscopicAtmosphere
from ..core import MEProblem
from .degradation import DegradationOperator


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
        degradation_operators: Optional[Dict[int, DegradationOperator]] = None,
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

        # Collect parameters to optimize (atmosphere parameters + any trainable operator parameters)
        params = list(self.atmosphere.parameters())
        for op in self.degradation_operators.values():
            params.extend(list(op.parameters()))

        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Print High-Level Optimization Setup Summary
        lines = []
        lines.append("┌────────────────────────────────────────────────────────────┐")
        lines.append("│         STEREOSCOPIC SOLVER INITIALIZATION SUMMARY         │")
        lines.append("├────────────────────────────────────────────────────────────┤")
        atm_type = type(self.atmosphere).__name__
        lines.append(f"│  Atmosphere Model   : {atm_type:<36} │")
        lines.append(f"│  Grid Shape         : {str(self.atmosphere.grid_shape):<36} │")
        lines.append(f"│  Loss Weight (alpha): {self.alpha:<36.3f} │")
        lines.append(f"│    - Telescope 1    : {1.0 - self.alpha:<36.3f} │")
        lines.append(f"│    - Telescope 2    : {self.alpha:<36.3f} │")
        lines.append(f"│  Learning Rate      : {lr:<36} │")
        
        for inst_idx, prob, name in [(1, self.problem1, "Telescope 1 (Reference View)"), 
                                     (2, self.problem2, "Telescope 2 (Secondary View)")]:
            lines.append("├────────────────────────────────────────────────────────────┤")
            lines.append(f"│  {name:<56} │")
            lines.append(f"│    - Line ID        : {prob.nlinea:<34} │")
            lines.append(f"│    - Wavelengths    : {len(prob.obs.wavelengths):<34d} │")
            data_shape_str = str(list(prob.obs.data.shape))
            lines.append(f"│    - Data Shape     : {data_shape_str:<34} │")
            op = self.degradation_operators.get(inst_idx)
            if op is not None:
                op_name = type(op).__name__
                # Check if operator has local epsilon
                if getattr(op, "epsilon", None) is not None:
                    eps_val = op.epsilon.item()
                    is_learnable = isinstance(op.epsilon, nn.Parameter) and op.epsilon.requires_grad
                    learn_str = "learnable" if is_learnable else "fixed"
                    lines.append(f"│    - Degradation    : {op_name:<34} │")
                    lines.append(f"│                       ({learn_str} ε = {eps_val:.4f})              │")
                else:
                    # Check if atmosphere has epsilon
                    eps_val = getattr(self.atmosphere, "epsilon", None)
                    if eps_val is not None:
                        eps_val = eps_val.item()
                        is_learnable = self.atmosphere.epsilon.requires_grad
                        learn_str = "learnable" if is_learnable else "fixed"
                        lines.append(f"│    - Degradation    : {op_name:<34} │")
                        lines.append(f"│                       ({learn_str} atmos ε = {eps_val:.4f})        │")
                    else:
                        lines.append(f"│    - Degradation    : {op_name:<34} │")
            else:
                lines.append(f"│    - Degradation    : {'None (Ideal observations)':<34} │")
        lines.append("└────────────────────────────────────────────────────────────┘")
        print("\n" + "\n".join(lines) + "\n")

    def solve(
        self,
        n_iterations: int = 1000,
        verbose: bool = True,
        normalize_gradients: bool = False,
        return_breakdown: bool = False,
    ) -> Union[List[float], Tuple[List[float], List[float], List[float]]]:
        """
        Runs the optimization loop.
        """
        self.atmosphere.train()
        losses = []
        losses1 = []
        losses2 = []

        iterator = range(n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Stereo Inversion")

        for i in iterator:
            self.optimizer.zero_grad()

            # 1. Get Views
            atm1 = self.atmosphere.get_atmosphere(1)
            atm2 = self.atmosphere.get_atmosphere(2)

            # 2. Forward Pass (Compute Spectra)
            syn1 = self.problem1.compute_forward_model(atm1)
            syn2 = self.problem2.compute_forward_model(atm2)

            # 3. Apply Degradation (Stray Light / Calibration)
            if 1 in self.degradation_operators:
                syn1 = self.degradation_operators[1](syn1, self.atmosphere)
            if 2 in self.degradation_operators:
                syn2 = self.degradation_operators[2](syn2, self.atmosphere)

            # 4. Compute Loss
            loss1 = self.problem1.compute_loss_from_syn(syn1)
            loss2 = self.problem2.compute_loss_from_syn(syn2)

            # Combined Loss
            loss = (1.0 - self.alpha) * loss1 + self.alpha * loss2

            loss.backward()

            # Optional Gradient Normalization
            if normalize_gradients:
                for param in self.optimizer.param_groups[0]["params"]:
                    if param.grad is not None:
                        param.grad /= torch.mean(torch.abs(param.grad), dim=0) + 1e-8

            # 5. Optimization Step
            self.optimizer.step()

            # 6. Clamp Parameters (CRITICAL for Legacy Reproduction)
            if hasattr(self.atmosphere, "clamp_parameters"):
                self.atmosphere.clamp_parameters()
            for op in self.degradation_operators.values():
                if hasattr(op, "clamp_parameters"):
                    op.clamp_parameters()

            losses.append(loss.item())
            losses1.append(loss1.item())
            losses2.append(loss2.item())

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "L": f"{loss.item():.2e}",
                        "L1": f"{loss1.item():.2e}",
                        "L2": f"{loss2.item():.2e}",
                    }
                )

        if return_breakdown:
            return losses, losses1, losses2
        return losses
