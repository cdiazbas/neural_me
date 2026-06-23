"""
Stereoscopic Solver Implementation.

Dedicated solver for optimizing StereoscopicAtmosphere using two
MEProblems (Instruments). Includes support for Stray Light correction on
the second instrument.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict

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
            loss1 = self.problem1.compute_loss_from_syn(syn1)
            loss2 = self.problem2.compute_loss_from_syn(syn2)

            # Combined Loss
            loss = (1.0 - self.alpha) * loss1 + self.alpha * loss2

            loss.backward()

            # 5. Optimization Step
            self.optimizer.step()

            # 6. Clamp Parameters (CRITICAL for Legacy Reproduction)
            if hasattr(self.atmosphere, "clamp_parameters"):
                self.atmosphere.clamp_parameters()

            losses.append(loss.item())

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "L": f"{loss.item():.2e}",
                        "L1": f"{loss1.item():.2e}",
                        "L2": f"{loss2.item():.2e}",
                    }
                )

        return losses
