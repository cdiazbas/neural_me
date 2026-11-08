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
        balance: str = "fixed",
    ):
        """
        Args:
            atmosphere: The unified model.
            problem1: MEProblem for Instrument 1 (Reference/Hinode).
            problem2: MEProblem for Instrument 2 (Secondary/SolO).
            lr: Learning rate.
            alpha: Weighting factor for loss: (1-alpha)*L1 + alpha*L2.
            degradation_operators: Dict mapping view index (1 or 2) to a DegradationOperator.
            balance: "fixed" (default) minimises (1-alpha)*L1 + alpha*L2.
                "dynamic" divides each term by its current value (detached),
                i.e. follows the gradient of (1-alpha)*log L1 + alpha*log L2:
                both views pull with the same relative strength whatever
                their absolute chi2 (the reported loss is still the fixed sum).
        """
        if balance not in ("fixed", "dynamic"):
            raise ValueError(f"balance must be 'fixed' or 'dynamic', got {balance!r}")
        self.balance = balance
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
        W = 62  # total width including the two │ characters
        inner = W - 2  # usable characters between │ and │

        def row(text=""):
            """Pad text to fill exactly inner columns."""
            return f"│{text:<{inner}}│"

        lines = []
        lines.append("┌" + "─" * inner + "┐")
        lines.append(row("         STEREOSCOPIC SOLVER INITIALIZATION SUMMARY         "))
        lines.append("├" + "─" * inner + "┤")
        atm_type = type(self.atmosphere).__name__
        lines.append(row(f"  Atmosphere Model   : {atm_type}"))
        lines.append(row(f"  Grid Shape         : {self.atmosphere.grid_shape}"))
        lines.append(row(f"  Loss Weight (alpha): {self.alpha:.3f}"))
        lines.append(row(f"    - Telescope 1    : {1.0 - self.alpha:.3f}"))
        lines.append(row(f"    - Telescope 2    : {self.alpha:.3f}"))
        lines.append(row(f"  Learning Rate      : {lr}"))
        lines.append(row(f"  Loss balance       : {balance} (alpha={alpha})"))

        for inst_idx, prob, name in [(1, self.problem1, "Telescope 1 (Reference View)"),
                                     (2, self.problem2, "Telescope 2 (Secondary View)")]:
            lines.append("├" + "─" * inner + "┤")
            lines.append(row(f"  {name}"))
            lines.append(row(f"    - Line ID        : {prob.nlinea}"))
            lines.append(row(f"    - Wavelengths    : {len(prob.obs.wavelengths)}"))
            lines.append(row(f"    - Data Shape     : {list(prob.obs.data.shape)}"))
            op = self.degradation_operators.get(inst_idx)
            if op is not None:
                op_name = type(op).__name__
                lines.append(row(f"    - Degradation    : {op_name}"))
            else:
                lines.append(row("    - Degradation    : None (Ideal observations)"))

        # Parameter trainability summary
        lines.append("├" + "─" * inner + "┤")
        all_params = list(self.atmosphere.named_parameters())
        learnable = [n for n, p in all_params if p.requires_grad]
        frozen    = [n for n, p in all_params if not p.requires_grad]
        clean = lambda n: n.replace("_raw", "")
        lines.append(row(f"  Atmosphere Parameters"))
        lines.append(row(f"    - Total          : {len(all_params)}  (learnable: {len(learnable)}, frozen: {len(frozen)})"))
        if frozen:
            frozen_str = ", ".join(clean(n) for n in frozen)
            # wrap into multiple rows if too long
            prefix = "    - Frozen         : "
            available = inner - 2 - len(prefix)
            words, current_line = [], ""
            for part in frozen_str.split(", "):
                if current_line and len(current_line) + 2 + len(part) > available:
                    lines.append(row(f"  {prefix}{current_line}"))
                    prefix = " " * len(prefix)
                    current_line = part
                else:
                    current_line = f"{current_line}, {part}" if current_line else part
            if current_line:
                lines.append(row(f"  {prefix}{current_line}"))
        lines.append("└" + "─" * inner + "┘")
        print("\n" + "\n".join(lines) + "\n")

    def solve(
        self,
        n_iterations: int = 1000,
        verbose: bool = True,
        normalize_gradients: bool = False,
        return_breakdown: bool = False,
        warmup: int = 0,
    ) -> Union[List[float], Tuple[List[float], List[float], List[float]]]:
        """
        Runs the optimization loop.

        ``warmup``: the learning rate rises linearly from lr / warmup to lr over
        the first ``warmup`` iterations. Use it after a warm start: Adam's first
        step moves every parameter by ~lr whatever its gradient, which on a
        2.3 M-parameter grid field undoes much of the start (B_w correlation with
        the warm-start maps 0.999 -> 0.856 after one step at lr 1e-2).
        """
        base_lr = [g["lr"] for g in self.optimizer.param_groups]
        self.atmosphere.train()
        losses = []
        losses1 = []
        losses2 = []

        iterator = range(n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Stereo Inversion")

        for i in iterator:
            if warmup > 0:
                f = min(1.0, (i + 1) / warmup)
                for g, lr0 in zip(self.optimizer.param_groups, base_lr):
                    g["lr"] = lr0 * f
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

            if self.balance == "dynamic":
                objective = ((1.0 - self.alpha) * loss1 / (loss1.detach() + 1e-30)
                             + self.alpha * loss2 / (loss2.detach() + 1e-30))
                objective.backward()
            else:
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
