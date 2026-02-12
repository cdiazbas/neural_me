"""
LMAdam: Levenberg-Marquardt inspired Adam optimizer for PyTorch.

Combines Adam's adaptive learning rates and momentum with key convergence
strategies from the Levenberg-Marquardt algorithm:

  1. Lambda damping: modulates step size via (1 + lambda) factor on the
     denominator, interpolating between gradient descent (large lambda)
     and curvature-adapted steps (small lambda).
  2. Step acceptance/rejection: only accepts steps that reduce the loss,
     reverting parameters otherwise.
  3. Adaptive lambda: decreases lambda on successful steps (allowing larger
     steps) and increases it on rejected steps (enforcing smaller, safer steps).
  4. Best-model tracking: always maintains a checkpoint of the best
     parameters found during optimization.

The update rule:
    theta_{t+1} = theta_t - alpha / ((1 + lambda) * (sqrt(v_hat_t) + eps)) * m_hat_t

where lambda is adapted based on whether the step improved the loss,
analogous to LM's trust region adjustment.

Reference:
    The LM strategy is adapted from the C++ implementation in pyMilne
    (J. de la Cruz Rodriguez, ISP-SU), specifically the lambda-bracketing
    and step acceptance logic from lm.hpp.

Author: C. Diaz Baso (2026)
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, List
import copy
import math


class LMAdam(Optimizer):
    r"""
    LMAdam: Adam optimizer with Levenberg-Marquardt inspired trust region.

    Adds LM-style step acceptance/rejection and adaptive lambda damping to
    the standard Adam update. This can improve convergence for physics-based
    inverse problems where the loss landscape has narrow valleys.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Base learning rate (alpha). Default: 1e-3.
        betas (tuple): Coefficients for computing running averages of
            gradient and its square. Default: (0.9, 0.999).
        eps (float): Term added to denominator for numerical stability.
            Default: 1e-8.
        weight_decay (float): L2 penalty coefficient. Default: 0.
        init_lambda (float): Initial damping factor. Default: 1.0.
            Larger values make the optimizer more conservative (closer
            to gradient descent). Smaller values allow curvature-aware
            steps (closer to Gauss-Newton).
        lambda_factor (float): Multiplicative factor for lambda adaptation.
            On success: lambda /= lambda_factor.
            On failure: lambda *= lambda_factor^2.
            Default: sqrt(10) ~ 3.162 (matching pyMilne's LM).
        min_lambda (float): Minimum allowed lambda. Default: 1e-4.
        max_lambda (float): Maximum allowed lambda. Default: 1e3.
        max_rejects (int): Maximum consecutive rejected steps before
            forcing acceptance and resetting lambda. Default: 5.
        acceptance_mode (str): How to handle step acceptance.
            'strict'  – reject any step that increases loss.
            'relaxed' – accept if loss increase < tolerance fraction.
            'adam'     – always accept (disables LM rejection, keeps
                         only lambda-based lr scaling).
            Default: 'strict'.
        acceptance_tol (float): Tolerance for 'relaxed' mode.
            Step accepted if (new_loss - old_loss) / old_loss < tol.
            Default: 0.01.
        amsgrad (bool): Whether to use AMSGrad variant. Default: False.

    Example::

        optimizer = LMAdam(model.parameters(), lr=1e-3, init_lambda=1.0)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

            # Pass loss value for LM step acceptance
            optimizer.step(loss=loss.item())

    Note:
        If ``step()`` is called without ``loss``, the optimizer behaves
        like standard Adam with fixed lambda damping (no acceptance logic).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        init_lambda: float = 1.0,
        lambda_factor: float = math.sqrt(10.0),  # ~3.162, same as pyMilne
        min_lambda: float = 1e-4,
        max_lambda: float = 1e3,
        max_rejects: int = 5,
        acceptance_mode: str = 'strict',
        acceptance_tol: float = 0.01,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if acceptance_mode not in ('strict', 'relaxed', 'adam'):
            raise ValueError(f"Invalid acceptance_mode: {acceptance_mode}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

        # LM state (shared across all parameter groups)
        self._lm_lambda = self._clamp_lambda(init_lambda, min_lambda, max_lambda)
        self._lambda_factor = lambda_factor
        self._min_lambda = min_lambda
        self._max_lambda = max_lambda
        self._max_rejects = max_rejects
        self._acceptance_mode = acceptance_mode
        self._acceptance_tol = acceptance_tol

        # Tracking
        self._best_loss = float('inf')
        self._prev_loss = None
        self._n_rejects = 0
        self._n_accepts = 0
        self._total_steps = 0
        self._best_params = None  # Checkpoint of best parameters
        self._prev_params = None  # Parameters before last step (for rejection)

        # History for diagnostics
        self._lambda_history: List[float] = []
        self._loss_history: List[float] = []
        self._accepted_history: List[bool] = []

    # ------------------------------------------------------------------ #
    #  Properties for monitoring
    # ------------------------------------------------------------------ #

    @property
    def lm_lambda(self) -> float:
        """Current lambda damping value."""
        return self._lm_lambda

    @property
    def best_loss(self) -> float:
        """Best loss observed so far."""
        return self._best_loss

    @property
    def n_rejects(self) -> int:
        """Number of consecutive rejected steps."""
        return self._n_rejects

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic info for monitoring."""
        return {
            'lambda': self._lm_lambda,
            'best_loss': self._best_loss,
            'prev_loss': self._prev_loss,
            'n_rejects': self._n_rejects,
            'n_accepts': self._n_accepts,
            'total_steps': self._total_steps,
            'lambda_history': self._lambda_history,
            'loss_history': self._loss_history,
            'accepted_history': self._accepted_history,
        }

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clamp_lambda(val: float, mi: float, ma: float) -> float:
        """Clamp lambda to [min, max] range (from pyMilne's checkLambda)."""
        return max(mi, min(ma, val))

    def _save_params(self) -> List[torch.Tensor]:
        """Save a deep copy of all parameters."""
        saved = []
        for group in self.param_groups:
            for p in group['params']:
                saved.append(p.data.clone())
        return saved

    def _restore_params(self, saved: List[torch.Tensor]):
        """Restore parameters from a saved checkpoint."""
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(saved[idx])
                idx += 1

    def save_best(self):
        """Manually save current parameters as best checkpoint."""
        self._best_params = self._save_params()

    def restore_best(self):
        """Restore the best parameters found during optimization."""
        if self._best_params is not None:
            self._restore_params(self._best_params)

    # ------------------------------------------------------------------ #
    #  Core step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, loss: Optional[float] = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss.
                Not typically used with mini-batch training.
            loss: The current loss value (scalar). Used for LM-style step
                acceptance/rejection. If None, step acceptance is disabled
                and the optimizer behaves like damped Adam.

        Returns:
            The loss value if closure is provided, else None.
        """
        # Evaluate closure if provided
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
                if loss is None and loss_val is not None:
                    loss = loss_val.item() if torch.is_tensor(loss_val) else loss_val

        # ---- Step acceptance/rejection logic (LM-inspired) ---- #
        accepted = True

        if loss is not None and self._acceptance_mode != 'adam':
            self._loss_history.append(loss)

            if self._prev_loss is not None:
                accepted = self._evaluate_step(loss)

                if not accepted:
                    # Reject: revert parameters, increase lambda
                    if self._prev_params is not None:
                        self._restore_params(self._prev_params)
                    self._lm_lambda = self._clamp_lambda(
                        self._lm_lambda * self._lambda_factor ** 2,
                        self._min_lambda, self._max_lambda
                    )
                    self._n_rejects += 1
                    self._accepted_history.append(False)
                    self._lambda_history.append(self._lm_lambda)

                    # If too many rejects, force accept and reset
                    if self._n_rejects >= self._max_rejects:
                        self._n_rejects = 0
                        self._lm_lambda = self._clamp_lambda(
                            1.0, self._min_lambda, self._max_lambda
                        )
                    return loss_val

            # Save current parameters before taking the new step
            self._prev_params = self._save_params()
            self._prev_loss = loss

            if accepted and loss < self._best_loss:
                self._best_loss = loss
                self._best_params = self._save_params()

        # ---- Adam update with LM lambda damping ---- #
        lm_factor = 1.0 + self._lm_lambda

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            amsgrad = group.get('amsgrad', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LMAdam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Weight decay (decoupled, like AdamW)
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                step_size = lr / bias_correction1

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # LM damping: multiply denominator by (1 + lambda)
                # This is the key modification: large lambda → small steps (gradient descent),
                # small lambda → larger, curvature-adapted steps (Gauss-Newton-like)
                denom.mul_(lm_factor)

                # Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # ---- Post-step lambda adaptation (LM-inspired) ---- #
        if loss is not None and accepted and self._acceptance_mode != 'adam':
            # Successful step: decrease lambda to allow larger future steps
            if self._lm_lambda > 1.0001 * self._min_lambda:
                self._lm_lambda = self._clamp_lambda(
                    self._lm_lambda / self._lambda_factor,
                    self._min_lambda, self._max_lambda
                )
            else:
                self._lm_lambda = 10 * self._min_lambda

            self._n_rejects = 0
            self._n_accepts += 1
            self._accepted_history.append(True)

        self._lambda_history.append(self._lm_lambda)
        self._total_steps += 1

        return loss_val

    def _evaluate_step(self, current_loss: float) -> bool:
        """
        Evaluate whether a step should be accepted.

        Based on pyMilne's LM logic:
        - Accept if loss decreased
        - In 'relaxed' mode, also accept small increases

        Args:
            current_loss: Loss after the step.

        Returns:
            True if step is accepted, False if rejected.
        """
        if self._prev_loss is None:
            return True

        if self._acceptance_mode == 'strict':
            return current_loss < self._prev_loss

        elif self._acceptance_mode == 'relaxed':
            if current_loss < self._prev_loss:
                return True
            # Accept small increases within tolerance
            if self._prev_loss != 0:
                relative_increase = (current_loss - self._prev_loss) / abs(self._prev_loss)
                return relative_increase < self._acceptance_tol
            return False

        return True  # 'adam' mode: always accept


class LMAdamW(LMAdam):
    """
    LMAdam with decoupled weight decay (AdamW-style).

    Identical to LMAdam since weight decay is already decoupled in the
    base implementation. This alias is provided for API consistency
    with torch.optim.AdamW.
    """
    pass


# ======================================================================= #
#  Convenience: LM-aware training loop
# ======================================================================= #


def lmadam_train_step(
    model: torch.nn.Module,
    optimizer: LMAdam,
    loss_fn: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    **loss_kwargs,
) -> Dict[str, float]:
    """
    Single training step with LMAdam, handling loss-based acceptance.

    This wraps the forward → backward → step pattern, passing the loss
    to the optimizer for LM-style step management.

    Args:
        model: PyTorch model.
        optimizer: LMAdam optimizer instance.
        loss_fn: Loss function (callable).
        inputs: Input tensor.
        targets: Target tensor.
        **loss_kwargs: Additional keyword args for loss_fn.

    Returns:
        Dictionary with 'loss', 'lambda', 'accepted' keys.
    """
    optimizer.zero_grad()
    output = model(inputs)
    loss = loss_fn(output, targets, **loss_kwargs)
    loss.backward()

    loss_val = loss.item()
    optimizer.step(loss=loss_val)

    return {
        'loss': loss_val,
        'lambda': optimizer.lm_lambda,
        'accepted': optimizer._accepted_history[-1] if optimizer._accepted_history else True,
    }
