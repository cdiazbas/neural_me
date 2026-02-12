"""
Levenberg-Marquardt solver for Milne-Eddington inversion in PyTorch.

A faithful port of the C++ LM algorithm from pyMilne (J. de la Cruz
Rodriguez, ISP-SU), adapted for batched per-pixel inversion using
PyTorch's functional Jacobian computation.

Key features matching the C++ implementation:
  - Per-pixel Jacobian via torch.func.jacrev + vmap
  - Hessian approximation: J^T J
  - Diagonal damping: (1 + lambda) * diag(J^T J)
  - Step acceptance/rejection with lambda adaptation
  - Best-model tracking and convergence criteria
  - Parameter bounds checking

The LM update solves:
    (J^T J + (1 + lambda) * diag(J^T J)) * delta = J^T * r

per pixel, where J is (n_data, n_par), r is the residual vector,
and lambda is adapted based on step acceptance.

Author: C. Diaz Baso (2026)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Tuple, Dict
from tqdm import trange

from neural_me.core.problem import MEProblem
from neural_me.core.atmospheres.base import Atmosphere

# Try importing torch.func (PyTorch >= 2.0)
try:
    from torch.func import jacrev, vmap
except ImportError:
    from functorch import jacrev, vmap


class LevMarSolver:
    """
    Levenberg-Marquardt solver for pixel-wise ME inversion.

    Directly solves the nonlinear least-squares problem per pixel using
    the Gauss-Newton Hessian approximation J^T J, with LM damping and
    step acceptance logic matching pyMilne's C++ implementation.

    Args:
        problem: MEProblem instance (holds observation + forward model).
        atmosphere: Atmosphere to optimize (CartesianAtmosphere, etc.).
        device: Torch device.

    Usage::

        solver = LevMarSolver(problem, atmosphere)
        solver.solve(max_iter=20, init_lambda=3.16)
        atm = solver.get_atmosphere()
    """

    # Default parameter scales (matching pyMilne normalization).
    # These scale the Jacobian columns so that all parameters contribute
    # equally. The values match DEFAULT_NORMALIZATION in base.py.
    PARAM_SCALES = torch.tensor([
        500.0,   # B [G]
        1.0,     # gamma [rad]
        1.0,     # xi [rad]
        2.0,     # vlos [km/s]
        100.0,   # eta0
        0.2,     # a (damping)
        1.0,     # ddop [A]
        1.0,     # S0
        1.0,     # S1
    ])

    def __init__(
        self,
        problem: MEProblem,
        atmosphere: Optional[Atmosphere] = None,
        use_cartesian: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.problem = problem
        self.device = problem.device

        if atmosphere is not None:
            self.atmosphere = atmosphere
        elif use_cartesian:
            from neural_me.core.atmospheres import CartesianAtmosphere
            self.atmosphere = CartesianAtmosphere(
                grid_shape=problem.obs.grid_shape, device=self.device,
            )
        else:
            raise ValueError("Provide 'atmosphere' or set 'use_cartesian=True'.")

        self.n_pixels = problem.obs.n_pixels
        self.grid_shape = problem.obs.grid_shape

        # Observed data: (n_pixels, 4 * n_wav)
        mask = self.problem.obs.active_wav_idx
        self._obs_flat = torch.cat([
            self.problem.obs.stokes_I[:, mask],
            self.problem.obs.stokes_Q[:, mask],
            self.problem.obs.stokes_U[:, mask],
            self.problem.obs.stokes_V[:, mask],
        ], dim=-1).to(self.device)  # (n_pixels, 4*n_wav)

        self._n_wav = mask.sum().item()
        self._n_dat = 4 * self._n_wav

        # Noise weights (sigma). Default: uniform.
        # Can be set externally via set_weights()
        self._sigma = torch.ones(self._n_dat, device=self.device)

        # Wavelengths (broadcast-ready)
        self._wl = self.problem.obs.wavelengths[mask].to(self.device)  # (n_wav,)

        # Loss history
        self.chi2_history = []
        self.lambda_history = []

    def set_weights(self, sigma_I: float = 1.0, sigma_Q: float = 1.0,
                    sigma_U: float = 1.0, sigma_V: float = 1.0):
        """Set per-Stokes noise weights (inverse weights for Chi2)."""
        nw = self._n_wav
        self._sigma = torch.cat([
            torch.full((nw,), sigma_I, device=self.device),
            torch.full((nw,), sigma_Q, device=self.device),
            torch.full((nw,), sigma_U, device=self.device),
            torch.full((nw,), sigma_V, device=self.device),
        ])

    # ------------------------------------------------------------------ #
    #  Forward model: single pixel function for Jacobian computation
    # ------------------------------------------------------------------ #

    def _make_pixel_fn(self):
        """
        Returns a function f(params_9) -> synth_4nw suitable for jacrev.

        params_9: (9,) physical parameters [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
        returns:  (4*n_wav,) concatenated [I, Q, U, V] synthetic spectra
        """
        model = self.problem.model
        wl = self._wl.unsqueeze(0)  # (1, n_wav)

        def pixel_fn(params_9):
            # params_9: (9,) -> (1, 9)
            p = params_9.unsqueeze(0)
            I, Q, U, V = model(p, wl)
            # Each is (1, n_wav) -> squeeze and cat
            return torch.cat([I.squeeze(0), Q.squeeze(0),
                              U.squeeze(0), V.squeeze(0)])

        return pixel_fn

    # ------------------------------------------------------------------ #
    #  Core LM routines
    # ------------------------------------------------------------------ #

    def _get_params_tensor(self) -> torch.Tensor:
        """Get current atmosphere as (n_pixels, 9) physical parameter tensor."""
        return self.atmosphere.to_stack().reshape(-1, 9).detach().clone()

    def _set_params_tensor(self, params: torch.Tensor):
        """Set atmosphere from (n_pixels, 9) physical parameter tensor."""
        with torch.no_grad():
            # Unpack and set via Cartesian components
            B = params[:, 0]
            gamma = params[:, 1]
            xi = params[:, 2]
            vlos = params[:, 3]
            eta0 = params[:, 4]
            a = params[:, 5]
            ddop = params[:, 6]
            S0 = params[:, 7]
            S1 = params[:, 8]

            # Convert B, gamma, xi -> Blos, BQ, BU
            Blos = B * torch.cos(gamma)
            Bt = B * torch.sin(gamma)
            BQ = Bt * torch.cos(2.0 * xi)
            BU = Bt * torch.sin(2.0 * xi)

            self.atmosphere.set_parameters({
                'Blos': Blos, 'BQ': BQ, 'BU': BU,
                'vlos': vlos, 'eta0': eta0, 'a': a,
                'ddop': ddop, 'S0': S0, 'S1': S1,
            })

    def _compute_residuals(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted residuals for all pixels.

        Args:
            params: (n_pixels, 9) physical parameters.

        Returns:
            residuals: (n_pixels, 4*n_wav) weighted residuals.
        """
        wl = self._wl.unsqueeze(0).expand(params.shape[0], -1)
        I, Q, U, V = self.problem.model(params, wl)
        syn = torch.cat([I, Q, U, V], dim=-1)  # (n_pixels, 4*n_wav)
        return (self._obs_flat - syn) / self._sigma.unsqueeze(0)

    def _compute_chi2(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel Chi2.

        Args:
            residuals: (n_pixels, n_dat)

        Returns:
            chi2: (n_pixels,)
        """
        return torch.sum(residuals ** 2, dim=-1)

    def _compute_jacobian_batched(self, params: torch.Tensor,
                                  batch_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-pixel Jacobian and synthetic spectra using vmap+jacrev.

        Args:
            params: (n_pixels, 9)
            batch_size: Process pixels in batches to control memory.

        Returns:
            J: (n_pixels, n_dat, 9) Jacobian
            syn: (n_pixels, n_dat) synthetic spectra
        """
        pixel_fn = self._make_pixel_fn()
        jac_fn = jacrev(pixel_fn)
        vmap_jac_fn = vmap(jac_fn)

        n_pix = params.shape[0]
        J_list = []
        syn_list = []

        for start in range(0, n_pix, batch_size):
            end = min(start + batch_size, n_pix)
            batch_params = params[start:end]

            # Jacobian: (batch, n_dat, 9)
            J_batch = vmap_jac_fn(batch_params)

            # Synthetic: (batch, n_dat)
            with torch.no_grad():
                syn_batch = vmap(pixel_fn)(batch_params)

            J_list.append(J_batch)
            syn_list.append(syn_batch)

        J = torch.cat(J_list, dim=0)
        syn = torch.cat(syn_list, dim=0)
        return J, syn

    def _solve_lm_step(
        self,
        J: torch.Tensor,
        r: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Solve the LM system for parameter correction.

        Solves: (J^T J + (1+lambda)*diag(J^T J)) * delta = J^T * r
        per pixel (batched).

        Args:
            J: (n_pixels, n_dat, n_par) Jacobian
            r: (n_pixels, n_dat) residuals
            lam: current lambda

        Returns:
            delta: (n_pixels, n_par) correction
        """
        # Scale Jacobian by sigma weights
        sigma_inv = (1.0 / self._sigma).unsqueeze(0).unsqueeze(-1)  # (1, n_dat, 1)
        J_scaled = J * sigma_inv

        # J^T J: (n_pixels, n_par, n_par)
        JtJ = torch.bmm(J_scaled.transpose(1, 2), J_scaled)

        # J^T r: (n_pixels, n_par)
        Jtr = torch.bmm(J_scaled.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)

        # Damping: (1 + lambda) * diag(J^T J)
        diag = torch.diagonal(JtJ, dim1=-2, dim2=-1)  # (n_pixels, n_par)
        damping = (1.0 + lam) * diag
        JtJ_damped = JtJ.clone()
        JtJ_damped.diagonal(dim1=-2, dim2=-1).copy_(damping)

        # Add small regularization for numerical stability
        JtJ_damped.diagonal(dim1=-2, dim2=-1).add_(1e-12)

        # Solve batched linear system
        try:
            delta = torch.linalg.solve(JtJ_damped, Jtr)
        except torch.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            delta = torch.bmm(torch.linalg.pinv(JtJ_damped), Jtr.unsqueeze(-1)).squeeze(-1)

        return delta

    def _check_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Apply physical parameter bounds (matching pyMilne's Par::Check).

        Args:
            params: (n_pixels, 9) [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]

        Returns:
            params: clamped parameters
        """
        with torch.no_grad():
            params[:, 0].clamp_(min=0.0)                    # B >= 0
            params[:, 1].clamp_(min=0.0, max=torch.pi)      # gamma in [0, pi]
            params[:, 2].clamp_(min=-torch.pi, max=torch.pi) # xi in [-pi, pi]
            params[:, 3].clamp_(min=-5.0, max=5.0)          # vlos in [-5, 5] km/s
            params[:, 4].clamp_(min=0.01, max=600.0)        # eta0
            params[:, 5].clamp_(min=0.001, max=1.0)         # a (damping)
            params[:, 6].clamp_(min=0.001, max=0.5)         # ddop
            params[:, 7].clamp_(min=0.01)                    # S0 > 0
            params[:, 8].clamp_(min=0.01)                    # S1 > 0
        return params

    # ------------------------------------------------------------------ #
    #  Main inversion loop (matching pyMilne's fitData)
    # ------------------------------------------------------------------ #

    def solve(
        self,
        max_iter: int = 20,
        init_lambda: float = 3.1623,       # sqrt(10), pyMilne default
        chi2_threshold: float = 1.0,        # Stop if Chi2 < threshold
        fx_threshold: float = 2e-3,         # Stop if relative improvement < threshold
        max_rejects: int = 6,               # Max consecutive rejects before stopping
        delay_bracket: int = 2,             # Not used (kept for API parity)
        batch_size: int = 4096,             # Pixels per Jacobian batch
        verbose: bool = True,
    ) -> Dict:
        """
        Run the Levenberg-Marquardt inversion.

        Follows pyMilne's fitData() logic faithfully:
        1. Compute initial Chi2 and Jacobian
        2. Iterate:
           a. Solve LM system for correction
           b. Apply correction, compute new Chi2
           c. If improved: save best, decrease lambda
           d. If worse: revert, increase lambda, retry
           e. Check convergence criteria
        3. Restore best model

        Args:
            max_iter: Maximum LM iterations.
            init_lambda: Initial damping parameter.
            chi2_threshold: Stop if mean Chi2 drops below this.
            fx_threshold: Stop if relative Chi2 improvement < this for 2 iterations.
            max_rejects: Max consecutive rejected steps before stopping.
            batch_size: Batch size for Jacobian computation.
            verbose: Print iteration info.

        Returns:
            Dictionary with convergence diagnostics.
        """
        # Constants (matching pyMilne)
        fac_lam = 3.1622776601683795  # sqrt(10)
        max_lam = 100.0 * fac_lam
        min_lam = 1e-4

        lam = max(min_lam, min(max_lam, init_lambda))

        # Get initial parameters: (n_pixels, 9)
        model_params = self._get_params_tensor()
        model_params = self._check_params(model_params)

        best_params = model_params.clone()
        best_chi2_per_pixel = torch.full((self.n_pixels,), 1e32, device=self.device)
        best_chi2 = 1e32

        if verbose:
            print(f"LevMarSolver: {self.n_pixels} pixels, {self._n_dat} data points/pixel, 9 parameters")
            print(f"  init_lambda={lam:.4f}, max_iter={max_iter}, batch_size={batch_size}")

        # Initial forward pass + Jacobian
        t0 = time.time()
        J, syn = self._compute_jacobian_batched(model_params, batch_size)
        r = (self._obs_flat - syn) / self._sigma.unsqueeze(0)
        chi2_pp = self._compute_chi2(r)
        best_chi2 = chi2_pp.mean().item()
        best_chi2_pp = chi2_pp.clone()
        best_syn = syn.clone()

        if verbose:
            dt = time.time() - t0
            print(f"  [Init] mean Chi2={best_chi2:.5f}, dt={dt:.2f}s")

        self.chi2_history.append(best_chi2)
        self.lambda_history.append(lam)

        # Iteration
        n_rejected = 0
        too_small = False
        quit_flag = False

        iterator = range(max_iter)
        if verbose:
            iterator = trange(max_iter, desc="LM", leave=True)

        for it in iterator:
            # Get correction from current Jacobian
            trial_params = model_params.clone()
            delta = self._solve_lm_step(J, r, lam)
            trial_params = trial_params + delta
            trial_params = self._check_params(trial_params)

            # Compute new Chi2
            trial_r = self._compute_residuals(trial_params)
            trial_chi2_pp = self._compute_chi2(trial_r)
            trial_chi2 = trial_chi2_pp.mean().item()

            # Did Chi2 improve?
            if trial_chi2 < best_chi2:
                dfx = (best_chi2 - trial_chi2) / best_chi2
                old_lam = lam

                best_chi2 = trial_chi2
                best_params = trial_params.clone()
                model_params = trial_params

                # Decrease lambda
                if lam > 1.0001 * min_lam:
                    lam = max(min_lam, min(max_lam, lam / fac_lam))
                else:
                    lam = 10 * min_lam

                # Check convergence
                if dfx < fx_threshold:
                    if too_small:
                        quit_flag = True
                    else:
                        too_small = True

                n_rejected = 0

                # Recompute Jacobian at new best point
                t0 = time.time()
                J, syn = self._compute_jacobian_batched(model_params, batch_size)
                r = (self._obs_flat - syn) / self._sigma.unsqueeze(0)
                dt_jac = time.time() - t0

                if verbose:
                    iterator.set_postfix({
                        'Chi2': f'{best_chi2:.5f}',
                        'λ': f'{lam:.2e}',
                        'Δ': f'{dfx:.1e}',
                        'dt': f'{dt_jac:.1f}s',
                    })

            else:
                # Increase lambda and retry
                old_lam = lam
                lam = max(min_lam, min(max_lam, lam * fac_lam ** 2))
                n_rejected += 1

                if verbose:
                    iterator.set_postfix({
                        'Chi2': f'{trial_chi2:.5f}>{best_chi2:.5f}',
                        'λ': f'{old_lam:.2e}→{lam:.2e}',
                        'rej': n_rejected,
                    })

                if n_rejected < max_rejects:
                    # Don't update Jacobian, just retry with higher lambda
                    continue

            self.chi2_history.append(best_chi2)
            self.lambda_history.append(lam)

            # Convergence checks
            if n_rejected >= max_rejects:
                if verbose:
                    print(f"\n  Stopped: max rejects ({max_rejects}) reached")
                break

            if best_chi2 < chi2_threshold:
                if verbose:
                    print(f"\n  Stopped: Chi2 ({best_chi2:.5f}) < threshold ({chi2_threshold})")
                break

            if quit_flag:
                if verbose:
                    print(f"\n  Stopped: Chi2 improvement too small for 2 iterations")
                break

        # Restore best parameters
        self._set_params_tensor(best_params)
        self.atmosphere.clamp_parameters()

        total_time = time.time() - t0

        if verbose:
            print(f"  Final: mean Chi2={best_chi2:.5f}")

        return {
            'chi2': best_chi2,
            'chi2_history': self.chi2_history,
            'lambda_history': self.lambda_history,
            'n_iterations': it + 1 if 'it' in dir() else 0,
        }

    def get_atmosphere(self) -> Atmosphere:
        """Return the optimized atmosphere."""
        return self.atmosphere

    def initialize_parameters(self, method: str = 'default'):
        """Initialize parameters (placeholder for more sophisticated init)."""
        pass  # CartesianAtmosphere already uses sensible defaults
