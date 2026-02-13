"""
Levenberg-Marquardt solver for Milne-Eddington inversion in PyTorch.

A faithful port of the C++ LM algorithm from pyMilne (J. de la Cruz
Rodriguez, ISP-SU), using the same PyTorch forward model interface as
the Adam-based PixelSolver.

The Jacobian (response functions) is obtained from the C++ backend's
forward pass — the same RF that milne_cpp.py uses for autograd backward().
No torch.func.jacrev or vmap is used when using the C++ backend.
For the torch backend, autograd (jacrev+vmap) is used as fallback.

Key features matching the C++ implementation:
  - Hessian approximation: J^T J
  - Diagonal damping: (1 + lambda) * diag(J^T J)
  - Step acceptance/rejection with lambda adaptation
  - Best-model tracking and convergence criteria
  - Parameter bounds via CartesianAtmosphere.clamp_parameters()

The LM update solves per pixel:
    (J^T J + (1 + lambda) * diag(J^T J)) * delta = J^T * r

Author: C. Diaz Baso (2026)
"""

import torch
import numpy as np
import time
from typing import Optional, Dict
from tqdm import trange

from neural_me.core.problem import MEProblem
from neural_me.core.atmospheres.base import Atmosphere


class LevMarSolver:
    """
    Levenberg-Marquardt solver for pixel-wise ME inversion.

    Uses the same MEProblem / Atmosphere interface as PixelSolver (Adam).
    The Jacobian comes from the C++ backend's response functions, computed
    during the forward pass, or from autograd for the torch backend.
    
    **Note**: The LM algorithm optimizes in physical spherical coordinates 
    [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]. For CartesianAtmosphere, 
    the solver automatically converts to/from Cartesian [Blos, BQ, BU]. For 
    spherical-based atmospheres (SphericalAtmosphere, SphericalSoftplusAtmosphere),
    parameters are used directly.

    Args:
        problem: MEProblem instance (holds observation + forward model).
        atmosphere: Atmosphere to optimize (any Atmosphere subclass).

    Usage::

        from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
        atm = SphericalSoftplusAtmosphere(grid_shape=obs.grid_shape, device='cpu')
        solver = LevMarSolver(problem, atmosphere=atm)
        solver.solve(max_iter=20, init_lambda=3.16)
        atm = solver.get_atmosphere()
    """

    def __init__(
        self,
        problem: MEProblem,
        atmosphere: Optional[Atmosphere] = None,
        device: Optional[torch.device] = None,
    ):
        self.problem = problem
        self.device = device if device is not None else problem.device

        if atmosphere is None:
            raise ValueError(
                "LevMarSolver requires an 'atmosphere' object.\n"
                "Example (CartesianAtmosphere):\n"
                "  from neural_me.core.atmospheres import CartesianAtmosphere\n"
                "  atm = CartesianAtmosphere(grid_shape=obs.grid_shape, device='cpu')\n"
                "  solver = LevMarSolver(problem, atmosphere=atm)\n"
                "\n"
                "Example (SphericalSoftplusAtmosphere):\n"
                "  from neural_me.core.atmospheres import SphericalSoftplusAtmosphere\n"
                "  atm = SphericalSoftplusAtmosphere(grid_shape=obs.grid_shape, device='cpu')\n"
                "  solver = LevMarSolver(problem, atmosphere=atm)"
            )
        self.atmosphere = atmosphere

        self.n_pixels = problem.obs.n_pixels
        self.grid_shape = problem.obs.grid_shape

        # Observed data: (n_pixels, 4 * n_wav)
        # NOTE: Use ALL wavelengths for forward model (like MEProblem does),
        # then apply mask to both observed and synthetic data
        mask = self.problem.obs.active_wav_idx
        self._obs_flat = torch.cat(
            [
                self.problem.obs.stokes_I[:, mask],
                self.problem.obs.stokes_Q[:, mask],
                self.problem.obs.stokes_U[:, mask],
                self.problem.obs.stokes_V[:, mask],
            ],
            dim=-1,
        ).to(self.device)  # (n_pixels, 4*n_wav_active)

        self._n_wav_active = mask.numel()  # Number of active wavelengths
        self._n_wav_total = self.problem.obs.wavelengths.shape[0]  # Total wavelengths
        self._n_dat = 4 * self._n_wav_active
        self._mask = mask  # Store mask for applying to synthetic spectra

        # Noise weights (sigma). Default: uniform.
        self._sigma = torch.ones(self._n_dat, device=self.device)

        # Wavelengths: use ALL wavelengths for forward model (matching MEProblem)
        self._wl = self.problem.obs.wavelengths.to(self.device)

        # Detect backend
        self._is_cpp = self.problem.backend == "cpp"

        # Loss history
        self.chi2_history = []  # Chi2 (L2) for internal LM optimization
        self.l1_history = []    # L1 loss for comparison with Adam
        self.lambda_history = []

    def set_weights(
        self,
        sigma_I: float = 1.0,
        sigma_Q: float = 1.0,
        sigma_U: float = 1.0,
        sigma_V: float = 1.0,
    ):
        """
        Set per-Stokes noise weights (inverse weights for Chi2).
        """
        nw = self._n_wav
        self._sigma = torch.cat(
            [
                torch.full((nw,), sigma_I, device=self.device),
                torch.full((nw,), sigma_Q, device=self.device),
                torch.full((nw,), sigma_U, device=self.device),
                torch.full((nw,), sigma_V, device=self.device),
            ]
        )

    # ------------------------------------------------------------------ #
    #  Forward model + Jacobian
    # ------------------------------------------------------------------ #

    def _forward_and_jacobian(self, params: torch.Tensor):
        """
        Compute synthetic spectra and Jacobian for all pixels.

        For the C++ backend: runs the forward pass (which internally computes
        the RF via synthesize_rf), then retrieves the RF as the Jacobian.
        This is the same RF that backward() uses for autograd gradients.

        For the torch backend: uses jacrev+vmap as fallback.

        Args:
            params: (n_pixels, 9) physical parameters
                    [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]

        Returns:
            syn: (n_pixels, 4*n_wav_active) synthetic spectra (masked)
            J:   (n_pixels, 9, 4*n_wav_active) Jacobian (RF) (masked), reordered to match
                 our parameter order [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
        """
        wl = self._wl.unsqueeze(0).expand(params.shape[0], -1)

        if self._is_cpp:
            # Forward pass through PyTorch -> C++ backend
            # Compute at ALL wavelengths (like MEProblem does)
            with torch.no_grad():
                I, Q, U, V = self.problem.model(params, wl)
            
            # Apply mask to get only active wavelengths
            I_masked = I[:, self._mask]
            Q_masked = Q[:, self._mask]
            U_masked = U[:, self._mask]
            V_masked = V[:, self._mask]
            syn = torch.cat([I_masked, Q_masked, U_masked, V_masked], dim=-1)

            # Get the RF from the C++ backend's last forward pass
            # Shape: (n_pixels, 9, 4*n_wav_total) in CPP parameter order
            rf = self.problem.model.get_last_response_functions()

            # Reorder from CPP -> our parameter order:
            #   CPP:    [B, gamma, xi, vlos, ddop, eta0, a,   S0, S1]
            #   Ours:   [B, gamma, xi, vlos, eta0, a,   ddop, S0, S1]
            J_full = rf[:, [0, 1, 2, 3, 5, 6, 4, 7, 8], :]

            # Apply mask to Jacobian: extract active wavelengths for each Stokes
            # J_full is (n_pixels, 9, 4*n_wav_total)
            # We need to extract [I_mask, Q_mask, U_mask, V_mask] for each parameter
            nw_total = self._n_wav_total
            J_I = J_full[:, :, :nw_total][:, :, self._mask]
            J_Q = J_full[:, :, nw_total:2*nw_total][:, :, self._mask]
            J_U = J_full[:, :, 2*nw_total:3*nw_total][:, :, self._mask]
            J_V = J_full[:, :, 3*nw_total:4*nw_total][:, :, self._mask]
            J = torch.cat([J_I, J_Q, J_U, J_V], dim=-1)  # (n_pixels, 9, 4*n_wav_active)

            # Apply eta0 scaling: C++ RF is w.r.t. eta0_cpp = eta0_pytorch/sqrt(pi)
            # Chain rule: d/d(eta0_pytorch) = d/d(eta0_cpp) * (1/sqrt(pi))
            J[:, 4, :] /= np.sqrt(np.pi)

            return syn.to(self.device), J.to(syn.dtype).to(self.device)

        else:
            # Torch backend fallback: autograd Jacobian
            from torch.func import jacrev, vmap

            model = self.problem.model
            wl_single = self._wl.unsqueeze(0)  # (1, n_wav_total)

            def pixel_fn(params_9):
                p = params_9.unsqueeze(0)
                I, Q, U, V = model(p, wl_single)
                # Apply mask
                I_m, Q_m, U_m, V_m = I[:, self._mask], Q[:, self._mask], U[:, self._mask], V[:, self._mask]
                return torch.cat([I_m.squeeze(0), Q_m.squeeze(0), U_m.squeeze(0), V_m.squeeze(0)])

            jac_fn = vmap(jacrev(pixel_fn))
            J = jac_fn(params)  # (n_pixels, 4*n_wav_active, 9)
            J = J.transpose(1, 2)  # (n_pixels, 9, 4*n_wav_active)

            with torch.no_grad():
                I, Q, U, V = model(params, wl)
            I_m, Q_m, U_m, V_m = I[:, self._mask], Q[:, self._mask], U[:, self._mask], V[:, self._mask]
            syn = torch.cat([I_m, Q_m, U_m, V_m], dim=-1)

            return syn, J

    def _forward_only(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass only (no Jacobian). Used for trial step evaluation.

        Args:
            params: (n_pixels, 9)

        Returns:
            syn: (n_pixels, 4*n_wav_active) masked synthetic spectra
        """
        wl = self._wl.unsqueeze(0).expand(params.shape[0], -1)
        with torch.no_grad():
            I, Q, U, V = self.problem.model(params, wl)
        # Apply mask to get only active wavelengths
        I_m, Q_m, U_m, V_m = I[:, self._mask], Q[:, self._mask], U[:, self._mask], V[:, self._mask]
        return torch.cat([I_m, Q_m, U_m, V_m], dim=-1)

    # ------------------------------------------------------------------ #
    #  Core LM routines
    # ------------------------------------------------------------------ #

    def _compute_residuals(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        """Weighted residuals: (obs - syn) / sigma."""
        return (obs - syn) / self._sigma.unsqueeze(0)

    def _compute_chi2(self, residuals: torch.Tensor) -> torch.Tensor:
        """Per-pixel Chi2: (n_pixels,)"""
        return torch.sum(residuals**2, dim=-1)
    
    def _compute_l1_loss(self, obs: torch.Tensor, syn: torch.Tensor) -> float:
        """Compute L1 loss matching MEProblem.compute_loss() with loss_type='l1'"""
        nw = self._n_wav_active
        # Split into Stokes components
        obs_I, obs_Q, obs_U, obs_V = obs[:, :nw], obs[:, nw:2*nw], obs[:, 2*nw:3*nw], obs[:, 3*nw:]
        syn_I, syn_Q, syn_U, syn_V = syn[:, :nw], syn[:, nw:2*nw], syn[:, 2*nw:3*nw], syn[:, 3*nw:]
        
        # Compute L1 error per Stokes (mean absolute error)
        err_I = torch.mean(torch.abs(syn_I - obs_I))
        err_Q = torch.mean(torch.abs(syn_Q - obs_Q))
        err_U = torch.mean(torch.abs(syn_U - obs_U))
        err_V = torch.mean(torch.abs(syn_V - obs_V))
        
        # Weighted sum (assuming equal weights [1,1,1,1])
        weights = self.problem.weights if hasattr(self.problem, 'weights') else [1.0, 1.0, 1.0, 1.0]
        l1_loss = weights[0] * err_I + weights[1] * err_Q + weights[2] * err_U + weights[3] * err_V
        return l1_loss.item()

    def _solve_lm_step(
        self,
        J: torch.Tensor,
        r: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Solve the LM normal equations for parameter correction.

        Matching C++ getCorrection() exactly:
            A = J^T J                     (Hessian approximation)
            B = J^T r                     (gradient)
            A[i,i] *= (1 + lambda)        (diagonal damping)
            Solve A * delta = B

        Args:
            J: (n_pixels, 9, n_dat) sigma-weighted Jacobian
            r: (n_pixels, n_dat) sigma-weighted residuals
            lam: current lambda

        Returns:
            delta: (n_pixels, 9) parameter correction
        """
        n_par = J.shape[1]

        # A = J J^T: (n_pixels, 9, 9)
        A = torch.bmm(J, J.transpose(1, 2))

        # B = J r: (n_pixels, 9)
        B = torch.bmm(J, r.unsqueeze(-1)).squeeze(-1)

        # Diagonal damping: A[i,i] *= (1 + lambda)
        diag_idx = torch.arange(n_par, device=J.device)
        A[:, diag_idx, diag_idx] *= 1.0 + lam

        # Solve A * delta = B
        try:
            delta = torch.linalg.solve(A, B)
        except torch.linalg.LinAlgError:
            delta = torch.bmm(torch.linalg.pinv(A), B.unsqueeze(-1)).squeeze(-1)

        return delta

    def _params_from_atmosphere(self) -> torch.Tensor:
        """
        Get (n_pixels, 9) physical params from atmosphere.
        """
        with torch.no_grad():
            return self.atmosphere.to_stack().reshape(-1, 9).clone()

    def _params_to_atmosphere(self, params: torch.Tensor):
        """
        Set atmosphere from (n_pixels, 9) physical params [B,gamma,xi,...].
        
        Detects atmosphere type and sets parameters appropriately:
        - CartesianAtmosphere: converts spherical [B,gamma,xi] -> Cartesian [Blos,BQ,BU]
        - Spherical-based atmospheres: uses [B,gamma,xi] directly
        """
        with torch.no_grad():
            B = params[:, 0]
            gamma = params[:, 1]
            xi = params[:, 2]
            vlos = params[:, 3]
            eta0 = params[:, 4]
            a = params[:, 5]
            ddop = params[:, 6]
            S0 = params[:, 7]
            S1 = params[:, 8]

            # Clamp physical parameters
            B = torch.clamp(B, min=0.0, max=4000.0)
            gamma = torch.clamp(gamma, min=0.0, max=np.pi)

            # Azimuth wrapping
            xi = torch.remainder(xi, np.pi)

            eta0 = torch.clamp(eta0, min=50.0, max=150.0)
            a = torch.clamp(a, min=0.2, max=0.5)
            ddop = torch.clamp(ddop, min=0.02, max=0.08)
            vlos = torch.clamp(vlos, min=-3.0, max=3.0)

            from neural_me.core.atmospheres import CartesianAtmosphere

            # Reshape to grid_shape if needed (for SphericalSoftplusAtmosphere, etc.)
            def reshape_if_needed(tensor):
                if tensor.shape[0] == self.n_pixels and len(self.grid_shape) > 1:
                    return tensor.reshape(self.grid_shape)
                return tensor

            if isinstance(self.atmosphere, CartesianAtmosphere):
                # CartesianAtmosphere expects Cartesian components (flattened)
                Blos = B * torch.cos(gamma)
                Bt = B * torch.sin(gamma)
                BQ = Bt * torch.cos(2.0 * xi)
                BU = Bt * torch.sin(2.0 * xi)

                self.atmosphere.set_parameters(
                    {
                        "Blos": Blos,
                        "BQ": BQ,
                        "BU": BU,
                        "vlos": vlos,
                        "eta0": eta0,
                        "a": a,
                        "ddop": ddop,
                        "S0": S0,
                        "S1": S1,
                    }
                )
            else:
                # Spherical-based atmospheres (SphericalAtmosphere, SphericalSoftplusAtmosphere)
                # Use spherical parameters directly, reshaped to grid_shape
                self.atmosphere.set_parameters(
                    {
                        "B": reshape_if_needed(B),
                        "gamma": reshape_if_needed(gamma),
                        "xi": reshape_if_needed(xi),
                        "vlos": reshape_if_needed(vlos),
                        "eta0": reshape_if_needed(eta0),
                        "a": reshape_if_needed(a),
                        "ddop": reshape_if_needed(ddop),
                        "S0": reshape_if_needed(S0),
                        "S1": reshape_if_needed(S1),
                    }
                )

    # ------------------------------------------------------------------ #
    #  Main inversion loop (matching pyMilne's fitData faithfully)
    # ------------------------------------------------------------------ #

    def solve(
        self,
        max_iter: int = 20,
        init_lambda: float = 3.1623,  # sqrt(10), pyMilne default
        chi2_threshold: float = 1.0,  # Stop if mean Chi2 < threshold
        fx_threshold: float = 2e-3,  # Stop if relative improvement < threshold
        max_rejects: int = 6,  # Max consecutive rejects before stopping
        recompute_jacobian_after: int = 4,  # Recompute J after N rejects
        verbose: bool = True,
    ) -> Dict:
        """
        Run the Levenberg-Marquardt inversion.

        Follows pyMilne C++ fitData() logic faithfully:
        1. Clamp parameters, compute initial Chi2 and Jacobian (RF)
        2. Iterate:
           a. Copy best model, solve LM for correction delta
           b. Apply correction, clamp, compute new Chi2 (forward only)
           c. If improved: save best, decrease lambda, recompute Jacobian
           d. If worse: increase lambda, retry with same Jacobian
           e. Check convergence
        3. Restore best model

        Args:
            max_iter: Maximum LM iterations.
            init_lambda: Initial damping parameter.
            chi2_threshold: Stop if mean Chi2 drops below this.
            fx_threshold: Stop if relative Chi2 improvement < this for 2 iters.
            max_rejects: Max consecutive rejected steps before stopping.
            recompute_jacobian_after: Recompute Jacobian after N consecutive rejections.
            verbose: Print iteration info.

        Returns:
            Dictionary with convergence diagnostics.
        """
        # --- Constants matching pyMilne --- #
        fac_lam = 3.1622776601683795  # sqrt(10)
        max_lam = 100.0 * fac_lam
        min_lam = 1e-4

        def check_lambda(val):
            return max(min_lam, min(max_lam, val))

        lam = check_lambda(init_lambda)

        # --- Initial parameters --- #
        self.atmosphere.clamp_parameters()
        best_params = self._params_from_atmosphere()  # (n_pixels, 9)

        if verbose:
            print(
                f"LevMarSolver: {self.n_pixels} pixels, "
                f"{self._n_dat} data points/pixel, 9 parameters"
            )
            print(f"  init_lambda={lam:.4f}, max_iter={max_iter}")

        # --- Initial forward + Jacobian --- #
        t0_total = time.time()

        syn, J = self._forward_and_jacobian(best_params)

        # Weight J by sigma for the normal equations
        sigma_inv = 1.0 / self._sigma.unsqueeze(0)  # (1, n_dat)
        J_w = J * sigma_inv.unsqueeze(1)  # (n_pix, 9, n_dat) weighted

        r = self._compute_residuals(self._obs_flat, syn)
        chi2_pp = self._compute_chi2(r)
        best_chi2 = chi2_pp.mean().item()
        
        # Also compute L1 loss for comparison with Adam
        l1_loss = self._compute_l1_loss(self._obs_flat, syn)

        if verbose:
            dt = time.time() - t0_total
            print(f"  [Init] mean Chi2={best_chi2:.5f}, L1={l1_loss:.5f}, dt={dt:.2f}s")

        self.chi2_history.append(best_chi2)
        self.l1_history.append(l1_loss)
        self.lambda_history.append(lam)

        # --- Iteration --- #
        n_rejected = 0
        n_jacobian_recomputes = 0  # Track Jacobian recomputations due to rejections
        too_small = False
        quit_flag = False

        iterator = range(max_iter)
        if verbose:
            iterator = trange(max_iter, desc="LM", leave=True)

        for it in iterator:
            # Start from best model each time (as in C++)
            trial_params = best_params.clone()

            # Solve LM system
            delta = self._solve_lm_step(J_w, r, lam)

            # Apply correction
            trial_params = trial_params + delta

            # Clamp via atmosphere
            self._params_to_atmosphere(trial_params)
            self.atmosphere.clamp_parameters()
            trial_params = self._params_from_atmosphere()

            # Forward only to evaluate Chi2
            trial_syn = self._forward_only(trial_params)
            trial_r = self._compute_residuals(self._obs_flat, trial_syn)
            trial_chi2 = self._compute_chi2(trial_r).mean().item()

            # --- Did Chi2 improve? --- #
            if trial_chi2 < best_chi2:
                dfx = (best_chi2 - trial_chi2) / best_chi2

                best_chi2 = trial_chi2
                best_params = trial_params.clone()

                # Decrease lambda (matching C++)
                if lam > 1.0001 * min_lam:
                    lam = check_lambda(lam / fac_lam)
                else:
                    lam = 10 * min_lam

                # Check convergence
                if dfx < fx_threshold:
                    if too_small:
                        quit_flag = True
                    else:
                        too_small = True

                n_rejected = 0
                n_jacobian_recomputes = 0  # Reset counter on successful step

                # Recompute Jacobian at new best (as in C++)
                t_jac = time.time()
                self._params_to_atmosphere(best_params)
                syn, J = self._forward_and_jacobian(best_params)
                J_w = J * sigma_inv.unsqueeze(1)
                r = self._compute_residuals(self._obs_flat, syn)
                dt_jac = time.time() - t_jac

                if verbose:
                    iterator.set_postfix(
                        {
                            "Chi2": f"{best_chi2:.5f}",
                            "lam": f"{lam:.2e}",
                            "d": f"{dfx:.1e}",
                            "dt": f"{dt_jac:.1f}s",
                        }
                    )

            else:
                # Increase lambda and retry (matching C++)
                old_lam = lam
                lam = check_lambda(lam * fac_lam**2)
                n_rejected += 1

                if verbose:
                    iterator.set_postfix(
                        {
                            "Chi2": f"{trial_chi2:.5f}>{best_chi2:.5f}",
                            "lam": f"{old_lam:.2e}->{lam:.2e}",
                            "rej": n_rejected,
                        }
                    )

                # Recompute Jacobian after several rejections to get fresh direction
                if n_rejected > 0 and n_rejected % recompute_jacobian_after == 0:
                    n_jacobian_recomputes += 1
                    if verbose:
                        iterator.write(f"  [Recomputing Jacobian after {n_rejected} rejections (#{n_jacobian_recomputes})]")
                    t_jac = time.time()
                    self._params_to_atmosphere(best_params)
                    syn, J = self._forward_and_jacobian(best_params)
                    J_w = J * sigma_inv.unsqueeze(1)
                    r = self._compute_residuals(self._obs_flat, syn)
                    dt_jac = time.time() - t_jac
                    # Reset lambda after Jacobian recompute to be less conservative
                    lam = check_lambda(init_lambda)
                    if verbose:
                        iterator.write(f"  [Reset lambda to {lam:.2e}, dt={dt_jac:.2f}s]")
                    
                    # Stop if we've recomputed Jacobian too many times without progress
                    if n_jacobian_recomputes >= 3:
                        if verbose:
                            iterator.write(f"  [Stopping: {n_jacobian_recomputes} Jacobian recomputations without progress - likely at local minimum]")
                        quit_flag = True

                if quit_flag:
                    break  # Exit immediately if flagged for stopping

                if n_rejected < max_rejects:
                    continue  # Retry with same J, higher lambda

            self.chi2_history.append(best_chi2)
            l1_loss = self._compute_l1_loss(self._obs_flat, syn)
            self.l1_history.append(l1_loss)
            self.lambda_history.append(lam)

            # --- Convergence checks (matching C++ order) --- #
            if n_rejected >= max_rejects:
                if verbose:
                    print(f"\n  Stopped: max rejects ({max_rejects}) reached")
                break

            if best_chi2 < chi2_threshold:
                if verbose:
                    print(f"\n  Stopped: Chi2 ({best_chi2:.5f}) < threshold")
                break

            if quit_flag:
                if verbose:
                    print("\n  Stopped: improvement too small for 2 iterations")
                break

        # --- Restore best model --- #
        self._params_to_atmosphere(best_params)
        self.atmosphere.clamp_parameters()

        total_time = time.time() - t0_total
        if verbose:
            print(f"  Final: mean Chi2={best_chi2:.5f}, total={total_time:.1f}s")

        return {
            "chi2": best_chi2,
            "chi2_history": self.chi2_history,
            "l1_history": self.l1_history,
            "lambda_history": self.lambda_history,
            "n_iterations": it + 1 if "it" in dir() else 0,
        }

    def get_atmosphere(self) -> Atmosphere:
        """
        Return the optimized atmosphere.
        """
        return self.atmosphere
