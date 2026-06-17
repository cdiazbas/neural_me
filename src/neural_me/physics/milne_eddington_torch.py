"""
Pure PyTorch implementation of Milne-Eddington synthesis.

This is the original, reference implementation.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .zeeman import fvoigt, paramLine


class MilneEddingtonTorch(nn.Module):
    """
    Pure PyTorch Milne-Eddington implementation.
    """

    def __init__(self, nlinea=None, compile=False, checkpoint_gradients=False):
        """
        Initialize MilneEddingtonTorch.

        Args:
            nlinea: Line configuration index
            compile: If True, use torch.compile for potential speedup on large batches.
                     Set to False for small cases to avoid compilation overhead.
            checkpoint_gradients: If True, use gradient checkpointing to reduce memory usage
                                  at the cost of extra computation during backward pass.
        """
        super().__init__()
        if nlinea is not None:
            self.param = paramLine(nlinea)
        else:
            self.param = None

        # Optionally compile the synthesis function
        self._compile = compile
        self.checkpoint_gradients = checkpoint_gradients
        if compile:
            self._stokesSyn_compiled = torch.compile(self._stokesSyn_core)
        else:
            self._stokesSyn_compiled = None

    def stokesSyn(self, param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1):
        """
        Dispatch to compiled or non-compiled implementation.
        """
        if self.checkpoint_gradients:
            # Use checkpointing
            # Note: param is a list, checkpoint handles args.
            # We pass use_reentrant=False for better stability with modern PyTorch
            if self._stokesSyn_compiled is not None:
                return checkpoint(
                    self._stokesSyn_compiled,
                    param,
                    x,
                    B,
                    gamma,
                    xi,
                    vlos,
                    eta0,
                    a,
                    ddop,
                    S_0,
                    S_1,
                    use_reentrant=False,
                )
            else:
                return checkpoint(
                    self._stokesSyn_core,
                    param,
                    x,
                    B,
                    gamma,
                    xi,
                    vlos,
                    eta0,
                    a,
                    ddop,
                    S_0,
                    S_1,
                    use_reentrant=False,
                )
        else:
            if self._stokesSyn_compiled is not None:
                return self._stokesSyn_compiled(
                    param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1
                )
            else:
                return self._stokesSyn_core(
                    param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1
                )

    def _stokesSyn_core(self, param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1):
        """
        Core Milne-Eddington Stokes synthesis (vectorized over batch and wavelength).

        Computes the full polarized emergent spectrum (I, Q, U, V) for a set of
        atmospheric parameters using the analytical ME solution. Follows the
        same vectorization scheme as the C++ backend for numerical consistency.

        Args:
            param: Zeeman pattern from paramLine(), a nested list::

                    [[dpi, dsr, dsb], [spi, ssr, ssb], meta]

                  where d* are splitting tensors (1, n_comp) and s* are
                  normalized strengths (1, n_comp). meta[7] is the central
                  wavelength λ₀ in Å.
            x:      (batch, n_lambda) wavelength offsets from line center [Å].
            B:      (batch, 1) total magnetic field strength [G].
            gamma:  (batch, 1) field inclination [rad], 0=LOS.
            xi:     (batch, 1) field azimuth [rad].
            vlos:   (batch, 1) line-of-sight velocity [km/s].
            eta0:   (batch, 1) line-to-continuum absorption ratio.
            a:      (batch, 1) Voigt damping parameter.
            ddop:   (batch, 1) Doppler width [Å]. Clamped to ≥1e-6 internally
                    to prevent division-by-zero during autograd.
            S_0:    (batch, 1) source function constant.
            S_1:    (batch, 1) source function gradient.

        Returns:
            Tuple (I, Q, U, V), each of shape (batch, n_lambda).

        Numerical stability notes:
            - ddop is clamped to ≥1e-6 before the Voigt profile computation
              to prevent Inf when the optimizer drives ddop toward zero.
            - The polarization transfer matrix determinant (Det) is clamped away
              from zero before inversion to prevent NaN in both forward and backward.
        """
        # 1/sqrt(π) — normalization factor for Voigt profiles
        # Equivalent to 1.0 / math.sqrt(math.pi)
        sqrtpi = 1.0 / 1.7724538509055160

        # Magnetic field direction:
        singamma = torch.sin(gamma)
        cosgamma = torch.cos(gamma)
        sin2xi = torch.sin(2 * xi)
        cos2xi = torch.cos(2 * xi)

        # Calculating the Zeeman Pattern
        # Param structure: [[dpi, dsr, dsb], [spi, ssr, ssb], meta]
        dpi_0, dsr_0, dsb_0 = param[0]
        spi_0, ssr_0, ssb_0 = param[1]
        l0 = param[2][7]

        # Ensure parameters are on the same device as input B
        device = B.device
        dpi_0 = dpi_0.to(device)
        dsr_0 = dsr_0.to(device)
        dsb_0 = dsb_0.to(device)
        spi_0 = spi_0.to(device)
        ssr_0 = ssr_0.to(device)
        ssb_0 = ssb_0.to(device)

        # Lorentz splitting [Å] and velocity shift [Å]
        lB = 4.67e-13 * l0**2.0 * B  # (batch, 1)
        cc = 3.0e5  # speed of light [km/s]
        vlosA = l0 * vlos / cc  # Doppler shift in Å (batch, 1)

        # Apply Lorentz factor to component splittings
        dpi = dpi_0 * lB
        dsr = dsr_0 * lB
        dsb = dsb_0 * lB

        # ============================================
        # Single fvoigt Call Optimization
        # ============================================

        # Concatenate all component shifts: (batch, n_pi + n_r + n_b)
        shifts = torch.cat([dpi, dsr, dsb], dim=1)

        # Reshape for broadcasting over wavelength and components simultaneously:
        #   x:      (batch, n_lambda)    → (batch, n_lambda, 1)
        #   shifts: (batch, n_comp)      → (batch, 1,        n_comp)
        #   vlosA:  (batch, 1)           → (batch, 1,        1)
        #   ddop:   (batch, 1)           → (batch, 1,        1)
        #   a:      (batch, 1)           → (batch, 1,        1)
        x_in      = x.unsqueeze(-1)
        shifts_in = shifts.unsqueeze(1)
        vlosA_in  = vlosA.unsqueeze(-1)
        ddop_in   = ddop.unsqueeze(-1)
        a_in      = a.unsqueeze(-1)

        # B02: Defensive clamp on ddop to prevent Inf if optimizer drives it toward 0.
        # The Atmosphere classes enforce ddop > 0 via clamp/softplus, but this guard
        # acts as a final safety net inside the forward model itself.
        ddop_safe = torch.clamp(ddop_in, min=1e-6)

        # Normalized frequency offsets: (batch, n_lambda, n_comp)
        xx = (x_in - shifts_in - vlosA_in) / ddop_safe
        H_all, F_all = fvoigt(a_in, xx)

        # Split Voigt profiles back by component type
        n_p = dpi.shape[1]
        n_r = dsr.shape[1]
        n_b = dsb.shape[1]

        H_p, H_r, H_b = torch.split(H_all, [n_p, n_r, n_b], dim=-1)
        F_p, F_r, F_b = torch.split(F_all, [n_p, n_r, n_b], dim=-1)

        def reduce_comp(H, F, s):
            """Weighted sum over Zeeman components → (batch, n_lambda) η and ρ."""
            # s: (1, n_comp) → (1, 1, n_comp) for broadcasting
            s_in = s.unsqueeze(1)
            eta = torch.sum(H * s_in, dim=-1) * sqrtpi
            rho = torch.sum(2.0 * F * s_in, dim=-1) * sqrtpi
            return eta, rho

        eta_p, rho_p = reduce_comp(H_p, F_p, spi_0)
        eta_r, rho_r = reduce_comp(H_r, F_r, ssr_0)
        eta_b, rho_b = reduce_comp(H_b, F_b, ssb_0)

        # ============================================
        # Polarized Transfer Matrix (analytical ME solution)
        # ============================================

        # Total absorption coefficient (Stokes I)
        eta_I = 1.0 + 0.5 * eta0 * (
            eta_p * (singamma**2.0) + 0.5 * (eta_b + eta_r) * (1.0 + cosgamma**2.0)
        )

        # Dichroism and anomalous dispersion common factors
        diff_eta = 0.5 * eta0 * (eta_p - 0.5 * (eta_b + eta_r)) * (singamma**2.0)
        diff_rho = 0.5 * eta0 * (rho_p - 0.5 * (rho_b + rho_r)) * (singamma**2.0)

        # Q, U, V components of the absorption and dispersion vectors
        eta_V_term = 0.5 * eta0 * (eta_r - eta_b) * cosgamma
        rho_V_term = 0.5 * eta0 * (rho_r - rho_b) * cosgamma

        # Stack into 3-vectors [Q, U, V]: (batch, n_lambda, 3)
        eta_vec = torch.stack(
            [diff_eta * cos2xi, diff_eta * sin2xi, eta_V_term], dim=-1
        )
        rho_vec = torch.stack(
            [diff_rho * cos2xi, diff_rho * sin2xi, rho_V_term], dim=-1
        )

        # Scalar products for the analytical inversion
        eta_sq_sum  = torch.sum(eta_vec**2, dim=-1)       # |η|²
        rho_sq_sum  = torch.sum(rho_vec**2, dim=-1)       # |ρ|²
        eta_dot_rho = torch.sum(eta_vec * rho_vec, dim=-1) # η·ρ

        # B08: Validate 3-vector shapes before cross product
        assert eta_vec.shape[-1] == 3, (
            f"eta_vec last dim must be 3 for cross product, got {eta_vec.shape[-1]}"
        )

        # Cross product: η × ρ → (batch, n_lambda, 3)
        cross_prod = torch.linalg.cross(eta_vec, rho_vec, dim=-1)

        # Transfer matrix determinant
        # Det = η_I² (η_I² - |η|² + |ρ|²) - (η·ρ)²
        Det = eta_I**2 * (eta_I**2 - eta_sq_sum + rho_sq_sum) - eta_dot_rho**2

        # B01: Guard Det against zero before inversion.
        # Pathological atmospheres (B≈0, eta0≈0) can drive Det→0, producing NaN
        # in the forward output AND infinite gradients during backward.
        # We preserve the sign while clamping |Det| to a minimum safe value.
        Det_safe = torch.sign(Det) * torch.clamp(torch.abs(Det), min=1e-30)
        IDet = 1.0 / Det_safe

        # Analytical ME solution
        term_S1 = IDet * S_1

        # Stokes I: I = S₀ + S₁ · η_I · (η_I² + |ρ|²) / Det
        I = S_0 + term_S1 * eta_I * (eta_I**2 + rho_sq_sum)

        # Stokes Q, U, V vector:
        # [Q, U, V] = -S₁/Det · (η_I² η⃗ - η_I (η⃗ × ρ⃗) + (η⃗·ρ⃗) ρ⃗)
        eta_I_uns       = eta_I.unsqueeze(-1)
        eta_dot_rho_uns = eta_dot_rho.unsqueeze(-1)

        QUV_vec = -term_S1.unsqueeze(-1) * (
            (eta_I_uns**2) * eta_vec
            - eta_I_uns * cross_prod
            + eta_dot_rho_uns * rho_vec
        )

        Q = QUV_vec[..., 0]
        U = QUV_vec[..., 1]
        V = QUV_vec[..., 2]

        return I, Q, U, V

    def forward(self, params, wl, nlinea=None):
        # params shape: (batch_size, 9)
        # wl shape: (batch_size, n_lambda)

        if nlinea is not None:
            # Update line parameters if provided (for flexibility)
            self.param = paramLine(nlinea)

        # Ensure wl has correct shape if it's 1D (assume same WL for all batch elements)
        if wl.dim() == 1:
            wl = wl.unsqueeze(0).repeat(params.shape[0], 1)

        # Parameter order: [B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1]
        B = params[:, 0][:, None]
        gamma = params[:, 1][:, None]
        xi = params[:, 2][:, None]
        vlos = params[:, 3][:, None]
        eta0 = params[:, 4][:, None]
        a = params[:, 5][:, None]
        ddop = params[:, 6][:, None]
        S_0 = params[:, 7][:, None]
        S_1 = params[:, 8][:, None]

        return self.stokesSyn(
            self.param, wl, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1
        )
