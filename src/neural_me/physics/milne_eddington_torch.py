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
        # Pi square factor:
        sqrtpi = 1.0 / 1.7724538509055160  # sqrt(pi)

        # Magnetic field direction:
        singamma = torch.sin(gamma)
        cosgamma = torch.cos(gamma)
        sin2xi = torch.sin(2 * xi)
        cos2xi = torch.cos(2 * xi)

        # Calculating the Zeeman Pattern
        # Param structure: [[dpi, dsr, dsb], [spi, ssr, ssb],meta]
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

        # Lorentz factor & vlosA
        lB = 4.67e-13 * l0**2.0 * B  # shape (batch, 1)
        cc = 3.0e5
        vlosA = l0 * vlos / cc  # shape (batch, 1)

        # Apply Lorentz factor to splittings
        dpi = dpi_0 * lB
        dsr = dsr_0 * lB
        dsb = dsb_0 * lB

        # ============================================
        # Single fvoigt Call Optimization
        # ============================================

        # Concatenate shifts and strengths
        # dpi, dsr, dsb: (batch, n_comp_p), (batch, n_comp_r), (batch, n_comp_b)
        shifts = torch.cat([dpi, dsr, dsb], dim=1)  # (batch, total_comp)

        # spi_0, ssr_0, ssb_0: (1, n_comp) -> broadcast to (batch, n_comp)
        B.shape[0]
        # strengths = torch.cat([spi_0, ssr_0, ssb_0], dim=1) # (1, total_comp)

        # Prepare inputs for broadcasting
        # x: (batch, n_lambda) -> (batch, n_lambda, 1)
        x_in = x.unsqueeze(-1)
        # shifts: (batch, total_comp) -> (batch, 1, total_comp)
        shifts_in = shifts.unsqueeze(1)
        # vlosA: (batch, 1) -> (batch, 1, 1)
        vlosA_in = vlosA.unsqueeze(-1)
        # ddop: (batch, 1) -> (batch, 1, 1)
        ddop_in = ddop.unsqueeze(-1)
        # a: (batch, 1) -> (batch, 1, 1)
        a_in = a.unsqueeze(-1)

        # Compute profiles for ALL components at once
        # xx: (batch, n_lambda, total_comp)
        xx = (x_in - shifts_in - vlosA_in) / ddop_in
        H_all, F_all = fvoigt(a_in, xx)

        # Split results back
        # We need the number of components for each part
        n_p = dpi.shape[1]
        n_r = dsr.shape[1]
        n_b = dsb.shape[1]

        # Split across last dimension (components)
        H_p, H_r, H_b = torch.split(H_all, [n_p, n_r, n_b], dim=-1)
        F_p, F_r, F_b = torch.split(F_all, [n_p, n_r, n_b], dim=-1)

        # Compute eta/rho components
        # strengths are needed for each part. strengths tensor was (1, total_comp).
        # We can split it too or just use the original _,_0 variables.
        # strengths needed shape: (1, 1, n_comp) for broadcasting

        def reduce_comp(H, F, s):
            # s: (1, n_comp) -> (1, 1, n_comp)
            s_in = s.unsqueeze(1)
            term1 = torch.sum(H * s_in, dim=-1)
            term2 = torch.sum(2.0 * F * s_in, dim=-1)
            # Note: C++ backend does NOT normalize by ddop, so we match that behavior
            eta = term1 * sqrtpi  # (batch, n_lambda)
            rho = term2 * sqrtpi
            return eta, rho

        eta_p, rho_p = reduce_comp(H_p, F_p, spi_0)
        eta_r, rho_r = reduce_comp(H_r, F_r, ssr_0)
        eta_b, rho_b = reduce_comp(H_b, F_b, ssb_0)

        # ============================================
        # Vector Algebra for Stokes Q, U, V
        # ============================================

        # 1.- Absorption Element
        eta_I = 1.0 + 0.5 * eta0 * (
            eta_p * (singamma**2.0) + 0.5 * (eta_b + eta_r) * (1.0 + cosgamma**2.0)
        )

        # Common terms
        diff_eta = 0.5 * eta0 * (eta_p - 0.5 * (eta_b + eta_r)) * (singamma**2.0)
        diff_rho = 0.5 * eta0 * (rho_p - 0.5 * (rho_b + rho_r)) * (singamma**2.0)

        # Construct vectors eta_vec and rho_vec for (Q, U, V) components
        # eta_Q = diff_eta * cos2xi
        # eta_U = diff_eta * sin2xi
        # eta_V = 0.5 * eta0 * (eta_r - eta_b) * cosgamma

        # rho_Q = diff_rho * cos2xi
        # rho_U = diff_rho * sin2xi
        # rho_V = 0.5 * eta0 * (rho_r - rho_b) * cosgamma

        # Stack to form vectors (last dim size 3)
        # shapes: (batch, n_lambda)
        eta_V_term = 0.5 * eta0 * (eta_r - eta_b) * cosgamma
        rho_V_term = 0.5 * eta0 * (rho_r - rho_b) * cosgamma

        # (batch, n_lambda, 3)
        # Note: eta_vec is constructed to match (Q, U, V) contributions order
        eta_vec = torch.stack(
            [diff_eta * cos2xi, diff_eta * sin2xi, eta_V_term], dim=-1
        )
        rho_vec = torch.stack(
            [diff_rho * cos2xi, diff_rho * sin2xi, rho_V_term], dim=-1
        )

        # Vector operations
        # dot product: sum(a*b, dim=-1) -> (batch, n_lambda, 1) (keep dim for broadcasting if needed)
        eta_sq_sum = torch.sum(eta_vec**2, dim=-1)  # |eta|^2
        rho_sq_sum = torch.sum(rho_vec**2, dim=-1)  # |rho|^2
        eta_dot_rho = torch.sum(eta_vec * rho_vec, dim=-1)

        # Cross product: eta x rho
        cross_prod = torch.linalg.cross(eta_vec, rho_vec, dim=-1)

        # Matrix Solution (analytical)
        # Det = eta_I^2 * (eta_I^2 - |eta|^2 + |rho|^2) - (eta . rho)^2
        Det = eta_I**2 * (eta_I**2 - eta_sq_sum + rho_sq_sum) - eta_dot_rho**2
        IDet = 1.0 / Det

        # Source function terms: S_0, S_1 are (batch, 1)
        term_S1 = IDet * S_1

        # I = S_0 + term_S1 * eta_I * (eta_I^2 + |rho|^2)
        I = S_0 + term_S1 * eta_I * (eta_I**2 + rho_sq_sum)

        # QUV vector calculation
        # Formula derived from legacy scalar equations:
        # Vec = -term_S1 * ( eta_I^2 * eta_vec - eta_I * cross_prod + (eta . rho) * rho_vec )

        # Broadcast scalars to (batch, n_lambda, 1)
        eta_I_uns = eta_I.unsqueeze(-1)
        eta_dot_rho_uns = eta_dot_rho.unsqueeze(-1)

        QUV_vec = -term_S1.unsqueeze(-1) * (
            (eta_I_uns**2) * eta_vec
            - eta_I_uns * cross_prod
            + eta_dot_rho_uns * rho_vec
        )

        # Unpack Q, U, V
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
