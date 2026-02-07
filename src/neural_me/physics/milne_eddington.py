
import torch
import torch.nn as nn
from copy import deepcopy
from .zeeman import fvoigt, paramLine

class MilneEddington(nn.Module):
    def __init__(self, nlinea=None):
        super().__init__()
        if nlinea is not None:
            self.param = paramLine(nlinea)
        else:
            self.param = None

    @torch.compile
    def stokesSyn(self, param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1):
        
        # Pi square factor:
        sqrtpi = 1. / 1.7724538509055160272981674833411451827975494561223871282138077898 # sqrt(pi)

        # Magnetic field direction:
        singamma = torch.sin(gamma)
        cosgamma = torch.cos(gamma)
        sin2xi = torch.sin(2 * xi)
        cos2xi = torch.cos(2 * xi)

        # Calculating the Zeeman Pattern
        # Param structure: [[dpi, dsr, dsb], [spi, ssr, ssb],meta]
        # These are lists of arrays/tensors? usually shapes like (1, n_comp)
        # We need to ensure they are tensors on the correct device/dtype
        
        dpi_0, dsr_0, dsb_0 = param[0]
        spi_0, ssr_0, ssb_0 = param[1]
        l0 = param[2][7]

        # Ensure parameters are on the same device as input B
        device = B.device
        dpi_0 = dpi_0.to(device); dsr_0 = dsr_0.to(device); dsb_0 = dsb_0.to(device)
        spi_0 = spi_0.to(device); ssr_0 = ssr_0.to(device); ssb_0 = ssb_0.to(device)

        # Lorentz factor
        lB = 4.67E-13 * l0**2. * B # shape (batch, 1)

        # Apply Lorentz factor to splittings
        # splitting shapes: (1, n_comp) -> (batch, n_comp) after * (batch, 1)
        dpi = dpi_0 * lB
        dsr = dsr_0 * lB
        dsb = dsb_0 * lB

        # VLOS: km2A
        cc = 3.0E+5        # veloc luz [km/s]
        vlosA = l0 * vlos / cc # shape (batch, 1)

        # x shape: (batch, n_lambda)
        # We want to broadcast everything to (batch, n_lambda, n_comp)
        
        # Reshape inputs for broadcasting
        # x: (batch, n_lambda, 1)
        x_in = x.unsqueeze(-1)
        # ddop: (batch, 1, 1)
        ddop_in = ddop.unsqueeze(-1)
        # vlosA: (batch, 1, 1)
        vlosA_in = vlosA.unsqueeze(-1)
        # a: (batch, 1, 1)
        a_in = a.unsqueeze(-1)
        
        # Prepare splittings: (batch, 1, n_comp)
        dpi_in = dpi.unsqueeze(1)
        dsr_in = dsr.unsqueeze(1)
        dsb_in = dsb.unsqueeze(1)
        
        # Prepare strengths: (1, 1, n_comp)
        # spi_0 is (1, n_comp). 
        # We need (1, 1, n_comp).
        spi_in = spi_0.unsqueeze(1)
        ssr_in = ssr_0.unsqueeze(1)
        ssb_in = ssb_0.unsqueeze(1)
        

        # ============================================
        # Profiles absorption and dispersion - VECTORIZED

        # Function to compute eta/rho for a set of components
        def compute_profile(shifts, strengths):
            # xx: (batch, n_lambda, n_comp)
            xx = (x_in - shifts - vlosA_in) / ddop_in
            
            # fvoigt handles broadcasting: (batch, n_lambda, n_comp)
            H, F = fvoigt(a_in, xx) 
            
            # Sum over components (last dim)
            # H * strengths: (batch, n_lambda, n_comp)
            # sum -> (batch, n_lambda)
            # Sum over components (last dim)
            # H * strengths: (batch, n_lambda, n_comp)
            # sum -> (batch, n_lambda)
            # ddop is (batch, 1), unsqueezed it was (batch, 1, 1). 
            # We want to divide by ddop (batch, 1). 
            # ddop_in is (batch, 1, 1). squeeze(-1) makes it (batch, 1). 
            # This should work.
            # Double check shapes:
            # H: (batch, n_lambda, n_comp)
            # strengths: (1, 1, n_comp)
            
            term1 = torch.sum(H * strengths, dim=-1) 
            term2 = torch.sum(2. * F * strengths, dim=-1)
            
            # term1 is (batch, n_lambda)
            # ddop is (batch, 1)
            
            eta = term1 * sqrtpi / ddop
            rho = term2 * sqrtpi / ddop
            return eta, rho

        eta_p, rho_p = compute_profile(dpi_in, spi_in)
        eta_b, rho_b = compute_profile(dsb_in, ssb_in)
        eta_r, rho_r = compute_profile(dsr_in, ssr_in)

        # ============================================
        # Propagation Matrix Elements

        # 1.- Absorption Element
        # Shapes: eta_p, eta_b, eta_r are (batch, n_lambda)
        # sin/cos factors are (batch, 1) -> broadcast ok
        
        eta_I = 1.0 + 0.5 * eta0 * \
            (eta_p * (singamma**2.) + 0.5 * (eta_b + eta_r) * (1. + cosgamma**2.))
        
        # 2.- Dichroism Elements
        eta_Q = 0.5 * eta0 * (eta_p - 0.5 * (eta_b + eta_r)) * (singamma**2.) * cos2xi
        eta_U = 0.5 * eta0 * (eta_p - 0.5 * (eta_b + eta_r)) * (singamma**2.) * sin2xi
        eta_V = 0.5 * eta0 * (eta_r - eta_b) * cosgamma
        
        # 3.- Dispersion Elements
        rho_Q = 0.5 * eta0 * (rho_p - 0.5 * (rho_b + rho_r)) * (singamma**2.) * cos2xi
        rho_U = 0.5 * eta0 * (rho_p - 0.5 * (rho_b + rho_r)) * (singamma**2.) * sin2xi
        rho_V = 0.5 * eta0 * (rho_r - rho_b) * cosgamma

        # Matrix Solution (analytical)
        Det = eta_I**2. * (eta_I**2. - eta_Q**2. - eta_U**2. - eta_V**2. + rho_Q**2. +
                           rho_U**2. + rho_V**2.) - (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)**2.
        
        IDet = 1. / Det
        
        # Source function terms: S_0, S_1 are (batch, 1)
        term_S1 = IDet * S_1
        
        I = S_0 + term_S1 * eta_I * (eta_I**2. + rho_Q**2. + rho_U**2. + rho_V**2.)
        Q = -term_S1 * (eta_I**2. * eta_Q + eta_I * (eta_V * rho_U - eta_U * rho_V) + rho_Q * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        U = -term_S1 * (eta_I**2. * eta_U + eta_I * (eta_Q * rho_V - eta_V * rho_Q) + rho_U * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        V = -term_S1 * (eta_I**2. * eta_V + eta_I * (eta_U * rho_Q - eta_Q * rho_U) + rho_V * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))

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

        B = params[:,0][:,None]*1e3
        gamma = params[:,1][:,None]
        xi = params[:,2][:,None]
        vlos = params[:,3][:,None]
        eta0 = params[:,4][:,None]
        a = params[:,5][:,None]
        ddop = params[:,6][:,None]
        S_0 = params[:,7][:,None]
        S_1 = params[:,8][:,None]
        
        return self.stokesSyn(self.param, wl, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)

