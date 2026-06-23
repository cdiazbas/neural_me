from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.atmospheres.stereoscopic import StereoscopicAtmosphere
from ..core.warp import differentiable_tps_warp

class DegradationOperator(nn.Module, ABC):
    """
    Abstract base class for instrumental degradation (Stray Light, PSF, warping, etc.)
    """

    @abstractmethod
    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        pass


class LegacyActiveOperator(DegradationOperator):
    """
    Implements the mixing logic from the legacy `ME_combinado_mecpp_caliV.py`.

    Mathematical form:
        out = epsilon * stokes + (1 - epsilon) * <stokes>

    Here, `epsilon` acts as the "clean local light filling factor".
    - If epsilon = 0.245, then 24.5% is true local light and 75.5% is scattered/stray light.
    """

    def __init__(self, epsilon: Optional[float] = None, tune_epsilon: bool = True):
        super().__init__()
        if epsilon is not None:
            if tune_epsilon:
                self.epsilon = nn.Parameter(torch.tensor([epsilon]))
            else:
                self.register_buffer("epsilon", torch.tensor([epsilon]))
        else:
            self.epsilon = None

        eps_str = f"ε={epsilon:.4f}" if epsilon is not None else "ε=None (using atmosphere ε)"
        print(f"LegacyActiveOperator initialized | {eps_str} | tune={tune_epsilon}")

    def clamp_parameters(self):
        if self.epsilon is not None and isinstance(self.epsilon, nn.Parameter):
            self.epsilon.data.clamp_(0.0, 1.0)

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        I, Q, U, V = stokes
        epsilon = self.epsilon if self.epsilon is not None else atmosphere.epsilon

        # Compute spatial mean (batch/pixels dim is 0)
        I_mean = I.mean(dim=0, keepdim=True)
        Q_mean = Q.mean(dim=0, keepdim=True)
        U_mean = U.mean(dim=0, keepdim=True)
        V_mean = V.mean(dim=0, keepdim=True)

        I_new = epsilon * I + (1.0 - epsilon) * I_mean
        Q_new = epsilon * Q + (1.0 - epsilon) * Q_mean
        U_new = epsilon * U + (1.0 - epsilon) * U_mean
        V_new = epsilon * V + (1.0 - epsilon) * V_mean

        return I_new, Q_new, U_new, V_new


class StrayLightOperator(DegradationOperator):
    """
    Implements a standard stray light correction mixing:
        out = (1 - epsilon) * stokes + epsilon * <stokes>

    Here, `epsilon` acts as the "stray light fraction" (astrophysical standard).
    - If epsilon = 0.245, then 24.5% is scattered/stray light and 75.5% is true local light.
    
    This is mathematically equivalent to LegacyActiveOperator with the mapping
    epsilon_straylight = 1.0 - epsilon_legacy.
    """

    def __init__(self, epsilon: Optional[float] = None, tune_epsilon: bool = True):
        super().__init__()
        if epsilon is not None:
            if tune_epsilon:
                self.epsilon = nn.Parameter(torch.tensor([epsilon]))
            else:
                self.register_buffer("epsilon", torch.tensor([epsilon]))
        else:
            self.epsilon = None

        eps_str = f"ε={epsilon:.4f}" if epsilon is not None else "ε=None (using atmosphere ε)"
        print(f"StrayLightOperator initialized | {eps_str} | tune={tune_epsilon}")

    def clamp_parameters(self):
        if self.epsilon is not None and isinstance(self.epsilon, nn.Parameter):
            self.epsilon.data.clamp_(0.0, 1.0)

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        I, Q, U, V = stokes
        epsilon = self.epsilon if self.epsilon is not None else atmosphere.epsilon

        I_mean = I.mean(dim=0, keepdim=True)
        Q_mean = Q.mean(dim=0, keepdim=True)
        U_mean = U.mean(dim=0, keepdim=True)
        V_mean = V.mean(dim=0, keepdim=True)

        I_new = (1.0 - epsilon) * I + epsilon * I_mean
        Q_new = (1.0 - epsilon) * Q + epsilon * Q_mean
        U_new = (1.0 - epsilon) * U + epsilon * U_mean
        V_new = (1.0 - epsilon) * V + epsilon * V_mean

        return I_new, Q_new, U_new, V_new


class SpatialPSFOperator(DegradationOperator):
    """
    Implements spatial PSF degradation using 2D convolution.
    Convolves each Stokes and wavelength channel spatially.
    """

    def __init__(self, kernel: torch.Tensor):
        super().__init__()
        # kernel of shape (1, 1, Ky, Kx) or (Ky, Kx)
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)
        self.padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
        print(f"SpatialPSFOperator initialized | Kernel shape: {tuple(kernel.shape[-2:])}")

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        ny, nx = atmosphere.grid_shape
        out = []
        kernel = self.kernel.to(dtype=stokes[0].dtype, device=stokes[0].device)
        for S in stokes:
            n_lambda = S.shape[-1]
            # Transpose to (n_lambda, 1, Ny, Nx) to apply conv2d per wavelength channel
            S_grid = S.T.reshape(n_lambda, 1, ny, nx)
            S_conv = F.conv2d(S_grid, kernel, padding=self.padding)
            out.append(S_conv.reshape(n_lambda, ny * nx).T)
        return tuple(out)


class SpectralPSFOperator(DegradationOperator):
    """
    Implements spectral PSF degradation using 1D convolution along wavelengths.
    """

    def __init__(self, kernel: torch.Tensor):
        super().__init__()
        # kernel of shape (1, 1, K_lambda) or (K_lambda,)
        if kernel.dim() == 1:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)
        self.padding = kernel.shape[-1] // 2
        print(f"SpectralPSFOperator initialized | Kernel length: {kernel.shape[-1]}")

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        out = []
        kernel = self.kernel.to(dtype=stokes[0].dtype, device=stokes[0].device)
        for S in stokes:
            # Reshape to (N_pixels, 1, n_lambda) for 1D convolution along wavelengths
            S_conv = F.conv1d(S.unsqueeze(1), kernel, padding=self.padding)
            out.append(S_conv.squeeze(1))
        return tuple(out)


class WarpOperator(DegradationOperator):
    """
    Implements coordinate alignment mapping using thin plate splines (TPS) warping.
    Warps the spatial Stokes grids from reference to target coordinates.
    """

    def __init__(self, src_pts: torch.Tensor, dst_pts: torch.Tensor):
        super().__init__()
        self.register_buffer("src_pts", src_pts)
        self.register_buffer("dst_pts", dst_pts)
        print(f"WarpOperator initialized | Control points: {src_pts.shape[0]}")

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        ny, nx = atmosphere.grid_shape
        out = []
        src_pts = self.src_pts.to(dtype=stokes[0].dtype, device=stokes[0].device)
        dst_pts = self.dst_pts.to(dtype=stokes[0].dtype, device=stokes[0].device)
        for S in stokes:
            n_lambda = S.shape[-1]
            # Transpose to (n_lambda, 1, Ny, Nx) to warp all wavelength channels in batch
            S_grid = S.T.reshape(n_lambda, 1, ny, nx)
            S_warped = differentiable_tps_warp(S_grid, src_pts, dst_pts)
            out.append(S_warped.reshape(n_lambda, ny * nx).T)
        return tuple(out)
