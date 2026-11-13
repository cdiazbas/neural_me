import math
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.atmospheres.stereoscopic import StereoscopicAtmosphere
from ..core.warp import differentiable_tps_warp

def _next_fast_len(n: int) -> int:
    """The smallest length >= n with prime factors 2, 3 and 5 only."""
    while True:
        k = n
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return n
        n += 1


class DegradationOperator(nn.Module, ABC):
    """
    Abstract base class for instrumental degradation (Stray Light, PSF, warping, etc.)
    """

    @abstractmethod
    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        pass




class StrayLightOperator(DegradationOperator):
    """
    Implements a standard stray light correction mixing:
        out = (1 - epsilon) * stokes + epsilon * <stokes>

    Here, `epsilon` acts as the "stray light fraction" (astrophysical standard).
    - If epsilon = 0.245, then 24.5% is scattered/stray light and 75.5% is true local light.
    
    This corresponds to a complementary convention where
    epsilon_straylight = 1.0 - epsilon_clean_fraction.
    """

    def __init__(self, epsilon: float, tune_epsilon: bool = True):
        super().__init__()
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if tune_epsilon:
            et = torch.tensor([float(epsilon)], dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            self.raw_epsilon = nn.Parameter(torch.logit(et))
            self._epsilon_trainable = True
        else:
            self.register_buffer("fixed_epsilon", torch.tensor([float(epsilon)], dtype=torch.float32))
            self._epsilon_trainable = False
        print(f"StrayLightOperator initialized | ε={epsilon:.4f} | tune={tune_epsilon}")

    def _get_epsilon(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._epsilon_trainable:
            return torch.sigmoid(self.raw_epsilon).to(device=device, dtype=dtype)
        return self.fixed_epsilon.to(device=device, dtype=dtype)

    def clamp_parameters(self):
        if self._epsilon_trainable:
            self.raw_epsilon.data.clamp_(-12.0, 12.0)

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        I, Q, U, V = stokes
        epsilon = self._get_epsilon(device=I.device, dtype=I.dtype)

        I_mean = I.mean(dim=0, keepdim=True)
        Q_mean = Q.mean(dim=0, keepdim=True)
        U_mean = U.mean(dim=0, keepdim=True)
        V_mean = V.mean(dim=0, keepdim=True)

        I_new = (1.0 - epsilon) * I + epsilon * I_mean
        Q_new = (1.0 - epsilon) * Q + epsilon * Q_mean
        U_new = (1.0 - epsilon) * U + epsilon * U_mean
        V_new = (1.0 - epsilon) * V + epsilon * V_mean

        return I_new, Q_new, U_new, V_new


class PSFKernel(nn.Module):
    """
    Learnable 2D PSF kernel parameterized by FWHM.

    Supported models:
    - gaussian
    - moffat: ``(1 + r^2 / alpha^2)^-beta`` with ``beta = moffat_beta`` (> 1);
      Lorentzian-like wings that still integrate to a finite value, so the
      shape is set by the FWHM, not by ``kernel_size``.
    - lorentzian (also accepts legacy spelling 'lorenztian'): a 2-D
      Lorentzian does not integrate to a finite value, so after normalising
      on the ``kernel_size`` support its shape depends on that support.
      Kept for compatibility; prefer 'moffat'.
    """

    def __init__(
        self,
        fwhm: float,
        kernel_size: int,
        model: str = "gaussian",
        tune_fwhm: bool = True,
        verbose: bool = True,
        moffat_beta: float = 2.5,
    ):
        super().__init__()
        if fwhm <= 0.0:
            raise ValueError("fwhm must be > 0")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        model_norm = model.lower()
        if model_norm == "lorenztian":
            model_norm = "lorentzian"
        if model_norm not in ["gaussian", "lorentzian", "moffat"]:
            raise ValueError("model must be 'gaussian', 'moffat' or 'lorentzian'")
        if model_norm == "moffat" and moffat_beta <= 1.0:
            raise ValueError("moffat_beta must be > 1 for a normalisable PSF")
        if model_norm == "lorentzian":
            warnings.warn(
                "PSFKernel(model='lorentzian'): a 2-D Lorentzian is not normalisable, "
                "so the kernel shape depends on kernel_size; use model='moffat'.",
                stacklevel=2,
            )
        self.moffat_beta = float(moffat_beta)

        self.kernel_size = int(kernel_size)
        self.model = model_norm
        self.tune_fwhm = tune_fwhm

        if tune_fwhm:
            self.raw_log_fwhm = nn.Parameter(torch.log(torch.tensor([fwhm], dtype=torch.float32)))
        else:
            self.register_buffer("fixed_fwhm", torch.tensor([fwhm], dtype=torch.float32))

        if verbose:
            print(
                f"PSFKernel initialized | model={self.model} | kernel_size={self.kernel_size} | "
                f"fwhm={fwhm:.4f} | tune_fwhm={self.tune_fwhm}"
            )

    def _get_fwhm(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.tune_fwhm:
            return torch.exp(self.raw_log_fwhm).to(device=device, dtype=dtype).clamp_min(1e-6)
        return self.fixed_fwhm.to(device=device, dtype=dtype)

    def clamp_parameters(self):
        if self.tune_fwhm:
            self.raw_log_fwhm.data.clamp_(math.log(1e-3), math.log(1e5))

    def forward(self) -> torch.Tensor:
        device = self.raw_log_fwhm.device if self.tune_fwhm else self.fixed_fwhm.device
        dtype = self.raw_log_fwhm.dtype if self.tune_fwhm else self.fixed_fwhm.dtype
        fwhm = self._get_fwhm(device=device, dtype=dtype)

        coords = torch.arange(self.kernel_size, device=device, dtype=dtype) - (self.kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr2 = xx**2 + yy**2

        if self.model == "gaussian":
            sigma = fwhm / (2.0 * torch.sqrt(torch.tensor(2.0 * torch.log(torch.tensor(2.0)), device=device, dtype=dtype)))
            kernel2d = torch.exp(-0.5 * rr2 / (sigma**2))
        elif self.model == "moffat":
            beta = self.moffat_beta
            alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
            kernel2d = (1.0 + rr2 / (alpha**2)) ** (-beta)
        else:
            gamma = fwhm / 2.0
            kernel2d = 1.0 / (1.0 + rr2 / (gamma**2))

        kernel2d = kernel2d / kernel2d.sum().clamp_min(1e-12)
        return kernel2d.unsqueeze(0).unsqueeze(0)


class SpatialPSFOperator(DegradationOperator):
    """
    Implements spatial PSF degradation using 2D convolution.
    Convolves each Stokes and wavelength channel spatially.
    Supports both FFT-based ('fft') and direct ('conv2d') convolution.
    Supports configurable boundary padding ('reflect', 'replicate', etc.).
    Optionally normalizes each spatial kernel to unity area.
    """

    def __init__(
        self,
        kernel: Union[torch.Tensor, nn.Module],
        method: str = "fft",
        padding_mode: str = "reflect",
        normalize_kernel: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.kernel_model: Optional[nn.Module] = None
        self.normalize_kernel = normalize_kernel
        if isinstance(kernel, nn.Module):
            self.kernel_model = kernel
            preview = self.kernel_model()
            if preview.dim() == 2:
                preview = preview.unsqueeze(0).unsqueeze(0)
            if preview.dim() != 4:
                raise ValueError("kernel module must return tensor of shape (Ky, Kx) or (1, 1, Ky, Kx)")
            self.padding = (preview.shape[-2] // 2, preview.shape[-1] // 2)
        else:
            # kernel tensor of shape (1, 1, Ky, Kx) or (Ky, Kx)
            if kernel.dim() == 2:
                kernel = kernel.unsqueeze(0).unsqueeze(0)
            if kernel.dim() != 4:
                raise ValueError("kernel tensor must be shape (Ky, Kx) or (1, 1, Ky, Kx)")
            self.register_buffer("kernel", kernel)
            self.padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)

        self.method = method.lower()
        self.padding_mode = padding_mode.lower()
        assert self.method in ["fft", "conv2d"], "method must be 'fft' or 'conv2d'"
        if self.method == "conv2d":
            ky, kx = 2 * self.padding[0] + 1, 2 * self.padding[1] + 1
            shape = (self.kernel_model() if self.kernel_model is not None else self.kernel).shape[-2:]
            if tuple(shape) != (ky, kx):
                raise ValueError(
                    f"method='conv2d' needs an odd-sized kernel, got {tuple(shape)}; "
                    "use method='fft' or an odd size"
                )
        assert self.padding_mode in ["reflect", "replicate", "constant"], "padding_mode must be 'reflect', 'replicate', or 'constant'"
        if verbose:
            if self.kernel_model is not None:
                print(
                    f"SpatialPSFOperator initialized | Kernel source: {type(self.kernel_model).__name__} "
                    f"| Kernel shape: {tuple(preview.shape[-2:])} | method: {self.method} | "
                    f"padding_mode: {self.padding_mode} | normalize_kernel: {self.normalize_kernel}"
                )
            else:
                print(
                    f"SpatialPSFOperator initialized | Kernel shape: {tuple(kernel.shape[-2:])} "
                    f"| method: {self.method} | padding_mode: {self.padding_mode} "
                    f"| normalize_kernel: {self.normalize_kernel}"
                )

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        ny, nx = atmosphere.grid_shape
        device = stokes[0].device
        dtype = stokes[0].dtype
        if self.kernel_model is not None:
            kernel = self.kernel_model().to(dtype=dtype, device=device)
            if kernel.dim() == 2:
                kernel = kernel.unsqueeze(0).unsqueeze(0)
            if kernel.dim() != 4:
                raise ValueError("kernel module must return tensor of shape (Ky, Kx) or (1, 1, Ky, Kx)")
        else:
            kernel = self.kernel.to(dtype=dtype, device=device)

        if self.normalize_kernel:
            kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        
        self.padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
        pad_y, pad_x = self.padding
        # Reflection padding requires pad < input size per spatial dimension.
        # Fall back to replicate for tiny test grids / oversized kernels.
        effective_padding_mode = self.padding_mode
        if self.padding_mode == "reflect" and (pad_y >= ny or pad_x >= nx):
            effective_padding_mode = "replicate"
        
        out = []
        if self.method == "conv2d":
            for S in stokes:
                n_lambda = S.shape[-1]
                # Transpose to (n_lambda, 1, Ny, Nx) to apply conv2d per wavelength channel
                S_grid = S.T.reshape(n_lambda, 1, ny, nx)
                S_padded = F.pad(S_grid, (pad_x, pad_x, pad_y, pad_y), mode=effective_padding_mode)
                # conv2d is a cross-correlation: flip the PSF to convolve, as the FFT path does
                S_conv = F.conv2d(S_padded, torch.flip(kernel, (-2, -1)), padding=0)
                out.append(S_conv.reshape(n_lambda, ny * nx).T)
        else: # fft
            # All channels in one batch with one OTF. The padded size goes up to a
            # fast FFT length (512 + 14 = 526 = 2 x 263 costs twice 540); the extra
            # padding lies beyond the reach of the kernel, so the result is the same.
            ky, kx = kernel.shape[-2], kernel.shape[-1]
            ny_pad = _next_fast_len(ny + 2 * pad_y)
            nx_pad = _next_fast_len(nx + 2 * pad_x)
            top, left = pad_y + (ny_pad - ny - 2 * pad_y) // 2, pad_x + (nx_pad - nx - 2 * pad_x) // 2
            bottom, right = ny_pad - ny - top, nx_pad - nx - left
            if effective_padding_mode == "reflect" and (max(top, bottom) >= ny or max(left, right) >= nx):
                effective_padding_mode = "replicate"
            sizes = [S.shape[-1] for S in stokes]
            # (n_channels, ny, nx), Stokes after Stokes
            S_grid = torch.cat([S.T.reshape(n_lambda, ny, nx) for S, n_lambda in zip(stokes, sizes)], dim=0)
            S_padded = F.pad(S_grid, (left, right, top, bottom), mode=effective_padding_mode)

            padded_kernel = torch.zeros((ny_pad, nx_pad), dtype=dtype, device=device)
            padded_kernel[:ky, :kx] = kernel.squeeze(0).squeeze(0)
            padded_kernel = torch.roll(padded_kernel, shifts=(-(ky // 2), -(kx // 2)), dims=(0, 1))
            OTF = torch.fft.rfft2(padded_kernel)

            S_conv_pad = torch.fft.irfft2(torch.fft.rfft2(S_padded) * OTF, s=(ny_pad, nx_pad))
            S_conv = S_conv_pad[:, top : top + ny, left : left + nx]
            out = [C.reshape(n_lambda, ny * nx).T for C, n_lambda in zip(S_conv.split(sizes, dim=0), sizes)]

        return tuple(out)


class SpectralPSFOperator(DegradationOperator):
    """
    Implements spectral PSF degradation using 1D convolution along wavelengths.
    """

    def __init__(self, kernel: torch.Tensor, normalize_kernel: bool = True):
        super().__init__()
        # kernel of shape (1, 1, K_lambda) or (K_lambda,)
        if kernel.dim() == 1:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
        if kernel.shape[-1] % 2 == 0:
            raise ValueError(
                f"spectral kernel must have an odd number of samples, got {kernel.shape[-1]}"
            )
        self.normalize_kernel = normalize_kernel
        self.register_buffer("kernel", kernel)
        self.padding = kernel.shape[-1] // 2
        print(
            f"SpectralPSFOperator initialized | Kernel length: {kernel.shape[-1]} "
            f"| normalize_kernel: {self.normalize_kernel}"
        )

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        out = []
        kernel = self.kernel.to(dtype=stokes[0].dtype, device=stokes[0].device)
        if self.normalize_kernel:
            kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # Convolve (conv1d is a cross-correlation, so flip the kernel) and extend
        # the band edges with their last value instead of zeros, which would
        # darken the continuum there.
        kernel = torch.flip(kernel, (-1,))
        for S in stokes:
            # Reshape to (N_pixels, 1, n_lambda) for 1D convolution along wavelengths
            S_pad = F.pad(S.unsqueeze(1), (self.padding, self.padding), mode="replicate")
            S_conv = F.conv1d(S_pad, kernel, padding=0)
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


class _SparseProduct(torch.autograd.Function):
    """y = A x with the backward Aᵀ g, ~20x faster than autograd through the sparse product."""

    @staticmethod
    def forward(ctx, x, matrices):
        ctx.adjoint = matrices[1]
        return matrices[0].matmul(x)

    @staticmethod
    def backward(ctx, g):
        return ctx.adjoint.matmul(g), None


class SparseWarpOperator(DegradationOperator):
    """
    Fixed resampling of the Stokes maps from the model grid to the pixels of an
    instrument, stored as a sparse matrix: S_out = A S_in. It is linear in S and
    has no parameters, so its gradient is the constant Aᵀ.

    Same values as ``F.grid_sample(..., mode='bilinear', padding_mode='zeros',
    align_corners=True)`` averaged over the K samples of each output pixel, at a
    fraction of the cost: 25 ms against 1390 ms per forward + backward for 4 Stokes
    x 6 wavelengths, 512x512, K = 9 (16 threads). Use ``WarpOperator`` when the
    coordinates are control points of a TPS.

    Args:
        grid: (H, W, 2) or (H, W, K, 2) sampling positions on the input grid, (x, y)
            normalised with align_corners=True: -1 and +1 are the centres of the
            first and last input pixels. K samples per output pixel integrate its
            footprint.
        valid: (H, W), True on the output pixels with data; the others are 0.
            Default: all of them.
        input_shape: (ny, nx) of the input grid. Default: (H, W).

    Output and input are flat as everywhere else: (H W, n_lambda), (ny nx, n_lambda).
    A PSF after this operator takes the shape from ``atmosphere.grid_shape``, so
    with one in the chain (H, W) must equal the model grid.
    """

    def __init__(self, grid, valid=None, input_shape: Optional[Tuple[int, int]] = None):
        super().__init__()
        grid = torch.as_tensor(grid, dtype=torch.float64)
        if grid.dim() == 3:
            grid = grid.unsqueeze(2)
        if grid.dim() != 4 or grid.shape[-1] != 2:
            raise ValueError("grid must be (H, W, 2) or (H, W, K, 2)")
        H, W, K, _ = grid.shape
        ny, nx = (H, W) if input_shape is None else input_shape
        if valid is None:
            valid = torch.ones((H, W), dtype=torch.bool)
        valid = torch.as_tensor(valid).bool()
        if valid.shape != (H, W):
            raise ValueError(f"valid must be {(H, W)}, got {tuple(valid.shape)}")

        x = (grid[..., 0] + 1) / 2 * (nx - 1)
        y = (grid[..., 1] + 1) / 2 * (ny - 1)
        x0, y0 = torch.floor(x), torch.floor(y)
        fx, fy = x - x0, y - y0
        rows = torch.arange(H * W).reshape(H, W, 1).expand(H, W, K)
        keep = valid.unsqueeze(-1).expand(H, W, K)
        r, c, v = [], [], []
        for dy, dx, weight in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                               (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0.long() + dx, y0.long() + dy
            ok = keep & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny) & (weight > 0)
            r.append(rows[ok])
            c.append((yi * nx + xi)[ok])
            v.append(weight[ok] / K)
        A = torch.sparse_coo_tensor(torch.stack([torch.cat(r), torch.cat(c)]), torch.cat(v),
                                    (H * W, ny * nx)).coalesce()
        self.register_buffer("_indices", A.indices())
        self.register_buffer("_values", A.values())
        self.output_shape = (H, W)
        self.input_shape = (ny, nx)
        self._matrices: Dict[tuple, tuple] = {}
        print(f"SparseWarpOperator initialized | {ny}x{nx} -> {H}x{W} | {K} samples per pixel | "
              f"{int(valid.sum())} valid pixels | {A.values().numel()} non-zeros")

    @classmethod
    def from_file(cls, path, footprint: bool = True, input_shape: Optional[Tuple[int, int]] = None):
        """From the ``.npz`` of 03_coalignment/phi_reverse_warp.ipynb
        (:func:`~neural_me.core.coregistration.save_sampling_grid`): ``grid_sub``
        (H, W, K, 2), ``grid`` (H, W, 2) and ``valid`` (H, W). ``footprint=False``
        uses one sample per output pixel."""
        import numpy as np

        with np.load(path) as w:
            return cls(w["grid_sub" if footprint else "grid"], w["valid"], input_shape=input_shape)

    def matrices(self, dtype: torch.dtype, device: torch.device):
        """A and Aᵀ in CSR, built once per dtype and device."""
        key = (dtype, device)
        if key not in self._matrices:
            values = self._values.to(dtype=dtype, device=device)
            indices = self._indices.to(device)
            size = (self.output_shape[0] * self.output_shape[1], self.input_shape[0] * self.input_shape[1])
            A = torch.sparse_coo_tensor(indices, values, size).coalesce().to_sparse_csr()
            At = torch.sparse_coo_tensor(indices.flip(0), values, size[::-1]).coalesce().to_sparse_csr()
            self._matrices[key] = (A, At)
        return self._matrices[key]

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: Optional[StereoscopicAtmosphere] = None
    ) -> Tuple[torch.Tensor, ...]:
        matrices = self.matrices(stokes[0].dtype, stokes[0].device)
        return tuple(_SparseProduct.apply(S, matrices) for S in stokes)


class ComposeOperator(DegradationOperator):
    """Applies several operators in order, for a solver that takes one per view."""

    def __init__(self, *operators: DegradationOperator):
        super().__init__()
        if not operators:
            raise ValueError("ComposeOperator needs at least one operator")
        self.operators = nn.ModuleList(operators)

    def clamp_parameters(self):
        for op in self.operators:
            if hasattr(op, "clamp_parameters"):
                op.clamp_parameters()

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        for op in self.operators:
            stokes = op(stokes, atmosphere)
        return stokes


class TwoPSFOperator(DegradationOperator):
    """
    Implements the Sinjan et al. (2026) extended PSF model:
        PSF = w1 * PSF_PD + w2 * G_stray
    
    where G_stray is a wide Gaussian wing (default width = 600 pixels / 300").
    Applied via 2D FFT multiplication in Fourier space.
    """

    def __init__(
        self,
        psf_pd_kernel: torch.Tensor,
        w1: float = 0.9009,
        w2: float = 0.0991,
        sigma_stray_pixels: float = 600.0,
        padding_mode: str = "reflect",
        tune_stray_fraction: bool = False,
        tune_sigma_stray_pixels: bool = False,
        normalize_psf_pd_kernel: bool = True,
    ):
        super().__init__()
        # psf_pd_kernel of shape (Ky, Kx) or (1, 1, Ky, Kx)
        if psf_pd_kernel.dim() == 4:
            psf_pd_kernel = psf_pd_kernel.squeeze(0).squeeze(0)
        elif psf_pd_kernel.dim() == 3:
            psf_pd_kernel = psf_pd_kernel.squeeze(0)

        if psf_pd_kernel.dim() != 2:
            raise ValueError("psf_pd_kernel must be 2D (Ky, Kx) or broadcastable to it")
        if sigma_stray_pixels <= 0.0:
            raise ValueError("sigma_stray_pixels must be > 0")
        if (w1 + w2) <= 0.0:
            raise ValueError("w1 + w2 must be > 0")

        self.padding_mode = padding_mode.lower()
        if self.padding_mode not in ["reflect", "replicate", "constant"]:
            raise ValueError("padding_mode must be 'reflect', 'replicate', or 'constant'")
        self.normalize_psf_pd_kernel = normalize_psf_pd_kernel
            
        self.register_buffer("psf_pd_kernel", psf_pd_kernel)
        self.padding = (psf_pd_kernel.shape[-2] // 2, psf_pd_kernel.shape[-1] // 2)

        # Constrained parameterization:
        # stray_fraction in [0, 1] using sigmoid(raw_stray_fraction)
        # sigma_stray_pixels > 0 using exp(raw_log_sigma_stray)
        stray_fraction = float(w2 / (w1 + w2))
        self.tune_stray_fraction = tune_stray_fraction
        self.tune_sigma_stray_pixels = tune_sigma_stray_pixels

        if tune_stray_fraction:
            sf = torch.tensor([stray_fraction], dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            self.raw_stray_fraction = nn.Parameter(torch.logit(sf))
        else:
            self.register_buffer("fixed_stray_fraction", torch.tensor([stray_fraction], dtype=torch.float32))

        if tune_sigma_stray_pixels:
            self.raw_log_sigma_stray = nn.Parameter(
                torch.log(torch.tensor([sigma_stray_pixels], dtype=torch.float32))
            )
        else:
            self.register_buffer(
                "fixed_sigma_stray_pixels",
                torch.tensor([sigma_stray_pixels], dtype=torch.float32),
            )

        # Cache is only valid when parameters are fixed.
        self._cache_enabled = not (tune_stray_fraction or tune_sigma_stray_pixels)
        self._otf_cache = {}
        print(
            f"TwoPSFOperator initialized | stray_fraction={stray_fraction:.4f} "
            f"(tunable={tune_stray_fraction}), sigma_stray={sigma_stray_pixels} px "
            f"(tunable={tune_sigma_stray_pixels}), core shape={tuple(psf_pd_kernel.shape)} | "
            f"padding_mode: {self.padding_mode} | normalize_psf_pd_kernel: {self.normalize_psf_pd_kernel}"
        )

    def _get_stray_fraction(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.tune_stray_fraction:
            return torch.sigmoid(self.raw_stray_fraction).to(device=device, dtype=dtype)
        return self.fixed_stray_fraction.to(device=device, dtype=dtype)

    def _get_sigma_stray_pixels(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.tune_sigma_stray_pixels:
            # Positive by construction; small floor for numerical safety.
            return torch.exp(self.raw_log_sigma_stray).to(device=device, dtype=dtype).clamp_min(1e-6)
        return self.fixed_sigma_stray_pixels.to(device=device, dtype=dtype)

    def clamp_parameters(self):
        # Keep tunable parameters in a numerically safe range.
        if self.tune_stray_fraction:
            self.raw_stray_fraction.data.clamp_(-12.0, 12.0)
        if self.tune_sigma_stray_pixels:
            self.raw_log_sigma_stray.data.clamp_(math.log(1e-3), math.log(1e5))

    def _get_extended_otf(
        self,
        ny_pad: int,
        nx_pad: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (ny_pad, nx_pad, str(device), str(dtype))
        if self._cache_enabled and cache_key in self._otf_cache:
            return self._otf_cache[cache_key]

        # 1. Pad and shift psf_pd_kernel to match padded image size.
        ky, kx = self.psf_pd_kernel.shape
        padded_kernel = torch.zeros((ny_pad, nx_pad), dtype=dtype, device=device)
        psf_pd_kernel = self.psf_pd_kernel.to(dtype=dtype, device=device)
        if self.normalize_psf_pd_kernel:
            psf_pd_kernel = psf_pd_kernel / psf_pd_kernel.sum().clamp_min(1e-12)
        padded_kernel[:ky, :kx] = psf_pd_kernel
        padded_kernel = torch.roll(
            padded_kernel,
            shifts=(-(ky // 2), -(kx // 2)),
            dims=(0, 1),
        )

        # 2. Core OTF.
        OTF_pd = torch.fft.rfft2(padded_kernel)

        # 3. Analytical Gaussian-wing OTF.
        fy = torch.fft.fftfreq(ny_pad, d=1.0, device=device)
        fx = torch.fft.rfftfreq(nx_pad, d=1.0, device=device)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        f2 = FY**2 + FX**2
        sigma_stray_pixels = self._get_sigma_stray_pixels(device=device, dtype=dtype)
        stray_fraction = self._get_stray_fraction(device=device, dtype=dtype)
        core_fraction = 1.0 - stray_fraction

        pi = 3.141592653589793
        OTF_stray = torch.exp(
            -2.0 * (pi**2) * (sigma_stray_pixels**2) * f2
        ).to(dtype=OTF_pd.dtype)

        # 4. Extended OTF.
        OTF_extended = core_fraction * OTF_pd + stray_fraction * OTF_stray
        if self._cache_enabled:
            self._otf_cache[cache_key] = OTF_extended
        return OTF_extended

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        ny, nx = atmosphere.grid_shape
        device = stokes[0].device
        dtype = stokes[0].dtype
        pad_y, pad_x = self.padding
        effective_padding_mode = self.padding_mode
        if self.padding_mode == "reflect" and (pad_y >= ny or pad_x >= nx):
            effective_padding_mode = "replicate"

        ny_pad = ny + 2 * pad_y
        nx_pad = nx + 2 * pad_x
        OTF_extended = self._get_extended_otf(ny_pad, nx_pad, device, dtype)

        # Apply convolution in Fourier space with explicit boundary handling,
        # matching SpatialPSFOperator behavior.
        out = []
        for S in stokes:
            n_lambda = S.shape[-1]

            # Reshape S from (ny * nx, n_lambda) to (n_lambda, ny, nx)
            S_grid = S.T.reshape(n_lambda, ny, nx)

            # Pad to reduce edge artifacts and emulate non-circular boundaries.
            S_padded = F.pad(
                S_grid,
                (pad_x, pad_x, pad_y, pad_y),
                mode=effective_padding_mode,
            )

            S_fft = torch.fft.rfft2(S_padded)
            S_fft_conv = S_fft * OTF_extended.unsqueeze(0)

            S_conv_pad = torch.fft.irfft2(S_fft_conv, s=(ny_pad, nx_pad))
            S_conv = S_conv_pad[:, pad_y : pad_y + ny, pad_x : pad_x + nx]

            # Reshape back to (ny * nx, n_lambda)
            out.append(S_conv.reshape(n_lambda, ny * nx).T)

        return tuple(out)


class ExtendedPSFOperator(DegradationOperator):
    """
        Extended degradation model with two tunable physical parameters:
        1) PSF FWHM
        2) flat stray-light factor epsilon

        The model applies a spatial PSF convolution and then mixes with a flat
        FoV-mean component. Two mean-source conventions are supported through
        ``mean_source``:

        - ``mean_source='post_psf'``:
            S_out = (1 - epsilon) * (PSF * S) + epsilon * <PSF * S>_FoV

        - ``mean_source='pre_psf'`` (default):
            S_out = (1 - epsilon) * (PSF * S) + epsilon * <S>_FoV

        If the PSF and boundary handling preserve the spatial mean exactly, both
        forms are equivalent. In practice they can differ slightly due to finite
        FoV and padding/cropping effects.

        Notes:
        - ``mix_polarization_mean=True`` applies the same mean-mixing logic to
            Q, U, and V.
        - ``mix_polarization_mean=False`` mixes only I and attenuates Q/U/V by
            (1 - epsilon).
        - ``mask`` (ny, nx), True on the pixels with data: the PSF becomes a
            normalised convolution, PSF * (m S) / PSF * m, the FoV mean runs over
            the mask, and the pixels outside it are 0 on output. Their input
            values have no effect. Default None: every pixel counts.
    """

    def __init__(
        self,
        fwhm: float,
        epsilon: float,
        kernel_size: int = 31,
        model: str = "gaussian",
        tune_fwhm: bool = True,
        tune_epsilon: bool = True,
        method: str = "fft",
        padding_mode: str = "reflect",
        mix_polarization_mean: bool = True,
        mean_source: str = "pre_psf",
        moffat_beta: float = 2.5,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if fwhm < 0.0:
            raise ValueError("fwhm must be >= 0 (use 0.0 to disable PSF convolution)")
        if fwhm == 0.0 and tune_fwhm:
            raise ValueError("tune_fwhm=True requires fwhm > 0")
        self.mean_source = mean_source.lower()
        if self.mean_source not in ["post_psf", "pre_psf"]:
            raise ValueError("mean_source must be 'post_psf' or 'pre_psf'")

        self._use_psf = fwhm > 0.0
        if self._use_psf:
            self.psf_kernel = PSFKernel(
                fwhm=fwhm,
                kernel_size=kernel_size,
                model=model,
                tune_fwhm=tune_fwhm,
                verbose=False,
                moffat_beta=moffat_beta,
            )
            self.spatial_psf = SpatialPSFOperator(
                kernel=self.psf_kernel,
                method=method,
                padding_mode=padding_mode,
                verbose=False,
            )
        self.mix_polarization_mean = mix_polarization_mean
        if mask is not None:
            mask = torch.as_tensor(mask)
            if mask.dim() != 2:
                raise ValueError("mask must be (ny, nx)")
            mask = mask.bool().reshape(-1, 1)
            if not mask.any():
                raise ValueError("mask selects no pixels")
        self.register_buffer("mask", mask)

        if tune_epsilon:
            et = torch.tensor([float(epsilon)], dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            self.raw_epsilon = nn.Parameter(torch.logit(et))
            self._epsilon_trainable = True
        else:
            self.register_buffer("fixed_epsilon", torch.tensor([float(epsilon)], dtype=torch.float32))
            self._epsilon_trainable = False

        print(
            f"ExtendedPSFOperator initialized | "
            f"{'fwhm=' + f'{fwhm:.4f} (tunable=' + str(tune_fwhm) + ')' if self._use_psf else 'PSF=disabled (fwhm=0)'} | "
            f"epsilon={epsilon:.4f} (tunable={tune_epsilon}) | "
            f"mix_polarization_mean={mix_polarization_mean} | "
            f"mean_source={self.mean_source}"
        )

    def _get_epsilon(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._epsilon_trainable:
            return torch.sigmoid(self.raw_epsilon).to(device=device, dtype=dtype)
        return self.fixed_epsilon.to(device=device, dtype=dtype)

    def clamp_parameters(self):
        if self._use_psf and hasattr(self.psf_kernel, "clamp_parameters"):
            self.psf_kernel.clamp_parameters()
        if self._epsilon_trainable:
            self.raw_epsilon.data.clamp_(-12.0, 12.0)

    def forward(
        self, stokes: Tuple[torch.Tensor, ...], atmosphere: StereoscopicAtmosphere
    ) -> Tuple[torch.Tensor, ...]:
        I_in, Q_in, U_in, V_in = stokes
        m = None if self.mask is None else self.mask.to(dtype=I_in.dtype, device=I_in.device)
        if m is not None:
            I_in, Q_in, U_in, V_in = stokes = tuple(S * m for S in stokes)
        # Apply spatial PSF if enabled, otherwise pass through.
        if self._use_psf:
            I_psf, Q_psf, U_psf, V_psf = self.spatial_psf(stokes, atmosphere)
            if m is not None:
                weight = self.spatial_psf((m,), atmosphere)[0].clamp_min(1e-6)
                I_psf, Q_psf, U_psf, V_psf = I_psf / weight, Q_psf / weight, U_psf / weight, V_psf / weight
        else:
            I_psf, Q_psf, U_psf, V_psf = I_in, Q_in, U_in, V_in
        epsilon = self._get_epsilon(device=I_psf.device, dtype=I_psf.dtype)

        def fov_mean(S):
            if m is None:
                return S.mean(dim=0, keepdim=True)
            return (S * m).sum(dim=0, keepdim=True) / m.sum()

        # Flat FoV mean component: choose whether to compute mean from the
        # pre-PSF or post-PSF signal.
        if self.mean_source == "pre_psf":
            I_mean = fov_mean(I_in)
            if self.mix_polarization_mean:
                Q_mean = fov_mean(Q_in)
                U_mean = fov_mean(U_in)
                V_mean = fov_mean(V_in)
        else:
            I_mean = fov_mean(I_psf)
            if self.mix_polarization_mean:
                Q_mean = fov_mean(Q_psf)
                U_mean = fov_mean(U_psf)
                V_mean = fov_mean(V_psf)

        I_out = (1.0 - epsilon) * I_psf + epsilon * I_mean

        if self.mix_polarization_mean:
            Q_out = (1.0 - epsilon) * Q_psf + epsilon * Q_mean
            U_out = (1.0 - epsilon) * U_psf + epsilon * U_mean
            V_out = (1.0 - epsilon) * V_psf + epsilon * V_mean
        else:
            # Common physical approximation: only I carries flat additive stray-light.
            Q_out = (1.0 - epsilon) * Q_psf
            U_out = (1.0 - epsilon) * U_psf
            V_out = (1.0 - epsilon) * V_psf

        if m is not None:
            I_out, Q_out, U_out, V_out = I_out * m, Q_out * m, U_out * m, V_out * m
        return I_out, Q_out, U_out, V_out

