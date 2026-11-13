"""
Penalty on the part of the maps that a linear degradation erases.

The data of a view constrain a map x only through K x, with K the linear part
of its degradation: the warp A from the model grid to the instrument pixels,
the PSF P, or both (K = P A). Whatever lies in the null space of K moves
freely, and the fit fills it with noise (the chessboard at the scale of the
model pixels along the long axis of the PHI footprint). The penalty pulls
each map towards B K x, the map rebuilt from what the instrument sees:

    R(x) = sum_q w_q mean_mask (x_q - B K x_q)^2,   B = D^-1 K^T,  D = K^T K 1,

so B K is the normalised back-projection: a constant map is rebuilt exactly
and costs nothing, and only the structure that K removes is penalised.

P is the normalised convolution over the valid instrument pixels m, as
:class:`~neural_me.optimization.degradation.ExtendedPSFOperator` with a mask:
P z = G(m z) / G(m), P^T y = m G(y / G(m)). A is the matrix of
:class:`~neural_me.optimization.degradation.SparseWarpOperator`; A^T takes the
instrument pixels back to the model grid.

The penalty is quadratic, R = |L x|^2 with L = I - D^-1 K^T K, so its gradient
is L^T (2 w R / N) = g - K^T K (D^-1 g). It is computed by hand in a
``torch.autograd.Function`` with the same K and K^T: autograd through the
sparse product and the convolutions costs ~10x more. The Gaussian PSF is
applied as two 1-D convolutions (it factorises); a measured kernel by FFT.
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["NullSpacePenalty"]


class NullSpacePenalty:
    """
    ``sum_q w_q mean_mask (x_q - B K x_q)^2`` for K = P (PSF), A (warp) or P A,
    batched over the maps.

    Args:
        valid: (ny, nx) bool, the instrument pixels with data.
        fwhm: FWHM of a Gaussian PSF in instrument pixels. None or 0: no PSF in K.
        A: sparse (ny nx, n_model) torch tensor, the warp from the model grid to
            the instrument pixels (``SparseWarpOperator``'s matrix). None: the maps
            are on the instrument pixels. :meth:`from_warp` takes the operator.
        model_mask: (n_model,) or any shape with n_model elements, bool: the model
            pixels the mean runs over. Default: ``valid`` without a warp; with one,
            the model pixels the warp covers (K^T K 1 > 1e-4).
        ksize: size of the Gaussian kernel (odd), in instrument pixels.
        kernel: (k, k) array, a measured PSF used instead of the Gaussian (e.g.
            Hinode/SP, 41 x 41), normalised to unit sum. Applied by FFT as a
            'same' linear convolution with zero boundary; its transpose is the
            correlation (flipped kernel).
        dtype: of the operators and of the maps the penalty takes. float32 is
            what the inversions use; float64 for checks.

    Call it with the list of n maps (each with n_model elements, any shape) and
    n weights; it returns a scalar with the gradient to the maps.

    The PSF null space on the instrument pixels of a warped view is the case
    K = P applied to A x: build the penalty without ``A`` and with
    ``model_mask=valid``, and pass it the warped maps, e.g.
    ``warp((X.T,))[0].T`` for X (n, n_model) with ``warp`` the
    ``SparseWarpOperator`` (its backward is A^T).
    """

    def __init__(self, valid, fwhm: Optional[float] = None, A: Optional[torch.Tensor] = None, model_mask=None,
                 ksize: int = 15, kernel=None, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        self.ny, self.nx = valid.shape
        self.m = torch.as_tensor(np.asarray(valid), dtype=dtype).view(1, 1, self.ny, self.nx)
        self.use_psf = (fwhm is not None and fwhm > 0) or kernel is not None
        self.kernel = None
        if kernel is not None:
            k = torch.as_tensor(kernel, dtype=dtype)
            k = k / k.sum()
            self.r = k.shape[0] // 2
            H, W = self.ny + 2 * self.r, self.nx + 2 * self.r
            kp = torch.zeros(H, W, dtype=dtype)
            kp[:k.shape[0], :k.shape[1]] = k
            kp = torch.roll(kp, (-self.r, -self.r), (0, 1))      # kernel centre at (0, 0): circular = linear on the pad
            self.Kf = torch.fft.rfft2(kp)
            self.fshape = (H, W)
            self.kernel = k
            self.Gm = self._G(self.m).clamp_min(1e-6)
        elif self.use_psf:
            s = fwhm / (2 * math.sqrt(2 * math.log(2)))
            r = torch.arange(ksize, dtype=dtype) - ksize // 2
            g = torch.exp(-r ** 2 / (2 * s ** 2))
            g = g / g.sum()
            self.gx = g.view(1, 1, 1, ksize)
            self.gy = g.view(1, 1, ksize, 1)
            self.pad = ksize // 2
            self.Gm = self._G(self.m).clamp_min(1e-6)
        self.A = None
        self.n_model = self.ny * self.nx
        if A is not None:
            if A.layout != torch.sparse_coo:
                A = A.to_sparse_coo()
            A = A.coalesce().to(dtype)
            if A.shape[0] != self.ny * self.nx:
                raise ValueError(f"A has {A.shape[0]} rows, valid has {self.ny * self.nx} pixels")
            self.A = A.to_sparse_csr()
            self.At = A.t().coalesce().to_sparse_csr()
            self.n_model = A.shape[1]
        ones = torch.ones(1, self.n_model, dtype=dtype)
        self.KtK1 = self._Kt(self._K(ones)).clamp_min(1e-6)              # (1, n_model)
        if model_mask is None:
            model_mask = (self.KtK1[0] > 1e-4) if self.A is not None else torch.as_tensor(np.asarray(valid)).reshape(-1)
        self.mask = torch.as_tensor(np.asarray(model_mask)).reshape(-1).bool()
        if self.mask.numel() != self.n_model:
            raise ValueError(f"model_mask has {self.mask.numel()} elements, the model {self.n_model}")

    @classmethod
    def from_warp(cls, warp, valid=None, fwhm: Optional[float] = None, model_mask=None, ksize: int = 15,
                  kernel=None, dtype: torch.dtype = torch.float32):
        """K = A (``fwhm`` and ``kernel`` None) or P A, with A the matrix of a
        :class:`~neural_me.optimization.degradation.SparseWarpOperator`.

        ``valid`` (H, W) defaults to the output pixels with a non-empty row of A,
        which are the valid pixels the operator was built with."""
        H, W = warp.output_shape
        A = torch.sparse_coo_tensor(warp._indices, warp._values,
                                    (H * W, warp.input_shape[0] * warp.input_shape[1])).coalesce()
        if valid is None:
            valid = torch.zeros(H * W, dtype=torch.bool)
            valid[warp._indices[0]] = True
            valid = valid.reshape(H, W)
        return cls(valid, fwhm=fwhm, A=A, model_mask=model_mask, ksize=ksize, kernel=kernel, dtype=dtype)

    def _fftconv(self, z, conj):
        r = self.r
        zp = F.pad(z, (r, r, r, r))
        Z = torch.fft.rfft2(zp) * (self.Kf.conj() if conj else self.Kf)
        return torch.fft.irfft2(Z, s=self.fshape)[..., r:r + self.ny, r:r + self.nx]

    def _G(self, z):                                    # z (n, 1, ny, nx)
        if self.kernel is not None:
            return self._fftconv(z, conj=False)
        z = F.conv2d(F.pad(z, (self.pad, self.pad, 0, 0)), self.gx)      # separable Gaussian
        return F.conv2d(F.pad(z, (0, 0, self.pad, self.pad)), self.gy)

    def _Gt(self, z):                                   # transpose of _G (the Gaussian is symmetric)
        return self._fftconv(z, conj=True) if self.kernel is not None else self._G(z)

    def _P(self, y):                                    # y (n, n_pix) -> (n, n_pix)
        z = y.view(-1, 1, self.ny, self.nx)
        return (self._G(self.m * z) / self.Gm).view(y.shape[0], -1)

    def _Pt(self, y):
        z = y.view(-1, 1, self.ny, self.nx)
        return (self.m * self._Gt(z / self.Gm)).view(y.shape[0], -1)

    def _K(self, x):                                    # x (n, n_model) -> (n, n_pix)
        y = (self.A @ x.t()).t() if self.A is not None else x
        return self._P(y) if self.use_psf else y

    def _Kt(self, y):                                   # y (n, n_pix) -> (n, n_model)
        y = self._Pt(y) if self.use_psf else y * self.m.view(1, -1)
        return (self.At @ y.t()).t() if self.A is not None else y

    def residual(self, X: torch.Tensor) -> torch.Tensor:
        """L X = X - D^-1 K^T K X for X (n, n_model), without gradient."""
        with torch.no_grad():
            return X - self._Kt(self._K(X)) / self.KtK1

    def __call__(self, maps: Sequence[torch.Tensor], weights: Optional[Sequence[float]] = None) -> torch.Tensor:
        """maps: n tensors on the model grid (n_model elements each); weights: n floats (default 1)."""
        X = torch.stack([q.reshape(-1) for q in maps])                   # (n, n_model)
        if X.shape[1] != self.n_model:
            raise ValueError(f"maps have {X.shape[1]} elements, the model {self.n_model}")
        w = torch.ones(X.shape[0], dtype=X.dtype) if weights is None else torch.as_tensor(weights, dtype=X.dtype)
        return _Quadratic.apply(X, w, self)


class _Quadratic(torch.autograd.Function):
    """sum_q w_q mean_mask (L x_q)^2 with L = I - D^-1 K^T K; backward by hand: dX = L^T G, G = 2 w R mask / N."""

    @staticmethod
    def forward(ctx, X, w, pen):
        with torch.no_grad():
            R = X - pen._Kt(pen._K(X)) / pen.KtK1
            m = pen.mask.to(X.dtype)
            n = m.sum()
            val = (w * ((R * R) * m).sum(1) / n).sum()
            ctx.save_for_backward(R, w)
            ctx.pen = pen
            ctx.m = m
            ctx.n = n
        return val

    @staticmethod
    def backward(ctx, gout):
        R, w = ctx.saved_tensors
        pen = ctx.pen
        with torch.no_grad():
            G = (2.0 * gout / ctx.n) * w[:, None] * R * ctx.m
            dX = G - pen._Kt(pen._K(G / pen.KtK1))                       # L^T G = G - K^T K D^-1 G
        return dX, None, None
