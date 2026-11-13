"""
C++ implementation wrapper for Milne-Eddington synthesis.

Uses the compiled C++ extension via milne_cpp.py interface.
"""

import torch
import torch.nn as nn


def is_available():
    """
    Check if C++ backend is available.
    """
    try:
        return True
    except ImportError:
        return False


class MilneEddingtonCPP(nn.Module):
    """
    C++ Milne-Eddington implementation with PyTorch interface.
    """

    def __init__(
        self,
        nlinea=None,
        lambda_axis=None,
        mu=1.0,
        spectralPSF=None,
        nthreads=None,
        lambda0=None,
        JUp=None,
        JLow=None,
        gUp=None,
        gLow=None,
        precision="float64",
        spectral_step=None,
        **kwargs,
    ):
        super().__init__()

        # Support both nlinea (legacy/file) and explicit parameters
        if nlinea is not None:
            # Get line parameters from LINEAS.txt
            from ..zeeman import paramLine

            param_info = paramLine(nlinea)
            # param_info format: [[dpi, dsr, dsb], [spi, ssr, ssb], [ju, lu, su, jl, ll, sl, elem, l0, gu, gl, gef]]
            ju, lu, su, jl, ll, sl, elem, l0, gu, gl, gef = param_info[2]

            self.nlinea = nlinea
            self.lambda0 = l0
            self.JUp = ju
            self.JLow = jl
            self.gUp = gu
            self.gLow = gl
        elif lambda0 is not None:
            # Explicit initialization
            self.nlinea = None
            self.lambda0 = lambda0
            self.JUp = JUp
            self.JLow = JLow
            self.gUp = gUp
            self.gLow = gLow
        else:
            raise ValueError("C++ backend requires nlinea OR explicit line parameters")

        self.mu = mu
        self.spectralPSF = spectralPSF
        self.spectral_step = spectral_step
        self.nthreads = nthreads
        self.precision = precision
        # Keep the full response functions of each forward (for the LM solver); off by default:
        # the gradient then comes from pyMilne's vector-Jacobian product, no RF array.
        self.store_rf = False

        # Initialize C++ model if lambda_axis provided
        if lambda_axis is not None:
            self._init_cpp_model(lambda_axis)
        else:
            self.cpp_model = None

    def _init_cpp_model(self, lambda_axis):
        """
        Initialize C++ model with wavelength axis.
        """
        from .milne_cpp import MILNE

        # Ensure lambda_axis is float64 for C++ compatibility if needed,
        # but milne_cpp generic wrapper handles data types usually.
        # However, verifying inputs is good.

        self.cpp_model = MILNE(
            self.lambda0,
            self.JUp,
            self.JLow,
            self.gUp,
            self.gLow,
            lambda_axis,
            self.mu,
            spectralPSF=self.spectralPSF,
            nthreads=self.nthreads,
            precision=self.precision,
            spectral_step=self.spectral_step,
        )

    def forward(
        self, params, wl=None, nlinea=None
    ):  # wl is optional if already initialized
        """
        Forward pass for Milne-Eddington synthesis using C++ backend.

        Args:
            params: (batch, 9) - [B(G), gamma, xi, vlos, eta0, a, ddop, S_0, S_1]
            wl: (batch, n_lambda) or (n_lambda,) - wavelength axis (optional if initialized)
            nlinea: optional line ID (not used after initialization)

        Returns:
            (I, Q, U, V) tuple of tensors, each (batch, n_lambda)
        """
        # If wl is provided and model not initialized (or needs re-init?), handle it.
        # But usually we initializing once.
        # For compatibility with forward(params, wl), we accept wl.

        if self.cpp_model is None:
            if wl is None:
                raise ValueError("Wavelength axis required for first initialization")

            if wl.dim() == 1:
                lambda_axis = wl
            else:
                lambda_axis = wl[0]

            self._init_cpp_model(lambda_axis)

        # Extract parameters (C++ expects individual tensors)
        # Assuming params is (Batch, 9)

        # Ensure inputs are on CPU and correct type if needed (wrapper handles torch tensors)

        B = params[:, 0][:, None]  # Already in G
        gamma = params[:, 1][:, None]
        xi = params[:, 2][:, None]
        vlos = params[:, 3][:, None]
        eta0 = params[:, 4][:, None]
        a = params[:, 5][:, None]
        ddop = params[:, 6][:, None]
        S_0 = params[:, 7][:, None]
        S_1 = params[:, 8][:, None]

        self.cpp_model.store_rf = self.store_rf
        stokes = self.cpp_model(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
        if isinstance(stokes, tuple):  # VJP path: already (I, Q, U, V)
            return stokes

        # store_rf path: concatenated IQUV

        # Split into I, Q, U, V to match PyTorch backend interface
        n_lambda = stokes.shape[1] // 4
        I = stokes[:, 0 * n_lambda : 1 * n_lambda]
        Q = stokes[:, 1 * n_lambda : 2 * n_lambda]
        U = stokes[:, 2 * n_lambda : 3 * n_lambda]
        V = stokes[:, 3 * n_lambda : 4 * n_lambda]

        return I, Q, U, V

    def get_last_response_functions(self):
        """
        Return the response functions (Jacobian) from the last forward pass.

        The C++ backend computes the full RF during forward(). This method
        returns them as a torch tensor for use by the LM solver.

        Returns:
            torch.Tensor: (n_pixels, 9, 4*n_wav) response functions.
                          CPP parameter order: [B, gamma, xi, vlos, ddop, eta0, a, S0, S1]
            None: if no forward pass has been done yet.
        """
        from .milne_cpp import _last_response_functions

        if not self.store_rf:
            raise RuntimeError("cpp backend: set model.store_rf = True to keep the response functions")
        if _last_response_functions is None:
            return None
        return torch.as_tensor(_last_response_functions)
