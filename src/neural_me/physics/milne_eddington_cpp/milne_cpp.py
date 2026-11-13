import warnings

import torch
from . import MilneEddington as ME
import numpy as np
from einops import rearrange

# Module-level storage for response functions from last C++ forward pass.
# Used by the LM solver to access the full Jacobian without re-computation.
_last_response_functions = None



# Lines hard-coded in pyMilne's MilneEddington.pyLines (vendored, not tracked).
CPP_LINES = (6301, 6302, 6173)
# pyMilne clips the line-of-sight velocity to this value [km/s].
VLOS_LIMIT = 13.0
# log gf of pyMilne's table (6302 from Socas-Navarro 2011, the rest from VALD).
PYMILNE_LOG_GF = {6301: -0.718, 6302: -1.160, 6173: -2.880}


class _LineasMilneEddington(ME.MilneEddington):
    """pyMilne solver whose line (centre, J, g) is the one given, not pyMilne's table.

    The torch backend takes the line from LINEAS; building pyMilne's line from
    the same values keeps the two backends on the same atomic data.
    """

    def __init__(self, regions, line, **kwargs):
        self._line = line
        super().__init__(regions, [int(line["cw"])], **kwargs)

    def _initLine(self, label, anomalous, dw, precision):
        d = self._line
        cls = ME.pyMilne.pyLines if precision == "float64" else ME.pyMilne.pyLinesf
        return cls(j1=d["jl"], j2=d["ju"], g1=d["gl"], g2=d["gu"], cw=d["cw"],
                   gf=10.0 ** PYMILNE_LOG_GF[int(d["cw"])], anomalous=anomalous, dw=dw)

def _model_array(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, np_dtype):
    """(1, n_pixels, 9) model in pyMilne's order [B, gamma, xi, vlos, ddop, eta0, a, S0, S1]; see MILNE_function."""
    eps = 1e-6
    model = np.zeros((1, B.shape[0], 9), dtype=np_dtype)
    model[0, :, 0] = B.detach().cpu().numpy()[:, 0]
    model[0, :, 1] = np.clip(gamma.detach().cpu().numpy()[:, 0], eps, np.pi - eps)
    np_xi = xi.detach().cpu().numpy()[:, 0]
    np_xi[np_xi < 0] += np.pi
    np_xi[np_xi > np.pi] -= np.pi
    model[0, :, 2] = np_xi
    model[0, :, 3] = vlos.detach().cpu().numpy()[:, 0]
    if np.abs(model[0, :, 3]).max() > VLOS_LIMIT:
        warnings.warn(
            f"cpp backend: |vlos| > {VLOS_LIMIT} km/s is clipped by pyMilne; the "
            "profiles and gradients of those pixels saturate. Use backend='torch' "
            "for faster flows.",
            stacklevel=3,
        )
    model[0, :, 4] = ddop.detach().cpu().numpy()[:, 0]
    model[0, :, 5] = eta0.detach().cpu().numpy()[:, 0] / np.sqrt(np.pi)
    model[0, :, 6] = np.abs(a.detach().cpu().numpy()[:, 0] + eps)
    model[0, :, 7] = S_0.detach().cpu().numpy()[:, 0]
    model[0, :, 8] = S_1.detach().cpu().numpy()[:, 0]
    return model


def _grads_in_signature_order(g, dtype):
    """(n_pixels, 9) gradient in pyMilne's order -> the 9 gradients of forward(B, gamma, xi, vlos, eta0, a, ddop, S0, S1)."""
    g = g.to(dtype)
    return (g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4], g[:, 5:6] / np.sqrt(np.pi), g[:, 6:7], g[:, 4:5],
            g[:, 7:8], g[:, 8:9])


class MILNE_VJP(torch.autograd.Function):
    """
    C++ Milne-Eddington synthesis whose backward is pyMilne's vector-Jacobian
    product: the forward computes the Stokes only, the backward the 9 parameter
    gradients per pixel, without the (n_pixels, 9, 4 n_lambda) response
    functions. Same model array, same parameter order and eta0 scaling as
    MILNE_function; the Stokes are bit-identical to it.

    Returns I, Q, U, V, each (n_pixels, n_lambda) contiguous (blocks of one
    (4, n_pixels, n_lambda) buffer), written by pyMilne in the tensors' dtype.
    """

    @staticmethod
    def forward(ctx, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, milne_instance, mu):
        np_dtype = np.float32 if milne_instance._get_dtype() == "float32" else np.float64
        model = _model_array(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, np_dtype)
        if isinstance(mu, torch.Tensor):
            m = mu.detach().cpu().reshape(-1)
            if not bool((m == m[0]).all()):          # pyMilne takes one mu (synthesize_RF as well)
                raise ValueError("cpp backend: mu must be one value for all the pixels")
            mu = float(m[0])
        mu = float(mu)
        n, nw = B.shape[0], milne_instance.n_wavelength
        buf = torch.empty((4, n, nw), dtype=B.dtype)
        out = buf.unbind(0)
        milne_instance.Me.stokes_io(model, [o.data_ptr() for o in out], nw, 1, int(B.dtype == torch.float32), mu=mu)
        ctx.model, ctx.mu, ctx.milne, ctx.device = model, mu, milne_instance, B.device
        return tuple(o.to(B.device) for o in out)

    @staticmethod
    def backward(ctx, gI, gQ, gU, gV):
        gs = [g.detach().cpu() if g is not None else None for g in (gI, gQ, gU, gV)]
        dtype = next(g.dtype for g in gs if g is not None)
        # pyMilne reads the four gradients in place when they share dtype and strides
        gs = [None if g is None else g.to(dtype).contiguous() for g in gs]
        nw = ctx.milne.n_wavelength
        go = ctx.milne.Me.vjp_io(ctx.model, [0 if g is None else g.data_ptr() for g in gs], nw, 1,
                                 int(dtype == torch.float32), mu=ctx.mu)
        go = torch.as_tensor(go.reshape(-1, 9), device=ctx.device)
        return (*_grads_in_signature_order(go, dtype), None, None)


class MILNE_function(torch.autograd.Function):
    """
    PyTorch autograd function wrapping the C++ Milne-Eddington synthesis.

    Parameter Ordering Note:
        The C++ backend expects model parameters in this order:
        [B, gamma, xi, vlos, ddop, eta0, a, S0, S1]

        The forward() signature uses a different order for PyTorch API compatibility:
        (B, gamma, xi, vlos, eta0, a, ddop, S0, S1)

        The model array construction (line ~34) and gradient extraction (backward)
        are carefully ordered to maintain consistency. This is intentional.

    Normalization Notes:
        - eta0 is scaled by 1/sqrt(pi) in forward() to match PyTorch profile normalization
        - grad_eta0 is also scaled by 1/sqrt(pi) in backward() (chain rule)

    C++ Backend Limitations:
        - vlos is hardcoded clipped to [-13, +13] km/s in the C++ library
        - gamma at 0 or π: response function gradients become zero, causing optimization to get stuck.
          We add a small epsilon to avoid these exact boundary values.
          PyTorch autodiff does not have this limitation.
        - Response functions may differ from PyTorch autodiff at boundary cases
    """

    @staticmethod
    def forward(ctx, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, milne_instance, mu):
        # print('==> forward')
        # Move tensors to CPU before numpy conversion (C++ backend is CPU-only)
        # Get precision from the milne instance
        precision = milne_instance._get_dtype()
        np_dtype = np.float32 if precision == "float32" else np.float64

        # Pre-allocate model array to avoid redundant copies/transpositions
        # Shape: (1, n_pixels, 9) to match MilneEddington.py expectations
        # Clamp gamma to avoid exactly 0 or pi where C++ gradients become zero
        eps = 1e-6
        n_pixels = B.shape[0]
        model = np.zeros((1, n_pixels, 9), dtype=np_dtype)

        # Fill model array efficiently
        model[0, :, 0] = B.detach().cpu().numpy()[:, 0]
        model[0, :, 1] = np.clip(
            gamma.detach().cpu().numpy()[:, 0], eps, np.pi - eps
        )  # Inc

        # Azimuth wrapping
        np_xi = xi.detach().cpu().numpy()[:, 0]
        np_xi[np_xi < 0] += np.pi
        np_xi[np_xi > np.pi] -= np.pi
        model[0, :, 2] = np_xi

        model[0, :, 3] = vlos.detach().cpu().numpy()[:, 0]
        if np.abs(model[0, :, 3]).max() > VLOS_LIMIT:
            warnings.warn(
                f"cpp backend: |vlos| > {VLOS_LIMIT} km/s is clipped by pyMilne; the "
                "profiles and gradients of those pixels saturate. Use backend='torch' "
                "for faster flows.",
                stacklevel=2,
            )
        model[0, :, 4] = ddop.detach().cpu().numpy()[:, 0]

        # Normalize eta0 and apply damping positivity
        model[0, :, 5] = eta0.detach().cpu().numpy()[:, 0] / np.sqrt(np.pi)
        model[0, :, 6] = np.abs(a.detach().cpu().numpy()[:, 0] + eps)

        model[0, :, 7] = S_0.detach().cpu().numpy()[:, 0]
        model[0, :, 8] = S_1.detach().cpu().numpy()[:, 0]

        if isinstance(mu, torch.Tensor):
            np_mu = mu.detach().cpu().numpy().astype(np_dtype)
        else:
            np_mu = float(mu)

        nbatch = n_pixels

        # Use synthesize_rf which returns (stokes, rf)
        # Note: synthesize_rf in MilneEddington.py uses _get_dtype() to ensure arrays are contiguous and correct type
        stokes, dstokes = milne_instance.synthesize_rf(model, mu=np_mu)
        stokes = rearrange(stokes, "nx ny ns nw -> (ny nx) (ns nw)")

        # Zero-copy reshape: (nx, ny, 9, 4, nw) -> (Pixels, 9, TotalWvl)
        # This is a view in NumPy, avoiding a 500MB+ copy for 256x256 grid.
        dstokes = dstokes.reshape(n_pixels, 9, -1)

        ctx.dstokes = dstokes
        ctx.nbatch = nbatch
        ctx.nl4 = stokes.shape[0] * stokes.shape[1]
        ctx.input_device = B.device  # Store original device for backward

        # Store RF globally for LM solver access
        global _last_response_functions
        _last_response_functions = dstokes  # (n_pixels, 9, 4*n_wav) numpy array

        # global RF_global
        # RF_global = ctx.dstokes

        return torch.as_tensor(stokes, dtype=B.dtype, device=B.device)

        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # print('==> backward')

        # Get the device from context (stored during forward)
        device = ctx.input_device

        # Convert response functions to tensors on the correct device (Zero-copy if on CPU)
        # ctx.dstokes layout: (Pixels, 9, TotalWvl)
        # grad_output layout: (Pixels, TotalWvl)
        dstokes_t = torch.as_tensor(ctx.dstokes, device=device, dtype=grad_output.dtype)

        # Compute all 9 gradients at once using batched matrix multiplication
        # (Pixels, 9, TotalWvl) @ (Pixels, TotalWvl, 1) -> (Pixels, 9, 1)
        grad_all = torch.bmm(dstokes_t, grad_output.unsqueeze(-1)).squeeze(-1)

        # Unpack gradients in forward signature order:
        # CPP order: B[0], gamma[1], xi[2], vlos[3], ddop[4], eta0[5], a[6], S0[7], S1[8]
        # Signature: B, gamma, xi, vlos, eta0, a, ddop, S0, S1
        grad_B = grad_all[:, 0][:, None]
        grad_gamma = grad_all[:, 1][:, None]
        grad_xi = grad_all[:, 2][:, None]
        grad_vlos = grad_all[:, 3][:, None]

        # eta0 gradient needs sqrt(pi) scaling (it was 6th in CPP: index 5)
        grad_eta0 = grad_all[:, 5][:, None] / np.sqrt(np.pi)

        grad_a = grad_all[:, 6][:, None]
        grad_ddop = grad_all[:, 4][:, None]
        grad_S0 = grad_all[:, 7][:, None]
        grad_S1 = grad_all[:, 8][:, None]

        return (
            grad_B,
            grad_gamma,
            grad_xi,
            grad_vlos,
            grad_eta0,
            grad_a,
            grad_ddop,
            grad_S0,
            grad_S1,
            None,
            None,
        )


def spectral_psf_matrix(lambda_axis, spectralPSF, step=None):
    """
    Fine wavelength grid and the matrix that applies the spectral PSF.

    lambda_axis: observed wavelengths (A, relative to the line centre);
    spectralPSF: (n, 2) array, offset (mA) and transmission; step: grid step (mA), the profile's sampling by default.

    The grid is regular and extends one profile half-width beyond the observed range on both sides, so that every
    observed point sees the whole profile. Row i of M (n_observed, n_fine) holds the profile at the exact offset of
    every fine point from observed point i, normalized to unit sum: the observed points need not lie on the grid.
    The profile is applied as pyMilne applies it, a correlation, S_obs(l_i) = sum_x P(x) S(l_i + x); it differs
    from a convolution only for an asymmetric profile.

    Returns the fine grid (A) and M (float64).
    """
    psf = np.asarray(spectralPSF, dtype=np.float64)
    if psf.ndim != 2 or psf.shape[1] != 2:
        raise ValueError("spectralPSF must be an (n, 2) array: offset (mA) and transmission")
    w = np.asarray(lambda_axis, dtype=np.float64)
    dw = (psf[1, 0] - psf[0, 0] if step is None else step) / 1e3
    half = np.abs(psf[[0, -1], 0]).max() / 1e3
    n0 = int(np.ceil(half / dw)) + 1
    nf = int(np.ceil((w.max() - w.min()) / dw)) + 2 * n0 + 1
    wf = w.min() - n0 * dw + dw * np.arange(nf)
    M = np.interp((wf[None, :] - w[:, None]) * 1e3, psf[:, 0], psf[:, 1], left=0.0, right=0.0)
    return wf, M / M.sum(axis=1, keepdims=True)


class _ApplySpectralPSF(torch.autograd.Function):
    """S @ Mt with contiguous operands in both directions. The Stokes from pyMilne and the gradient that comes back
    through a spatial PSF are strided views, and the matrix product on them is about 20 times slower (torch.profiler,
    512 x 512 pixels)."""

    @staticmethod
    def forward(ctx, S, Mt):
        ctx.save_for_backward(Mt)
        return S.contiguous() @ Mt

    @staticmethod
    def backward(ctx, g):
        (Mt,) = ctx.saved_tensors
        return g.contiguous() @ Mt.T.contiguous(), None


class MILNE(torch.nn.Module):
    """
    Adding the possibility of using a spectral PSF.

    With a spectral PSF, pyMilne synthesizes without it on a fine regular grid (spectral_psf_matrix) and a fixed
    matrix applies the profile and samples the observed points. The matrix is the same for every pixel, so the
    gradient keeps pyMilne's fused vector-Jacobian product; pyMilne's own PSF path computes per-pixel response
    functions instead (4.4 and 7.5 times slower for SO/PHI-HRT and Hinode/SP, 512 x 512 pixels, fwd+bwd,
    16 threads, float64).
    spectral_step (mA) sets the fine grid, the profile's sampling by default.

    Stores response functions (RF) from last forward call for LM access.
    """

    def __init__(
        self,
        lambda0,
        JUp,
        JLow,
        gUp,
        gLow,
        lambda_axis,
        mu,
        spectralPSF=None,
        nthreads=None,
        precision="float64",
        spectral_step=None,
    ):
        super(MILNE, self).__init__()

        self.spectralPSF = spectralPSF

        if int(lambda0) not in CPP_LINES:
            raise ValueError(
                f"The cpp backend implements lines {CPP_LINES} only, got {lambda0}; "
                "use backend='torch' for other lines."
            )

        if isinstance(lambda_axis, torch.Tensor):
            lambda_axis = lambda_axis.clone().detach().numpy()

        if spectralPSF is None:
            regions = [[lambda_axis + lambda0, None]]
            self.Mt = None

        else:
            if isinstance(spectralPSF, torch.Tensor):
                spectralPSF = spectralPSF.detach().cpu().numpy()
            fine, M = spectral_psf_matrix(lambda_axis, spectralPSF, spectral_step)
            self.Mt = torch.from_numpy(M.T.copy())          # (n_fine, n_observed)
            print(f"[INFO] Using spectral PSF: {lambda_axis.size} observed -> {fine.size} fine points every "
                  f"{(fine[1] - fine[0]) * 1e3:.2f} mA")
            regions = [[fine + lambda0, None]]

        if nthreads is None:
            nthreads = torch.get_num_threads()
        print("Using num_threads: ", nthreads)
        print("Using precision: ", precision)
        line = {"cw": float(lambda0), "jl": float(JLow), "ju": float(JUp),
                "gl": float(gLow), "gu": float(gUp)}
        self.milne_class = _LineasMilneEddington(
            regions, line, nthreads=nthreads, precision=precision
        )
        self.milne_class.n_wavelength = int(self.milne_class.get_wavelength_array().size)
        self.mu = mu
        self._last_rf = None  # Store RF from last forward pass for LM solver
        # True: forward computes and keeps the full response functions (the LM solver reads
        # them); False: Stokes only, the gradient comes from pyMilne's VJP in backward.
        self.store_rf = False

    def forward(self, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1):
        """(n_pixels, 4 n_lambda) if store_rf, else the tuple (I, Q, U, V)."""
        if not self.store_rf:
            stokes = MILNE_VJP.apply(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, self.milne_class, self.mu)
            if self.Mt is not None:
                Mt = self.Mt.to(dtype=stokes[0].dtype, device=stokes[0].device)
                stokes = tuple(_ApplySpectralPSF.apply(s, Mt) for s in stokes)
            self.stokes = stokes
            return stokes

        self.stokes = MILNE_function.apply(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1, self.milne_class, self.mu)
        if self.Mt is not None:
            # (n_pixels, 4 n_fine) -> (n_pixels, 4 n_observed); the response functions the LM solver reads go
            # through the same matrix
            global _last_response_functions
            n, nf, no = self.stokes.shape[0], self.Mt.shape[0], self.Mt.shape[1]
            Mt = self.Mt.to(dtype=self.stokes.dtype, device=self.stokes.device)
            self.stokes = (self.stokes.reshape(n, 4, nf) @ Mt).reshape(n, 4 * no)
            rf = torch.as_tensor(_last_response_functions).reshape(n, 9, 4, nf)
            _last_response_functions = (rf @ self.Mt.to(rf.dtype)).reshape(n, 9, 4 * no).numpy()

        return self.stokes


if __name__ == "__main__":
    import numpy as np

    lambda0 = 6301.5080
    JUp = 2.0
    JLow = 2.0
    gUp = 1.5
    gLow = 1.833
    lambdaStart = 6300.8
    lambdaStep = 0.03
    nLambda = 50
    lambda_axis = np.linspace(lambdaStart, lambdaStep * nLambda + lambdaStart, nLambda)

    B = torch.tensor([[992.0]], requires_grad=True)
    gamma = torch.tensor([[np.deg2rad(134.0)]], requires_grad=True)
    xi = torch.tensor([[np.deg2rad(145.0)]], requires_grad=True)
    vlos = torch.tensor([[1.0]], requires_grad=True)
    eta0 = torch.tensor([[73.0]], requires_grad=True)
    a = torch.tensor([[0.2 * 10]], requires_grad=True)
    ddop = torch.tensor([[0.02]], requires_grad=True)
    S_0 = torch.tensor([[0.5]], requires_grad=True)
    S_1 = torch.tensor([[0.5]], requires_grad=True)
    mu = torch.tensor([1.0])

    fn = MILNE(lambda0, JUp, JLow, gUp, gLow, lambda_axis, mu)
    stokes = fn(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
    print("stokes.shape", stokes.shape)

    # (B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
    # 0,  1,     2,  3,    5,    6, 4,    7,   8
    loss = torch.mean(stokes**2)
    loss.backward()

    print(RF_global.shape)
    # S_1
    with torch.no_grad():
        S_1new = torch.tensor([[S_1.data]], requires_grad=False)
        eps = S_1new.data * 0.001
        S_1new += eps
        print(S_1, "->", S_1new)
        stokes2 = fn(B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1new)
        rf_num = ((stokes2 - stokes) / eps).detach().numpy()

        loss2 = torch.mean(stokes2**2)

        rf_num2 = ((loss2 - loss) / eps).detach().numpy()

        print("(loss2 - loss)", (loss2 - loss))
        print("new.grad", S_1.grad.detach().numpy())
        print("rf_num2", rf_num2)

    # print(rf_num.shape)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(211)
    # plt.plot(stokes[0,:].detach().numpy())
    plt.plot(stokes[0, :].detach().numpy() - stokes2[0, :].detach().numpy(), ls="--")
    plt.subplot(212)
    plt.plot(rf_num[0, :])
    plt.plot(RF_global[0, :, 8], ls="--")
    plt.show()
