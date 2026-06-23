import torch
from typing import Optional, List, Tuple, Union, Dict

from neural_me.core.observation import Observation
from neural_me.core.atmospheres.base import (
    Atmosphere,
    DEFAULT_NORMALIZATION,
)
from neural_me.physics.milne_eddington import MilneEddington
from neural_me.regularization.spatial import smoothness_loss


class MEProblem:
    """
    Defines the Milne-Eddington (ME) inversion problem.

    1. Holds the Observation (Data).
    2. Initializes the Forward Model (Physics).
    3. Computes the Loss (Data vs Model).
    """

    def __init__(
        self,
        observation: Observation,
        line: int = 6302,
        weights: List[float] = [1.0, 1.0, 1.0, 1.0],
        loss_type: str = "mse",
        regularization_weights: Optional[Dict[str, float]] = None,
        backend: str = "auto",
        compile: bool = True,
        device: Optional[torch.device] = None,
        precision: str = "float32",
        **kwargs,
    ):
        """
        Args:
            observation: The observed data.
            line: Spectral line — accepts wavelength in Å (e.g. 6302, 6173) or line ID (1-7).
            weights: Loss weights for [Stokes I, Q, U, V].
            loss_type: 'mse' or 'l1'.
            regularization_weights: Dict mapping parameter names (or 'all') to spatial smoothing weights.
                                    Example: {'all': 1e-3} or {'B': 1e-2, 'vlos': 1e-3}.
            backend: 'auto' (default), 'cpp', or 'torch'.
                     - 'auto': Prefers 'cpp' if available, falls back to 'torch'.
            compile: If True and backend='torch', use torch.compile for speedup.
                     Recommended for large batches (>10k pixels). Skip for small cases.
            device: Torch device for computation.
            precision: 'float32' (default) or 'float64'.
            **kwargs: Extra arguments for backend.
                - checkpoint_gradients (bool): If True, use gradient checkpointing (backend='torch' only).
        """
        self.obs = observation
        self.loss_type = loss_type.lower()
        self.regularization_weights = regularization_weights

        # Resolve line: accept wavelength in Å (e.g. 6302, 6173) or line ID (1-7)
        from neural_me.physics.lineas import get_lineas, get_line_by_wavelength
        known_ids = set(get_lineas().keys())
        if line not in known_ids:
            # Treat as wavelength — find nearest line
            line_info = get_line_by_wavelength(line)
            print(f"MEProblem: wavelength {line} → Line ID={line_info.line_id} ({line_info.element} {line_info.cw:.3f} Å)")
            nlinea = line_info.line_id
        else:
            nlinea = line

        self.nlinea = nlinea  # stored for downstream use (e.g. WFA init)

        # Backend Auto-Detection
        if backend == "auto":
            from neural_me.physics.milne_eddington import get_available_backends

            avail = get_available_backends()
            if avail.get("cpp", False):
                backend = "cpp"
                print("MEProblem: Auto-selected backend='cpp' (High Performance)")
            else:
                backend = "torch"
                print(
                    "MEProblem: Auto-selected backend='torch' (C++ extension not found)"
                )

        self.backend = backend

        # Determine device - C++ backend requires CPU
        if backend == "cpp":
            # Handle device as string or torch.device
            is_gpu = False
            if device is not None:
                device_str = str(device) if isinstance(device, torch.device) else device
                is_gpu = "cuda" in device_str.lower()

            if is_gpu:
                import warnings

                warnings.warn(
                    f"C++ backend does not support GPU. Forcing device to CPU "
                    f"(requested: {device})"
                )
            self.device = torch.device("cpu")
        else:
            # Convert string to torch.device if needed
            if device is None:
                self.device = observation.device
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device

        # Ensure obs is on device
        if self.obs.device != self.device:
            self.obs = self.obs.to(self.device)

        # Initialize Physics Model with chosen backend
        # We pass precision only if backend is cpp, but kwargs can handle it generally
        kwargs.pop("nlinea", None)
        kwargs.pop("line", None)
        self.model = MilneEddington(
            nlinea=nlinea,
            backend=backend,
            compile=compile,
            precision=precision,
            **kwargs,
        )
        self.model.to(self.device)

        # Parameter normalization weights - use centralized constants
        self.normalization = DEFAULT_NORMALIZATION.copy()

        # Loss Weights
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

    def _compute_regularization(self, atmosphere: Atmosphere) -> torch.Tensor:
        """
        Computes spatial regularization loss.
        """
        if not self.regularization_weights:
            return torch.tensor(0.0, device=self.device)

        # We need the full grid for spatial regularization
        # If atmosphere is a subset (indices passed to compute_loss), spatial reg might be invalid
        # BUT PixelSolver passes the FULL atmosphere usually.
        # If compute_loss received indices, we probably shouldn't compute spatial reg on the subset
        # unless it's a contiguous patch.
        # For now, assume atmosphere has grid_shape populated if we want spatial reg.

        if len(atmosphere.grid_shape) < 2:
            return torch.tensor(0.0, device=self.device)

        reg_loss = torch.tensor(0.0, device=self.device)

        # Parameters to regularize
        # Atmosphere has 9 params.
        # We can key them by name.
        param_names = ["B", "gamma", "xi", "vlos", "eta0", "a", "ddop", "S0", "S1"]

        # Get stack: (Ny, Nx, 9)
        stack = atmosphere.to_stack()
        if stack.dim() != 3:  # Expect 2D map + params dim
            # Possibly (Batch, Ny, Nx, 9) ?
            if stack.dim() == 4:
                pass  # OK
            else:
                return torch.tensor(0.0, device=self.device)

        # Reshape to (C, Ny, Nx) or (Batch, C, Ny, Nx) for smoothness_loss
        # stack is (..., Ny, Nx, 9) -> permute to (..., 9, Ny, Nx)
        stack_perm = (
            stack.permute(0, 3, 1, 2) if stack.dim() == 4 else stack.permute(2, 0, 1)
        )

        # Global weight
        if "all" in self.regularization_weights:
            w = self.regularization_weights["all"]
            reg_loss += w * smoothness_loss(stack_perm, penalty=self.loss_type)

        # Specific weights
        for i, name in enumerate(param_names):
            if name in self.regularization_weights:
                w = self.regularization_weights[name]
                # Correctly slice the i-th parameter channel for 3D (9,Ny,Nx)
                # or 4D (Batch,9,Ny,Nx) tensor
                if stack_perm.dim() == 3:
                    param_map = stack_perm[i, :, :]
                else:
                    param_map = stack_perm[:, i, :, :]
                reg_loss += w * smoothness_loss(param_map, penalty=self.loss_type)

        return reg_loss

    def compute_forward_model(
        self, atmosphere: Atmosphere
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the forward physics model on the given atmosphere.

        Returns: (I, Q, U, V) tensors of shape (..., n_lambda)
        """
        # Get physical parameters stack [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]
        params = atmosphere.to_stack()  # Shape (..., 9)

        # Flatten parameters to (Batch, 9) for MilneEddington model
        # This handles both (Ny, Nx, 9) shaped atmospheres and (N_pixels, 9) flat/sliced ones
        params = params.reshape(-1, 9)

        # Broadcast wavelengths to match batch size
        batch_size = params.shape[0]
        wl = self.obs.wavelengths.unsqueeze(0).expand(batch_size, -1)

        I, Q, U, V = self.model(params, wl)

        return I, Q, U, V

    def compute_loss_from_syn(
        self,
        syn: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes weighted loss (MSE or L1 based on self.loss_type) from synthetic Stokes profiles.
        """
        I_mod, Q_mod, U_mod, V_mod = syn
        if indices is not None:
            obs_I = self.obs.stokes_I[indices]
            obs_Q = self.obs.stokes_Q[indices]
            obs_U = self.obs.stokes_U[indices]
            obs_V = self.obs.stokes_V[indices]
        else:
            obs_I = self.obs.stokes_I
            obs_Q = self.obs.stokes_Q
            obs_U = self.obs.stokes_U
            obs_V = self.obs.stokes_V

        mask = self.obs.active_wav_idx
        res_I = (I_mod - obs_I)[..., mask]
        res_Q = (Q_mod - obs_Q)[..., mask]
        res_U = (U_mod - obs_U)[..., mask]
        res_V = (V_mod - obs_V)[..., mask]

        if self.loss_type == "l1":
            err_I = torch.mean(torch.abs(res_I))
            err_Q = torch.mean(torch.abs(res_Q))
            err_U = torch.mean(torch.abs(res_U))
            err_V = torch.mean(torch.abs(res_V))
        else:
            err_I = torch.mean(res_I**2)
            err_Q = torch.mean(res_Q**2)
            err_U = torch.mean(res_U**2)
            err_V = torch.mean(res_V**2)

        return (
            self.weights[0] * err_I
            + self.weights[1] * err_Q
            + self.weights[2] * err_U
            + self.weights[3] * err_V
        )

    def compute_loss(
        self,
        atmosphere: Atmosphere,
        indices: Optional[torch.Tensor] = None,
        return_breakdown: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Computes weighted loss (MSE or L1 based on self.loss_type).
        """
        if indices is not None:
            atm_subset = atmosphere[indices]
            syn = self.compute_forward_model(atm_subset)
        else:
            syn = self.compute_forward_model(atmosphere)

        loss = self.compute_loss_from_syn(syn, indices=indices)

        # Regularization
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.regularization_weights and indices is None:
            reg_loss = self._compute_regularization(atmosphere)
            loss += reg_loss

        if return_breakdown:
            I_mod, Q_mod, U_mod, V_mod = syn
            if indices is not None:
                obs_I, obs_Q, obs_U, obs_V = self.obs.stokes_I[indices], self.obs.stokes_Q[indices], self.obs.stokes_U[indices], self.obs.stokes_V[indices]
            else:
                obs_I, obs_Q, obs_U, obs_V = self.obs.stokes_I, self.obs.stokes_Q, self.obs.stokes_U, self.obs.stokes_V
            mask = self.obs.active_wav_idx
            res_I = (I_mod - obs_I)[..., mask]
            res_Q = (Q_mod - obs_Q)[..., mask]
            res_U = (U_mod - obs_U)[..., mask]
            res_V = (V_mod - obs_V)[..., mask]

            if self.loss_type == "l1":
                err_I = torch.mean(torch.abs(res_I))
                err_Q = torch.mean(torch.abs(res_Q))
                err_U = torch.mean(torch.abs(res_U))
                err_V = torch.mean(torch.abs(res_V))
            else:
                err_I = torch.mean(res_I**2)
                err_Q = torch.mean(res_Q**2)
                err_U = torch.mean(res_U**2)
                err_V = torch.mean(res_V**2)

            return loss, {
                "loss_I": err_I.item(),
                "loss_Q": err_Q.item(),
                "loss_U": err_U.item(),
                "loss_V": err_V.item(),
                "loss_reg": reg_loss.item(),
            }

        return loss
