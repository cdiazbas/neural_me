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
        nlinea: int = 4,
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
            nlinea: Index for parameter line configuration.
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

        if atmosphere.grid_shape is None:
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
                # Slice specific channel
                param_map = stack_perm[..., i, :, :]
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

    def compute_loss(
        self,
        atmosphere: Atmosphere,
        indices: Optional[torch.Tensor] = None,
        return_breakdown: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Computes weighted loss (MSE or L1 based on self.loss_type).
        """
        # 1. Forward Pass
        if indices is not None:
            atm_subset = atmosphere[indices]
            I_mod, Q_mod, U_mod, V_mod = self.compute_forward_model(atm_subset)

            obs_I = self.obs.stokes_I[indices]
            obs_Q = self.obs.stokes_Q[indices]
            obs_U = self.obs.stokes_U[indices]
            obs_V = self.obs.stokes_V[indices]
        else:
            I_mod, Q_mod, U_mod, V_mod = self.compute_forward_model(atmosphere)
            obs_I = self.obs.stokes_I
            obs_Q = self.obs.stokes_Q
            obs_U = self.obs.stokes_U
            obs_V = self.obs.stokes_V

        # 2. Masking (Wavelengths)
        mask = self.obs.active_wav_idx

        # 3. Residuals
        res_I = (I_mod - obs_I)[..., mask]
        res_Q = (Q_mod - obs_Q)[..., mask]
        res_U = (U_mod - obs_U)[..., mask]
        res_V = (V_mod - obs_V)[..., mask]

        # 4. Error Calculation per Stokes
        if self.loss_type == "l1":
            err_I = torch.mean(torch.abs(res_I))
            err_Q = torch.mean(torch.abs(res_Q))
            err_U = torch.mean(torch.abs(res_U))
            err_V = torch.mean(torch.abs(res_V))
        else:  # 'mse' default
            err_I = torch.mean(res_I**2)
            err_Q = torch.mean(res_Q**2)
            err_U = torch.mean(res_U**2)
            err_V = torch.mean(res_V**2)

        # 5. Weighted Sum
        loss = (
            self.weights[0] * err_I
            + self.weights[1] * err_Q
            + self.weights[2] * err_U
            + self.weights[3] * err_V
        )

        # 6. Regularization
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.regularization_weights and indices is None:
            # Only compute reg if we are processing the full grid/frame
            # otherwise spatial topology is lost or incomplete
            reg_loss = self._compute_regularization(atmosphere)
            loss += reg_loss

        if return_breakdown:
            return loss, {
                "loss_I": err_I.item(),
                "loss_Q": err_Q.item(),
                "loss_U": err_U.item(),
                "loss_V": err_V.item(),
                "loss_reg": reg_loss.item(),
            }

        return loss


class MultiMEProblem:
    """
    Multi-instrument inversion problem for stereoscopic observations.

    Manages multiple MEProblem instances and computes a combined loss function.
    Works together with StereoscopicParameterModel for coupled inversions.

    Features:
    1. Weighted combination of losses from each instrument
    2. Optional stray light calibration (applied to specified instrument)
    3. Shared convergence tracking
    """

    def __init__(
        self,
        problems: List[MEProblem],
        weights: Optional[List[float]] = None,
        cali_instrument: int = -1,
    ):
        """
        Initialize MultiMEProblem.

        Args:
            problems: List of MEProblem instances, one per instrument
            weights: Weight for each instrument's loss (default: equal weights)
            cali_instrument: Index of instrument to apply stray light calibration (-1 = none)
        """
        self.problems = problems
        self.n_problems = len(problems)

        if weights is None:
            weights = [1.0] * self.n_problems
        self.weights = weights

        self.cali_instrument = cali_instrument
        self.device = problems[0].device

    def compute_loss(
        self,
        atmospheres: List[Atmosphere],
        cali: Optional[torch.Tensor] = None,
        return_breakdown: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute combined loss from all instruments.

        Args:
            atmospheres: List of Atmosphere objects (one per problem)
            cali: Optional calibration factor tensor (for stray light correction)
            return_breakdown: If True, return per-instrument breakdown

        Returns:
            Combined loss (and optional breakdown dict)
        """
        assert len(atmospheres) == self.n_problems, (
            f"Expected {self.n_problems} atmospheres, got {len(atmospheres)}"
        )

        total_loss = torch.tensor(0.0, device=self.device)
        breakdown = {}

        for i, (prob, atm, w) in enumerate(
            zip(self.problems, atmospheres, self.weights)
        ):
            # Apply stray light calibration if applicable
            # This would require modifying the forward model output, not the atmosphere
            # For now, compute loss without calibration (calibration handled externally)

            loss_i, bd_i = prob.compute_loss(atm, return_breakdown=True)
            weighted_loss = w * loss_i
            total_loss += weighted_loss

            breakdown[f"instrument_{i}"] = {
                "loss": loss_i.item(),
                "weighted_loss": weighted_loss.item(),
                **{f"{k}_{i}": v for k, v in bd_i.items()},
            }

        breakdown["total"] = total_loss.item()

        if return_breakdown:
            return total_loss, breakdown
        return total_loss

    def compute_forward_models(
        self, atmospheres: List[Atmosphere]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run forward models for all instruments.

        Returns:
            List of (I, Q, U, V) tuples, one per instrument
        """
        results = []
        for prob, atm in zip(self.problems, atmospheres):
            results.append(prob.compute_forward_model(atm))
        return results

    def apply_calibration(
        self,
        stokes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        cali: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply stray light calibration to Stokes profiles.

        Models stray light as: I_out = (1 - α) * I_in + α * <I_in>

        Args:
            stokes: Tuple of (I, Q, U, V) tensors
            cali: Calibration factor α in [0, 1]

        Returns:
            Calibrated (I, Q, U, V)
        """
        I, Q, U, V = stokes

        # Mean over all pixels (dim=0) for each wavelength
        I_mean = I.mean(dim=0, keepdim=True)
        Q_mean = Q.mean(dim=0, keepdim=True)
        U_mean = U.mean(dim=0, keepdim=True)
        V_mean = V.mean(dim=0, keepdim=True)

        I_cal = (1 - cali) * I + cali * I_mean
        Q_cal = (1 - cali) * Q + cali * Q_mean
        U_cal = (1 - cali) * U + cali * U_mean
        V_cal = (1 - cali) * V + cali * V_mean

        return I_cal, Q_cal, U_cal, V_cal
