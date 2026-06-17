import torch
import torch.nn.functional as F
import numpy as np
import kornia
from kornia.geometry.transform import get_tps_transform, warp_image_tps


# ---------------------------------------------------------------------------
# Level → npoints mapping
#   Level 0 : 5  pts  (4 corners + centre)
#   Level 1 : 9  pts  (3×3 uniform grid)
#   Level 2 : 16 pts  (4×4 uniform grid)
#   Level n : (n+2)² pts  for n ≥ 1
# ---------------------------------------------------------------------------

def _level_to_npoints(level: int) -> int:
    """Convert a refinement level to the corresponding number of control points."""
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    return 5 if level == 0 else (level + 2) ** 2


def _prepare_image(image) -> torch.Tensor:
    """
    Convert any common image format to a (1, C, H, W) float32 tensor.

    Accepted input shapes:
        (H, W)        → single-channel grayscale
        (C, H, W)     → channels-first (e.g. PyTorch convention)
        (H, W, C)     → channels-last (e.g. NumPy/OpenCV convention)
        (1, C, H, W)  → already in batch form
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.astype(np.float32))
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected numpy array or torch.Tensor, got {type(image)}")

    image = image.float()

    if image.dim() == 2:                          # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)   # → (1, 1, H, W)
    elif image.dim() == 3:
        # Heuristic: if the last dim is ≤ 4 and is smaller than the first dim,
        # assume channels-last (H, W, C).
        c_last = image.shape[-1]
        c_first = image.shape[0]
        if c_last <= 4 and c_last < c_first:
            image = image.permute(2, 0, 1)        # (H,W,C) → (C,H,W)
        image = image.unsqueeze(0)                # → (1, C, H, W)
    elif image.dim() == 4:
        pass                                       # (B, C, H, W) — already good
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return image


# ---------------------------------------------------------------------------
# Low-level primitives (kept for direct use)
# ---------------------------------------------------------------------------

def get_points_src(npoints: int = 9) -> torch.Tensor:
    """
    Return a regular grid of TPS source control points in normalised [-1, 1] space.

    Args:
        npoints: Number of control points. Supports 5 and any perfect-square ≥ 4.

    Returns:
        torch.Tensor of shape (1, npoints, 2).
    """
    if npoints == 5:
        return torch.tensor([[[-1.0, -1.0], [1.0, -1.0],
                               [-1.0,  1.0], [1.0,  1.0],
                               [0.0,   0.0]]])
    grid_size = int(round(npoints ** 0.5))
    coords = torch.linspace(-1.0, 1.0, grid_size)
    gy, gx = torch.meshgrid(coords, coords, indexing='ij')
    pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)
    return pts[:, :npoints]


def differentiable_tps_warp(
    image: torch.Tensor,
    points_src: torch.Tensor,
    points_dst: torch.Tensor,
) -> torch.Tensor:
    """
    Warp *image* differentiably using a Thin Plate Spline transformation.

    Args:
        image:      (B, C, H, W) or (H, W) input tensor.
        points_src: (1, N, 2) or (N, 2) source control points.
        points_dst: (1, N, 2) or (N, 2) destination control points.

    Returns:
        Warped tensor of the same shape as *image*.
    """
    is_2d = image.dim() == 2
    if is_2d:
        image = image.unsqueeze(0).unsqueeze(0)

    if points_src.dim() == 2:
        points_src = points_src.unsqueeze(0)
    if points_dst.dim() == 2:
        points_dst = points_dst.unsqueeze(0)

    points_src = points_src.to(dtype=image.dtype, device=image.device)
    points_dst = points_dst.to(dtype=image.dtype, device=image.device)

    kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
    warped = warp_image_tps(image, points_src, kernel_weights, affine_weights)

    return warped.squeeze(0).squeeze(0) if is_2d else warped


def extrapolate_points(
    points_src: torch.Tensor,
    points_dst: torch.Tensor,
    new_points_src: torch.Tensor,
    clamp: float | None = None,
) -> torch.Tensor:
    """
    Extrapolate destination locations for a new set of source control points
    using the TPS mapping fitted on (points_src → points_dst).

    Args:
        points_src:     Original source control points, (1, N, 2) or (N, 2).
        points_dst:     Optimised destination control points, (1, N, 2) or (N, 2).
        new_points_src: New source control points to evaluate, (1, M, 2) or (M, 2).
        clamp:          If set, clamps the maximum Euclidean displacement from
                        new_points_src to prevent wild boundary extrapolation.

    Returns:
        torch.Tensor of shape (1, M, 2) with the extrapolated destinations.
    """
    if points_src.dim() == 2:
        points_src = points_src.unsqueeze(0)
    if points_dst.dim() == 2:
        points_dst = points_dst.unsqueeze(0)
    if new_points_src.dim() == 2:
        new_points_src = new_points_src.unsqueeze(0)

    dtype, device = new_points_src.dtype, new_points_src.device
    points_src = points_src.to(dtype=dtype, device=device)
    points_dst = points_dst.to(dtype=dtype, device=device)

    B, N, _ = points_src.shape
    M = new_points_src.shape[1]

    kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)

    # Biharmonic RBF: U(r²) = r² · log(r²)
    diff = new_points_src.unsqueeze(2) - points_src.unsqueeze(1)  # (B, M, N, 2)
    dist_sq = torch.sum(diff ** 2, dim=-1)                        # (B, M, N)
    U = dist_sq * torch.log(dist_sq + 1e-12)

    kernel_term = torch.bmm(U, kernel_weights)

    ones = torch.ones(B, M, 1, dtype=dtype, device=device)
    aug  = torch.cat([ones, new_points_src], dim=-1)              # (B, M, 3)
    affine_term = torch.bmm(aug, affine_weights)

    new_points_dst = kernel_term + affine_term

    if clamp is not None:
        with torch.no_grad():
            disp = new_points_dst - new_points_src
            dist = torch.norm(disp, dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(dist, max=clamp) / dist
            new_points_dst = new_points_src + disp * scale

    return new_points_dst


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def align_images(
    image_src,
    image_tgt,
    levels = [1],
    clamp = None,
    alpha = 0.0,
    weights = None,
    nprints: int = 10,
    niter: int = 500,
    lr: float = 1e-3,
    optimizer: str = 'adam',
    verbose: bool = True,
    composition: bool = False,
    only_x: bool = False,
    only_y: bool = False,
    loss_callback = None,
):
    """
    Align *image_src* to *image_tgt* using a progressive multi-stage TPS warp.

    Control point levels:
        Level 0 →  5 pts  (4 corners + centre)
        Level 1 →  9 pts  (3×3 uniform grid)
        Level 2 → 16 pts  (4×4 uniform grid)
        Level n → (n+2)² pts  for n ≥ 1

    Multi-channel: images with C > 1 channels are warped and the MSE loss is
    computed jointly across *all* channels simultaneously.

    Args:
        image_src:   Source image to warp. Accepts np.ndarray or torch.Tensor.
                     Shapes: (H,W), (C,H,W), (H,W,C), or (1,C,H,W).
        image_tgt:   Target image (same format / spatial size as image_src).
        levels:      Ordered list of refinement levels, e.g. [0, 1, 2].
                     Successive stages extrapolate the control grid from the
                     previous stage's optimised solution.
                     Examples:
                       [1]          → single stage, 9-point grid (default)
                       [0, 1]       → 5-pt coarse, then refine to 9-pt
                       [0, 1, 2]    → 5-pt → 9-pt → 16-pt progressive
        clamp:       Per-stage clamp applied at every optimiser iteration AND
                     at the extrapolation initialisation between stages.
                     Accepts:
                       None              → no clamping at any stage
                       float             → same value broadcast to all stages
                       list of float     → one value per entry in *levels*
                     A tightening schedule (e.g. clamp=[5.0, 2.0, 1.0]) lets early
                     coarse stages explore freely while later fine stages stay
                     bounded to local movements.
                     NOTE: clamp represents the maximum Euclidean distance (displacement)
                     each control point is allowed to move from its source position
                     in the NORMALIZED coordinate space [-1, 1].
                     Because the total width of the normalized image is 2.0 units:
                       clamp_normalized = pixel_displacement / (image_width / 2)
                     Examples for a 512x512 image:
                       clamp = 5 / 256 ≈ 0.0195 (restricts to max 5 pixels displacement)
                       clamp = 10 / 256 ≈ 0.0390 (restricts to max 10 pixels displacement)
                       clamp = 50 / 256 ≈ 0.1953 (restricts to max 50 pixels displacement)
                       clamp = 1.0 (restricts to half the image width)
                       clamp = 5.0 (virtually unconstrained, as 5.0 is > entire image size)
        alpha:       Per-stage regularization weight for the total absolute
                     displacement distance of all control points.
                     Accepts:
                       0.0 (default)     → no regularization penalty
                       float             → same value broadcast to all stages
                       list of float     → one value per entry in *levels*
        weights:     Optional list, tuple, np.ndarray, or torch.Tensor of shape (C,)
                     containing channel weights for multi-channel images.
                     Default is None (all ones).
        nprints:     The exact number of progress/status messages to print
                     during the optimization loop (default 10).
        niter (int): Optimisation iterations per stage (default 500).
        lr:          Learning rate for Adam (L-BFGS ignores this).
                     Accepts:
                       float (default 1e-3) → same value broadcast to all stages
                       list of float       → one value per entry in *levels*
                     Reducing the learning rate in later stages (e.g. lr=[1e-3, 2e-4])
                     provides high-fidelity localized fine-tuning.
        optimizer:   'adam' or 'lbfgs'.
        verbose:     Print per-stage loss progress.
        composition: If True, recursively warps the previously warped image at each
                     stage rather than extrapolating the control grid coordinates.
                     Default: False (grid extrapolation).
        only_x:      If True, control points are only allowed to move along the x-axis
                     (horizontal). The y-coordinate of every point is locked to its
                     source position throughout optimisation. Default: False.
        only_y:      If True, control points are only allowed to move along the y-axis
                     (vertical). The x-coordinate of every point is locked to its
                     source position throughout optimisation. Default: False.
        loss_callback: Optional callable invoked as ``loss_callback(iteration, loss_value)``
                     after every Adam step (1-based iteration index, float loss).
                     Ignored for L-BFGS. Default: None.

    Returns:
        warped (torch.Tensor): Warped source image, shape (1, C, H, W).
        points_src (torch.Tensor): Source control points of the final stage,
                                   shape (1, N, 2) in normalised [-1, 1] space.
        points_dst (torch.Tensor): Optimised destination control points of the
                                   final stage, shape (1, N, 2).
    """
    # --- Validate & normalise arguments ----------------------------------------
    if only_x and only_y:
        raise ValueError("only_x and only_y cannot both be True.")

    if isinstance(levels, int):
        levels = [levels]

    # Broadcast clamp to a per-stage list
    if clamp is None:
        clamps = [None] * len(levels)
    elif isinstance(clamp, (int, float)):
        clamps = [float(clamp)] * len(levels)
    else:
        clamps = [float(c) if c is not None else None for c in clamp]
        if len(clamps) != len(levels):
            raise ValueError(
                f"clamp list length ({len(clamps)}) must match "
                f"levels list length ({len(levels)})"
            )

    # Broadcast alpha to a per-stage list
    if alpha is None:
        alphas = [0.0] * len(levels)
    elif isinstance(alpha, (int, float)):
        alphas = [float(alpha)] * len(levels)
    else:
        alphas = [float(a) if a is not None else 0.0 for a in alpha]
        if len(alphas) != len(levels):
            raise ValueError(
                f"alpha list length ({len(alphas)}) must match "
                f"levels list length ({len(levels)})"
            )

    # Broadcast lr to a per-stage list
    if lr is None:
        lrs = [1e-3] * len(levels)
    elif isinstance(lr, (int, float)):
        lrs = [float(lr)] * len(levels)
    else:
        lrs = [float(_lr) for _lr in lr]
        if len(lrs) != len(levels):
            raise ValueError(
                f"lr list length ({len(lrs)}) must match "
                f"levels list length ({len(levels)})"
            )

    # --- Prepare images --------------------------------------------------------
    src = _prepare_image(image_src)   # (1, C, H, W)
    tgt = _prepare_image(image_tgt)   # (1, C, H, W)

    if src.shape[-2:] != tgt.shape[-2:]:
        raise ValueError(
            f"image_src and image_tgt must have the same spatial size, "
            f"got {src.shape[-2:]} vs {tgt.shape[-2:]}"
        )

    C = src.shape[1]

    # Parse and validate channel weights
    if weights is None:
        weights_tensor = torch.ones(C, dtype=src.dtype, device=src.device)
    else:
        if isinstance(weights, (list, tuple, np.ndarray)):
            weights_tensor = torch.tensor(weights, dtype=src.dtype, device=src.device)
        elif isinstance(weights, torch.Tensor):
            weights_tensor = weights.to(dtype=src.dtype, device=src.device)
        else:
            raise TypeError(f"weights must be a list, tuple, np.ndarray, or torch.Tensor, got {type(weights)}")
        
        if weights_tensor.numel() != C:
            raise ValueError(f"Number of weights ({weights_tensor.numel()}) must match number of channels ({C})")
        weights_tensor = weights_tensor.view(C)

    # --- Setup working source image (enables composition) ----------------------
    current_src = src

    # Axis-lock helper: zero displacement in the frozen axis and its gradient
    frozen_axis = 1 if only_x else (0 if only_y else None)

    def _apply_axis_lock(pts_dst, pts_src):
        """Project pts_dst so the frozen axis matches pts_src exactly."""
        if frozen_axis is not None:
            with torch.no_grad():
                pts_dst[..., frozen_axis] = pts_src[..., frozen_axis]

    def _zero_frozen_grad(pts_dst):
        """Zero the gradient component along the frozen axis."""
        if frozen_axis is not None and pts_dst.grad is not None:
            pts_dst.grad[..., frozen_axis] = 0.0

    # --- Inner optimisation stage (captures current_src, tgt, optimizer, lr, niter) ---
    def _run_stage(pts_src, pts_dst_init, stage_label, stage_clamp, stage_alpha, stage_lr):
        pts_dst = pts_dst_init.clone().detach()
        _apply_axis_lock(pts_dst, pts_src)  # ensure init respects constraint
        pts_dst.requires_grad_(True)
        print_every = max(1, niter // nprints)

        if optimizer == 'lbfgs':
            opt = torch.optim.LBFGS([pts_dst], lr=stage_lr, max_iter=40,
                                     line_search_fn='strong_wolfe')
            step_count = [0]
            def closure():
                # Project into L2 displacement ball before every function evaluation
                if stage_clamp is not None:
                    with torch.no_grad():
                        disp = pts_dst - pts_src
                        dist = torch.norm(disp, dim=-1, keepdim=True).clamp(min=1e-8)
                        scale = torch.clamp(dist, max=stage_clamp) / dist
                        pts_dst.copy_(pts_src + disp * scale)
                _apply_axis_lock(pts_dst, pts_src)
                opt.zero_grad()
                warped = differentiable_tps_warp(current_src, pts_src, pts_dst)
                diff_sq = (warped - tgt).pow(2)
                channel_mses = diff_sq.mean(dim=(0, 2, 3))
                mse = (channel_mses * weights_tensor).mean()
                reg = torch.norm(pts_dst - pts_src, dim=-1).mean()
                loss = mse + stage_alpha * reg
                loss.backward()
                _zero_frozen_grad(pts_dst)
                step_count[0] += 1
                if verbose and (step_count[0] % print_every == 0 or step_count[0] == 1):
                    print(f"  [{stage_label}] L-BFGS Eval {step_count[0]:03d}"
                          f" - Loss: {loss.item():.6f} (MSE: {mse.item():.6f}, Reg: {reg.item():.4f})")
                return loss
            for _ in range(max(1, niter // 40)):
                opt.step(closure)
        else:
            opt = torch.optim.Adam([pts_dst], lr=stage_lr)
            for i in range(niter):
                opt.zero_grad()
                warped = differentiable_tps_warp(current_src, pts_src, pts_dst)
                diff_sq = (warped - tgt).pow(2)
                channel_mses = diff_sq.mean(dim=(0, 2, 3))
                mse = (channel_mses * weights_tensor).mean()
                reg = torch.norm(pts_dst - pts_src, dim=-1).mean()
                loss = mse + stage_alpha * reg
                loss.backward()
                _zero_frozen_grad(pts_dst)
                opt.step()
                # Project into L2 displacement ball after every gradient step
                if stage_clamp is not None:
                    with torch.no_grad():
                        disp = pts_dst - pts_src
                        dist = torch.norm(disp, dim=-1, keepdim=True).clamp(min=1e-8)
                        scale = torch.clamp(dist, max=stage_clamp) / dist
                        pts_dst.copy_(pts_src + disp * scale)
                _apply_axis_lock(pts_dst, pts_src)
                if loss_callback is not None:
                    loss_callback(i + 1, loss.item())
                if verbose and ((i + 1) % print_every == 0):
                    print(f"  [{stage_label}] Adam Iter {i+1:03d}"
                          f" - Loss: {loss.item():.6f} (MSE: {mse.item():.6f}, Reg: {reg.item():.4f})")

        return pts_dst

    # --- Progressive multi-stage optimisation ----------------------------------
    points_src = None
    points_dst = None
    stage_srcs = []
    stage_dsts = []

    for stage_idx, (level, stage_clamp, stage_alpha, stage_lr) in enumerate(zip(levels, clamps, alphas, lrs)):
        npts = _level_to_npoints(level)
        pts_src_new = get_points_src(npts).float()
        label = f"Stage {stage_idx+1}/{len(levels)} | Level {level} ({npts}pts)"
        clamp_str = f"  clamp={stage_clamp}" if stage_clamp is not None else ""
        alpha_str = f"  alpha={stage_alpha}" if stage_alpha > 0 else ""
        lr_str = f"  lr={stage_lr}"

        if verbose:
            print(f"\n[align_images] {label}{clamp_str}{alpha_str}{lr_str}")

        if stage_idx == 0 or not composition:
            if stage_idx == 0:
                pts_dst_init = pts_src_new.clone()
            else:
                pts_dst_init = extrapolate_points(
                    points_src, points_dst, pts_src_new, clamp=stage_clamp
                )
        else:
            # Under composition, each stage starts fresh from identity grid
            pts_dst_init = pts_src_new.clone()

        points_dst = _run_stage(pts_src_new, pts_dst_init, label, stage_clamp, stage_alpha, stage_lr)
        points_src = pts_src_new

        if composition:
            stage_srcs.append(points_src)
            stage_dsts.append(points_dst)
            # Compose warp: set working image to the output of this stage
            with torch.no_grad():
                current_src = differentiable_tps_warp(current_src, points_src, points_dst)

    # --- Final warped image ----------------------------------------------------
    if composition:
        warped = current_src
        return warped, stage_srcs, stage_dsts
    else:
        with torch.no_grad():
            warped = differentiable_tps_warp(src, points_src, points_dst)
        return warped, points_src, points_dst
