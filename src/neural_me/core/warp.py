import torch
import torch.nn.functional as F
import numpy as np
import kornia
from kornia.geometry.transform import get_tps_transform, warp_image_tps, warp_points_tps


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


def plot_grid(
    pts_src,
    pts_dst,
    ax=None,
    image_shape=None,
    original_grid: bool = True,
    warped_grid: bool = True,
    target_grid: bool | None = None,
    grid_mode: str = 'control',
    vis_n: int = 20,
    src_color: str = 'dodgerblue',
    dst_color: str = 'tomato',
    alpha: float = 0.8,
    linewidth: float = 1.2,
    markersize: float = 4,
):
    """
    Overlay TPS deformation grids on an existing matplotlib axes.

    By default (``grid_mode='control'``), this draws the *actual control-point
    lattice* (dot-connected real grid), not a dense synthetic mesh.

    Works with both composition (list) and non-composition (tensor) outputs
    from :func:`align_images`.

    Args:
        pts_src:      Source control points – tensor ``(1,N,2)`` or list.
        pts_dst:      Destination control points – tensor ``(1,N,2)`` or list.
        ax:           Matplotlib axes to draw on. Defaults to ``plt.gca()``.
        image_shape:  ``(H, W)`` of the displayed image used to convert from
                      normalised ``[-1, 1]`` to pixel coordinates.
                      If ``None``, draws in normalised space.
        original_grid: Draw the original (un-warped) regular grid.
        warped_grid:   Draw the warped (deformed) grid.
        target_grid:   Alias for ``warped_grid``.
        grid_mode:    ``'control'`` (default) for true control lattice,
                  ``'dense'`` for a dense ``vis_n × vis_n`` visual mesh.
        vis_n:        Grid resolution used only when ``grid_mode='dense'``.
        src_color:    Colour for the original grid. Default ``'dodgerblue'``.
        dst_color:    Colour for the warped grid. Default ``'tomato'``.
        alpha:        Line / marker transparency.
        linewidth:    Grid line width.
        markersize:   Control-point marker size (0 to hide markers).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if target_grid is not None:
        warped_grid = target_grid

    # Normalise to lists so composition and non-composition are handled uniformly
    if isinstance(pts_src, torch.Tensor):
        srcs = [pts_src]
        dsts = [pts_dst]
    else:
        srcs = list(pts_src)
        dsts = list(pts_dst)

    # Base grid in normalised space.
    if grid_mode == 'control':
        vis_flat = srcs[-1].float().clone()  # (1, N, 2)
        npts = vis_flat.shape[1]
        grid_n = int(round(npts ** 0.5))
        is_square = grid_n * grid_n == npts
        is_level0 = npts == 5

        if not is_square and not is_level0:
            # Fallback for uncommon custom control sets.
            grid_mode = 'dense'

    if grid_mode == 'dense':
        coords = torch.linspace(-1.0, 1.0, vis_n)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        vis_flat = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)
        grid_n = vis_n
        is_square = True
        is_level0 = False

    # --- Forward-warp the base grid through all TPS stages --------------------
    warped_flat = vis_flat.clone()
    for ps, pd in zip(srcs, dsts):
        ps = ps.float()
        pd = pd.float()
        kw, aw = get_tps_transform(ps, pd)
        warped_flat = warp_points_tps(warped_flat, pd, kw, aw)

    if is_square:
        # Reshape to (grid_n, grid_n, 2): index as [row, col, xy]
        orig_grid = vis_flat.view(grid_n, grid_n, 2).detach().numpy()
        warped_grid_arr = warped_flat.view(grid_n, grid_n, 2).detach().numpy()
    else:
        # Level 0 special case: 4 corners + center
        orig_grid = vis_flat.squeeze(0).detach().numpy()
        warped_grid_arr = warped_flat.squeeze(0).detach().numpy()

    # --- Helper: normalised [-1,1] → pixel coordinates -----------------------
    def _to_px(arr):
        """arr shape (..., 2) with [..., 0]=x, [..., 1]=y"""
        if image_shape is None:
            return arr[..., 0], arr[..., 1]
        H, W = image_shape
        # -1 and +1 are the centres of the first and last pixels.
        px_x = (arr[..., 0] + 1.0) / 2.0 * (W - 1)
        px_y = (arr[..., 1] + 1.0) / 2.0 * (H - 1)
        return px_x, px_y

    # --- Draw a grid (rows + columns) from a (grid_n, grid_n, 2) array --------
    def _draw(grid_arr, color):
        if is_square:
            xs, ys = _to_px(grid_arr)
            for r in range(grid_n):
                ax.plot(xs[r, :], ys[r, :], color=color, alpha=alpha,
                        linewidth=linewidth)
            for c in range(grid_n):
                ax.plot(xs[:, c], ys[:, c], color=color, alpha=alpha,
                        linewidth=linewidth)
            if markersize > 0:
                ax.plot(xs.reshape(-1), ys.reshape(-1), 'o', color=color,
                        markersize=markersize, alpha=alpha, markeredgewidth=0)
        else:
            # Level 0 visualisation (4-corner frame + center spokes)
            # Point order from get_points_src(5): tl, tr, bl, br, c
            pts_x, pts_y = _to_px(grid_arr)
            order = [0, 1, 3, 2, 0]
            ax.plot(pts_x[order], pts_y[order], color=color, alpha=alpha,
                    linewidth=linewidth)
            cx, cy = pts_x[4], pts_y[4]
            for k in [0, 1, 2, 3]:
                ax.plot([cx, pts_x[k]], [cy, pts_y[k]], color=color,
                        alpha=alpha, linewidth=linewidth)
            if markersize > 0:
                ax.plot(pts_x, pts_y, 'o', color=color, markersize=markersize,
                        alpha=alpha, markeredgewidth=0)

    if original_grid:
        _draw(orig_grid, src_color)
    if warped_grid:
        _draw(warped_grid_arr, dst_color)


def apply_warp(
    image,
    points_src,
    points_dst,
    padding: str = 'zeros',
    fill_gaps_kernel: int = 15,
    fill_gaps_sigma: float = 5.0,
) -> torch.Tensor:
    """
    Apply a pre-computed TPS warp (from :func:`align_images`) to a new image.

    Works transparently for both composition and non-composition results:

    * **Non-composition** – ``points_src`` / ``points_dst`` are single tensors
      ``(1, N, 2)``.
    * **Composition** – ``points_src`` / ``points_dst`` are lists of tensors,
      one per stage (as returned by ``align_images(..., composition=True)``).

    Args:
        image:       Image to warp. Accepts np.ndarray or torch.Tensor in any
                     shape supported by :func:`_prepare_image`.
        points_src:  Source control points – tensor ``(1, N, 2)`` or list thereof.
        points_dst:  Destination control points – tensor ``(1, N, 2)`` or list thereof.
        padding:     Gap-filling strategy applied to the final output:
                     ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'filter'``.
        fill_gaps_kernel: Gaussian kernel size for ``padding='filter'``. Default 15.
        fill_gaps_sigma:  Gaussian sigma for ``padding='filter'``. Default 5.0.

    Returns:
        Warped image tensor of shape ``(1, C, H, W)``.
    """
    img = _prepare_image(image)
    _pm = 'zeros' if padding == 'filter' else padding

    # Normalise to lists so the loop below handles both cases identically.
    if isinstance(points_src, torch.Tensor):
        srcs = [points_src]
        dsts = [points_dst]
    else:
        srcs = list(points_src)
        dsts = list(points_dst)

    result = img
    for i, (ps, pd) in enumerate(zip(srcs, dsts)):
        is_last = (i == len(srcs) - 1)
        with torch.no_grad():
            result = differentiable_tps_warp(
                result, ps, pd,
                padding_mode=_pm if is_last else 'zeros',
            )

    if padding == 'filter':
        gap_mask = (result == 0).all(dim=1, keepdim=True).float()
        if gap_mask.any():
            result = fill_gaps(result, gap_mask,
                               kernel_size=fill_gaps_kernel,
                               sigma=fill_gaps_sigma)

    return result


def fill_gaps(
    image: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int = 15,
    sigma: float = 5.0,
) -> torch.Tensor:
    """
    Fill masked (zero/invalid) regions in *image* using a Gaussian-weighted
    average of surrounding valid pixels (diffusion-based inpainting).

    Args:
        image:       (B, C, H, W) tensor, gaps can contain any value.
        mask:        (B, 1, H, W) or (B, C, H, W) boolean/float mask where
                     1 (True) marks *invalid* pixels to be filled.
        kernel_size: Size of the Gaussian blur kernel (odd integer).
        sigma:       Standard deviation of the Gaussian kernel.

    Returns:
        Inpainted tensor of the same shape as *image*.
    """
    from kornia.filters import gaussian_blur2d

    valid = 1.0 - mask.float()
    if valid.shape[1] == 1 and image.shape[1] > 1:
        valid = valid.expand_as(image)

    filled = gaussian_blur2d(image * valid, (kernel_size, kernel_size), (sigma, sigma))
    weight = gaussian_blur2d(valid,         (kernel_size, kernel_size), (sigma, sigma))
    filled = filled / (weight + 1e-8)

    return image * valid + filled * (1.0 - valid)


def differentiable_tps_warp(
    image: torch.Tensor,
    points_src: torch.Tensor,
    points_dst: torch.Tensor,
    padding_mode: str = 'zeros',
) -> torch.Tensor:
    """
    Warp *image* differentiably using a Thin Plate Spline transformation.

    Args:
        image:        (B, C, H, W) or (H, W) input tensor.
        points_src:   (1, N, 2) or (N, 2) source control points.
        points_dst:   (1, N, 2) or (N, 2) destination control points.
        padding_mode: Padding for out-of-bounds pixels. One of:
                      'zeros'      – fill with 0 (default, original behaviour)
                      'border'     – repeat edge pixels
                      'reflection' – mirror image at borders

    Control points are in normalised coordinates where -1 and +1 are the
    centres of the first and last pixels (``align_corners=True``), the same
    convention as the sampling grid, so identical point sets give the
    identity.

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

    if padding_mode == 'zeros':
        # Fast path: use kornia's built-in function
        warped = warp_image_tps(image, points_src, kernel_weights, affine_weights,
                                align_corners=True)
    else:
        # Build the dense sampling grid manually so we can pass any padding_mode
        # to F.grid_sample (older kornia versions don't expose this in warp_image_tps).
        B, C, H, W = image.shape
        coords_y = torch.linspace(-1, 1, H, dtype=image.dtype, device=image.device)
        coords_x = torch.linspace(-1, 1, W, dtype=image.dtype, device=image.device)
        gy, gx = torch.meshgrid(coords_y, coords_x, indexing='ij')
        # (1, H*W, 2) in (x, y) order — same convention as warp_image_tps
        grid_pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)
        # TPS maps each output pixel back to its source coordinate
        src_pts = warp_points_tps(grid_pts, points_src, kernel_weights, affine_weights)
        src_pts = src_pts.expand(B, -1, -1).view(B, H, W, 2)
        warped = F.grid_sample(image, src_pts, mode='bilinear',
                               padding_mode=padding_mode, align_corners=True)

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
    weight_wrap: bool = False,
    loss_callback = None,
    padding: str = 'zeros',
    fill_gaps_kernel: int = 15,
    fill_gaps_sigma: float = 5.0,
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
        weight_wrap: If True, uses a warped-intensity-weighted data term:
                 ``(warped - target)^2 * warped``. This naturally downweights
                 zero-padded regions (contribution becomes ~0 where warped is 0).
                 Default: False.
        loss_callback: Optional callable invoked as ``loss_callback(iteration, loss_value)``
                     after every Adam step (1-based iteration index, float loss).
                     Ignored for L-BFGS. Default: None.
        padding:     How to fill out-of-bounds pixels produced by the warp.
                     'zeros'      – fill with 0 (default, original behaviour)
                     'border'     – repeat edge pixels
                     'reflection' – mirror image at borders
                     'filter'     – warp with zeros then apply Gaussian-weighted
                                    inpainting to fill the empty regions (smart
                                    diffusion-based gap filling)
        fill_gaps_kernel: Kernel size (odd int) for the Gaussian inpainting used
                     when padding='filter'. Default: 15.
        fill_gaps_sigma: Standard deviation for the inpainting Gaussian.
                     Default: 5.0.

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
                if weight_wrap:
                    diff_sq = diff_sq * warped
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
                if weight_wrap:
                    diff_sq = diff_sq * warped
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

    # --- Final warped image ----------------------------------------------------
    def _maybe_fill(warped_img):
        """Apply Gaussian-weighted inpainting when padding='filter'."""
        if padding == 'filter':
            gap_mask = (warped_img == 0).all(dim=1, keepdim=True).float()
            if gap_mask.any():
                warped_img = fill_gaps(
                    warped_img, gap_mask,
                    kernel_size=fill_gaps_kernel,
                    sigma=fill_gaps_sigma,
                )
        return warped_img

    # 'filter' warps with zeros then post-processes; all other modes are passed
    # directly to F.grid_sample.
    _pm = 'zeros' if padding == 'filter' else padding

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
            # Compose warp: optimization is done, so apply the requested padding.
            # Intermediate stages use zeros (gaps will be overwritten by later
            # stages); only the last stage uses _pm so border/reflection fills in
            # correctly at the true boundary.
            is_last = (stage_idx == len(levels) - 1)
            with torch.no_grad():
                current_src = differentiable_tps_warp(
                    current_src, points_src, points_dst,
                    padding_mode=_pm if is_last else 'zeros',
                )

    if composition:
        warped = _maybe_fill(current_src)
        return warped, stage_srcs, stage_dsts
    else:
        with torch.no_grad():
            warped = differentiable_tps_warp(src, points_src, points_dst,
                                             padding_mode=_pm)
        warped = _maybe_fill(warped)
        return warped, points_src, points_dst
