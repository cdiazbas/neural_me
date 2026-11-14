"""
Velocity vector from the line-of-sight velocities of two views.

Each view measures one projection of the velocity: ``v_LOS,i = o_i - v . e_i``,
with ``e_i`` the unit vector towards observer ``i`` (redshift positive) and
``o_i`` the zero point of its wavelength scale. Two views give two projections,
so the component of ``v`` along ``e_1 x e_2`` is not measured. Two ways to fill it:

* :func:`field_aligned_velocity`: the flow is along the field, ``v = s b``, and the
  speed ``s`` follows pixel by pixel from both velocities (least squares). The
  residual perpendicular to ``(c_1, c_2) = (b . e_1, b . e_2)`` tests the hypothesis;
  the zero points are fitted globally or given.
* :func:`solve_velocity`: the same pixel by pixel in closed form (minimum norm, or with a
  penalty on the part of ``v`` perpendicular to the field), with no spatial coupling.
* :func:`fit_velocity_field`: a neural field (:class:`~neural_me.core.nf.GridField`)
  for the velocity in the local frame ``(v_r, v_tN, v_tW)`` fitted to both
  velocities, either free with an optional soft penalty on the part of ``v``
  perpendicular to the field (``mode="free"``, ``lam_align``), or with the flow
  along the field and a smooth speed (``mode="aligned"``).

Vectors are in the frame of view 1, ordered ``(los, n, w)`` as the field of
:class:`~neural_me.core.atmospheres.stereoscopic.StereoscopicAtmosphere`; the
local frame is that of :func:`~neural_me.core.geometry.local_vertical_matrix_from_map`
(rows ``r, tN, tW``).
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = ["view_los_vectors", "unseen_direction", "zero_at_umbral_core", "field_aligned_velocity",
           "to_local", "solve_velocity", "fit_velocity_field"]


def view_los_vectors(gamma: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unit vectors towards observers 1 and 2 in the ``(los, n, w)`` frame of view 1, for the angles
    (degrees) of :class:`~neural_me.core.atmospheres.stereoscopic.StereoscopicAtmosphere`: the
    line-of-sight component in view 2 of a vector ``B`` is ``B . e_2``."""
    g, b = np.radians(gamma), np.radians(beta)
    return np.array([1.0, 0.0, 0.0]), np.array([np.cos(g) * np.cos(b), np.sin(b), np.sin(g) * np.cos(b)])


def unseen_direction(e1: np.ndarray, e2: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit vector along ``e_1 x e_2``, the velocity component that neither view measures; in the
    local frame ``(r, tN, tW)`` if the rotation ``R`` (..., 3, 3) is given."""
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    return n if R is None else np.einsum("...ij,j->...i", R, n)


def zero_at_umbral_core(v: np.ndarray, continuum: np.ndarray, region: Optional[np.ndarray] = None,
                        umbra_level: float = 0.5, core_fraction: float = 0.2) -> float:
    """Zero point of a velocity map at the umbral core: median of ``v`` over the darkest
    ``core_fraction`` of the umbra, the pixels of ``region`` with a continuum below ``umbra_level``
    times its median there."""
    region = np.ones(v.shape, bool) if region is None else region
    umbra = region & (continuum < umbra_level * np.median(continuum[region]))
    core = umbra & (continuum < np.percentile(continuum[umbra], 100 * core_fraction))
    return float(np.median(v[core]))


def field_aligned_velocity(v1: np.ndarray, v2: np.ndarray, b: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           zero: Union[str, Sequence[float]] = "fit", kappa_min: float = 0.3) -> dict:
    """Speed along the field from two line-of-sight velocities, ``v_LOS,i = o_i - s c_i``.

    v1, v2: LOS velocities (redshift > 0); b: unit field vector (..., 3) in the frame of view 1;
    mask: pixels used (default all); zero: ``"fit"`` (zero points by least squares on the residual,
    which is linear in them) or ``(o1, o2)``; kappa_min: pixels with ``(c_1^2 + c_2^2)^(1/2)`` below it,
    where neither view sees a flow along the field, are left out.

    Returns a dict: ``s`` (speed along ``b``), ``residual`` (perpendicular residual, same units),
    ``offsets`` (o1, o2), ``use`` (pixels used), ``kappa``, ``robust_sigma`` of the residual and
    ``v`` (= s b, frame of view 1)."""
    c1, c2 = b @ e1, b @ e2
    k2 = c1 ** 2 + c2 ** 2
    k = np.sqrt(k2)
    use = (np.ones(v1.shape, bool) if mask is None else mask) & (k > kappa_min) & np.isfinite(v1) & np.isfinite(v2)
    if isinstance(zero, str):
        if zero != "fit":
            raise ValueError(f"zero must be 'fit' or (o1, o2), got {zero!r}")
        A = np.stack([-c2[use] / k[use], c1[use] / k[use]], axis=-1)
        y = -(v1[use] * c2[use] - v2[use] * c1[use]) / k[use]
        o = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        o = np.asarray(zero, dtype=float)
    d1, d2 = v1 - o[0], v2 - o[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -(d1 * c1 + d2 * c2) / k2
        r = (d1 * c2 - d2 * c1) / k
    rr = r[use]
    sigma = 1.4826 * float(np.median(np.abs(rr - np.median(rr)))) if rr.size else float("nan")
    return dict(s=s, residual=r, offsets=o, use=use, kappa=k, robust_sigma=sigma, v=s[..., None] * b)


def to_local(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Vector (..., 3) in the frame of view 1 to the local frame ``(r, tN, tW)``."""
    return np.einsum("...ij,...j->...i", R, v)


def solve_velocity(v1: np.ndarray, v2: np.ndarray, e1: np.ndarray, e2: np.ndarray, R: np.ndarray,
                   b: Optional[np.ndarray] = None, lam_align: float = 0.0, loss: str = "l2",
                   n_irls: int = 50, eps: float = 1e-4) -> np.ndarray:
    """Velocity in the local frame ``(v_r, v_tN, v_tW)``, pixel by pixel: the minimum of
    ``D(E v - y) + lam_align |(I - b b^T) v|^2`` with ``E`` the rows ``-e_1, -e_2``, ``y = (v1, v2)``, and
    ``D`` the sum of squares (``loss="l2"``, closed form) or of absolute values of the two residuals
    (``loss="l1"``, iteratively reweighted least squares from the l2 solution, ``n_irls`` iterations,
    residuals floored at ``eps``).

    ``lam_align = 0``: the minimum-norm solution (pseudo-inverse), with the component along
    :func:`unseen_direction` equal to 0. ``lam_align > 0`` (needs ``b``, frame of view 1): that component
    is filled towards the field; the larger ``lam_align``, the closer to the field-aligned solution and the
    larger the residual of the two velocities. No spatial coupling: for a smooth solution use
    :func:`fit_velocity_field`. Takes well under a second for 512 x 512 pixels."""
    E = -np.stack([np.einsum("...ij,j->...i", R, e1), np.einsum("...ij,j->...i", R, e2)], axis=-2)   # (..., 2, 3)
    y = np.stack([v1, v2], axis=-1)
    if lam_align == 0:
        return np.einsum("...ij,...j->...i", np.linalg.pinv(E), y)
    if b is None:
        raise ValueError("b is needed for lam_align > 0")
    if loss not in ("l2", "l1"):
        raise ValueError(f"loss must be 'l2' or 'l1', got {loss!r}")
    bl = to_local(b, R)
    P = lam_align * (np.eye(3) - bl[..., :, None] * bl[..., None, :])

    def weighted(w):                                        # minimum of sum_k w_k r_k^2 + lam |P v|^2
        A = np.einsum("...ki,...k,...kj->...ij", E, w, E) + P
        return np.linalg.solve(A, np.einsum("...ki,...k,...k->...i", E, w, y)[..., None])[..., 0]

    v = weighted(np.ones_like(y))
    if loss == "l1":
        # sum |r| + lam |P v|^2 is stationary where sum sign(r) E + 2 lam P v = 0, i.e. weights 1 / (2 |r|)
        for _ in range(n_irls):
            r = np.einsum("...ij,...j->...i", E, v) - y
            v = weighted(0.5 / np.maximum(np.abs(r), eps))
    return v


def fit_velocity_field(v1: np.ndarray, v2: np.ndarray, e1: np.ndarray, e2: np.ndarray, R: np.ndarray,
                       mask: np.ndarray, b: Optional[np.ndarray] = None, mode: str = "free",
                       lam_align: float = 0.0, loss: str = "l1", smooth: float = 0.0, n_iter: int = 1500, lr: float = 1e-2, scale: float = 2.0,
                       seed: int = 0, grid_kwargs: Optional[dict] = None, verbose: bool = False) -> dict:
    """Neural field for the velocity in the local frame, fitted to two LOS velocities.

    v1, v2: LOS velocities with their zero points removed (redshift > 0), (ny, nx); e1, e2: unit
    vectors towards the observers in the frame of view 1; R: (ny, nx, 3, 3) rotation to the local
    frame; mask: pixels in the loss; b: unit field vector (ny, nx, 3) in the frame of view 1, needed
    for ``mode="aligned"`` and for ``lam_align > 0``.

    ``mode="free"``: the field outputs ``(v_r, v_tN, v_tW)``; the loss is the mean absolute (``loss="l1"``)
    or squared (``loss="l2"``) residual of both views plus ``lam_align`` times the mean squared part of ``v`` perpendicular to ``b``. The
    component along :func:`unseen_direction` is set only by that penalty: the network gives the coordinates on a basis
    with two axes in the plane of the lines of sight and one along the unseen direction, all starting at zero, so the
    data term alone leaves the third at zero (minimum-norm solution).
    ``mode="aligned"``: the field outputs the speed ``s`` and ``v = s b`` (the smooth counterpart of
    :func:`field_aligned_velocity`).
    ``smooth``: weight of a smoothness penalty, the mean over ``mask`` of the squared difference between each
    component of the velocity and the weighted mean of its eight neighbours (sides 1, corners 0.5; the kernel of
    the spatial regularization of the inversion). The grids of the network reach the pixel scale, so without it
    the field reproduces the fine structure of the data, defects included.

    Outputs are multiplied by ``scale`` (km/s if the velocities are), and the network is a
    :class:`~neural_me.core.nf.GridField` (``grid_kwargs`` passed on). Returns a dict: ``v_local``
    (ny, nx, 3), ``data_residual`` (final mean absolute residual) and ``history`` (loss per iteration)."""
    from .nf import GridField

    if loss not in ("l1", "l2"):
        raise ValueError(f"loss must be 'l1' or 'l2', got {loss!r}")
    if mode not in ("free", "aligned"):
        raise ValueError(f"mode must be 'free' or 'aligned', got {mode!r}")
    if (mode == "aligned" or lam_align > 0) and b is None:
        raise ValueError("b is needed for mode='aligned' and for lam_align > 0")
    ny, nx = v1.shape
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    P1 = T(np.einsum("...ij,j->...i", R, e1)).reshape(-1, 3)             # local components of e_1
    P2 = T(np.einsum("...ij,j->...i", R, e2)).reshape(-1, 3)
    BL = T(to_local(b, R)).reshape(-1, 3) if b is not None else None
    # free mode: the network gives the coordinates on an orthonormal basis (u1, u2 in the plane of the two lines of sight,
    # n along e1 x e2), so the data term never touches the third one, whatever the optimizer
    u1 = e1 / np.linalg.norm(e1)
    u2 = e2 - (e2 @ u1) * u1; u2 = u2 / np.linalg.norm(u2)
    Q = T(np.stack([np.einsum("...ij,j->...i", R, u) for u in (u1, u2, unseen_direction(e1, e2))], axis=-2)).reshape(-1, 3, 3)
    M = T(mask).reshape(-1) > 0
    V1, V2 = T(np.nan_to_num(v1)).reshape(-1), T(np.nan_to_num(v2)).reshape(-1)
    torch.manual_seed(seed)
    # without a prior the third coordinate is 0 by construction: Adam normalizes each parameter, so even the ~1e-9
    # (float32) gradient of the data term on it would turn into steps of size lr
    dim_out = 1 if mode == "aligned" else (3 if lam_align > 0 else 2)
    # outputs start at exactly zero (last layer zeroed): without a prior, the gradient of the data term only moves the
    # velocity within the plane of the two lines of sight, so the unseen component stays at zero (minimum norm)
    kw = dict(zero_init=tuple(range(dim_out)))
    kw.update(grid_kwargs or {})
    net = GridField((ny, nx), dim_out=dim_out, **kw)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def velocity():
        o = scale * net()
        if mode == "aligned":
            return o[:, :1] * BL
        return torch.einsum("pk,pkj->pj", o, Q[:, :o.shape[1]])

    K = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]]) / 6.0
    K = K.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def roughness(v):                                       # mean over the mask of sum_c (v_c - K * v_c)^2
        img = v.T.reshape(1, 3, ny, nx)
        nb = torch.nn.functional.conv2d(torch.nn.functional.pad(img, (1, 1, 1, 1), mode="replicate"), K, groups=3)
        return ((img - nb) ** 2).sum(1).reshape(-1)[M].mean()

    history = []
    for it in range(n_iter):
        opt.zero_grad()
        v = velocity()
        r1 = -(v * P1).sum(-1) - V1
        r2 = -(v * P2).sum(-1) - V2
        data = 0.5 * (r1[M].abs().mean() + r2[M].abs().mean())
        loss_ = data if loss == "l1" else 0.5 * ((r1[M] ** 2).mean() + (r2[M] ** 2).mean())
        if lam_align > 0 and mode == "free":
            vp = v - (v * BL).sum(-1, keepdim=True) * BL
            loss_ = loss_ + lam_align * (vp[M] ** 2).sum(-1).mean()
        if smooth > 0:
            loss_ = loss_ + smooth * roughness(v)
        loss_.backward()
        opt.step()
        history.append(float(loss_))
        if verbose and (it % 250 == 0 or it == n_iter - 1):
            print(f"fit_velocity_field it {it:5d} | loss {float(loss_):.4f} | mean |r| {float(data):.4f}")
    with torch.no_grad():
        v = velocity()
        r1 = -(v * P1).sum(-1) - V1
        r2 = -(v * P2).sum(-1) - V2
        data = float(0.5 * (r1[M].abs().mean() + r2[M].abs().mean()))
    return dict(v_local=v.numpy().reshape(ny, nx, 3), data_residual=data, history=np.array(history))
