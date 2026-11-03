"""
Observer frame → local solar vertical frame geometry.

The stereoscopic inversion solves for a single Cartesian magnetic vector
``(B_los, B_n, B_w)`` expressed in the **reference view's** line-of-sight frame
(view index 1 of
:class:`~neural_me.core.atmospheres.stereoscopic.StereoscopicAtmosphere`).
For physical interpretation that vector must be rotated into the local solar
frame ``(B_r, B_tN, B_tW)``: radial (normal to the photosphere),
tangential-North (meridional) and tangential-West (zonal).

The rotation is a pure change of orthonormal basis, so ``|B|`` is preserved
exactly, ``det(R) = +1``, and the operation is trivially differentiable.

Supplying the geometry
----------------------
Everything here is driven by a ``sunpy.map.GenericMap`` describing the grid the
magnetic vector lives on — whichever instrument defines the reference view for
a given run, at whatever pointing.  Nothing in this module is specialised to
one dataset.

If the reference view's FITS headers have been stripped (common after warping
or reprojection), rebuild the WCS with ``sunpy.map.make_fitswcs_header`` from
the pointing used to construct the grid::

    ref = SkyCoord(crval_x, crval_y, obstime=date_obs, observer="earth",
                   frame=sunpy.coordinates.frames.Helioprojective)
    header = sunpy.map.make_fitswcs_header(
        np.zeros(grid_shape), ref,
        reference_pixel=[(nx - 1) / 2, (ny - 1) / 2] * u.pix,
        scale=[cdelt, cdelt] * u.arcsec / u.pix,
    )
    smap = sunpy.map.Map(np.zeros(grid_shape), header)

Pass ``observer=`` a real spacecraft coordinate when the reference view is not
near Earth.

Conventions
-----------
Two component orderings appear in this module:

``xyz``
    The right-handed sunpy ``Heliocentric`` ordering ``(x, y, z)`` =
    ``(West, North, toward-observer)``.  All vector algebra is done here so
    that cross products carry their usual signs.

``lnw``
    The ordering used by the atmosphere parameters, ``(B_los, B_n, B_w)``.
    This is ``xyz`` reversed, and is **left-handed** — never take a cross
    product in it.

Choice of tangential basis
--------------------------
The radial direction ``r̂`` is fixed by the pixel position and is independent of
any convention, so ``B_r`` is unambiguous.  Splitting the remaining transverse
field into "North" and "West" is not: it depends on which pole the tangential
basis is built around.

* ``north_axis = None`` → image-aligned basis (pole = projected solar north,
  i.e. the ``+y`` axis of the observer frame).  This reproduces the classical
  closed-form matrix

  .. math::

      \\begin{bmatrix} B_r \\\\ B_{tN} \\\\ B_{tW} \\end{bmatrix} =
      \\begin{bmatrix}
        \\cos\\Theta\\cos\\Phi & \\sin\\Theta & \\cos\\Theta\\sin\\Phi \\\\
       -\\sin\\Theta\\cos\\Phi & \\cos\\Theta & -\\sin\\Theta\\sin\\Phi \\\\
       -\\sin\\Phi            & 0           & \\cos\\Phi
      \\end{bmatrix}
      \\begin{bmatrix} B_{\\mathrm{LOS}} \\\\ B_N \\\\ B_W \\end{bmatrix}

  where :math:`\\Theta, \\Phi` are the **image-aligned** heliocentric latitude
  and longitude of the pixel (not Stonyhurst coordinates — see
  :func:`heliocentric_angles`).

* ``north_axis = ẑ_sun`` → true heliographic basis, with the pole on the solar
  rotation axis.  ``B_tN``/``B_tW`` are then genuinely meridional/zonal.  The
  two differ by a local rotation of order the B0 angle (up to ±7.2°), so this
  matters for the transverse split even though ``B_r`` is identical either way.

The image-aligned case is exactly the general case with
``north_axis = (0, 1, 0)``; there is a single code path.

Relation to :mod:`ISPy.util.remap`
----------------------------------
``ISPy.util.remap.vector_transformation`` is the same physics — the Gary & Hagyard (1990)
matrix as coded in Sun (2013), Eq. 7, which is what the SHARP pipeline uses.
The two agree to machine precision (see
``neural_me/tests/test_local_vertical_vs_remap.py``), with one sign to watch:

======================  ==========================================
this module             ``ISPy.util.remap``
======================  ==========================================
``B_r``                 ``field_z_h``          (identical)
``B_tW``                ``field_x_h``          (identical)
``B_tN``                ``-field_y_h``         (**negated**)
======================  ==========================================

``remap`` returns the SHARP/Sun ``B_theta`` convention, positive toward the
**south**; ``B_tN`` is positive toward the north.  Flip the sign before
comparing against SHARP ``Bt`` products.

The ``roll`` argument here is exactly ``remap``'s ``peff`` p-angle: the full
Gary & Hagyard matrix factorizes as ``A(P) = A(0) @ Rz(-P)``, i.e. the p-angle
only ever de-rotates the image components before the local-vertical rotation.
"""

from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly by the sunpy-dependent helpers
    import torch
except ImportError:  # pragma: no cover
    torch = None


__all__ = [
    "heliocentric_angles",
    "solar_north_axis",
    "wcs_roll_angle",
    "local_vertical_matrix",
    "local_vertical_matrix_from_map",
    "apply_local_vertical",
]


def _require_sunpy():
    try:
        import astropy.units as u  # noqa: F401
        import sunpy.map  # noqa: F401
        from sunpy.coordinates import frames  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sunpy and astropy are required for WCS-based geometry helpers. "
            "Install them, or build the rotation yourself with "
            "local_vertical_matrix(theta, phi)."
        ) from exc


def heliocentric_angles(smap) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel image-aligned heliocentric longitude/latitude.

    These are **not** Stonyhurst coordinates.  ``theta`` and ``phi`` are the
    latitude/longitude of a spherical system whose pole is the projected solar
    north of the image and whose prime meridian passes through disk centre:

    .. code::

        theta = asin(y / R)          y = North component of the pixel position
        phi   = atan2(x, z)          x = West, z = toward observer

    Substituting Stonyhurst lat/lon into :func:`local_vertical_matrix` instead
    would tilt ``r̂`` by the B0 angle and give the wrong radial field.

    Any image rotation (P-angle / ``CROTA`` / ``PC`` matrix) is already removed
    by ``pixel_to_world``, so the returned angles are relative to projected
    solar north regardless of detector roll.  This assumes the atmosphere's
    ``(B_n, B_w)`` are likewise referenced to solar north; if the Stokes Q
    reference direction of the data is rolled with respect to it, that roll
    must be removed before applying the rotation (see analysis/orientation.ipynb).

    Args:
        smap: A ``sunpy.map.GenericMap`` describing the inversion grid.

    Returns:
        ``(theta, phi, on_disk)``, each of shape ``smap.data.shape``.  Angles
        are in radians and are ``NaN`` off the solar disk, where ``on_disk``
        is ``False``.
    """
    _require_sunpy()
    import astropy.units as u
    from sunpy.coordinates import frames

    ny, nx = smap.data.shape
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
    world = smap.pixel_to_world(xi * u.pix, yi * u.pix)
    hc = world.transform_to(
        frames.Heliocentric(observer=smap.observer_coordinate, obstime=smap.date)
    )

    x = hc.x.to_value(u.R_sun)
    y = hc.y.to_value(u.R_sun)
    z = hc.z.to_value(u.R_sun)
    r = np.sqrt(x**2 + y**2 + z**2)

    on_disk = np.isfinite(r) & (r > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arcsin(np.clip(y / r, -1.0, 1.0))
        phi = np.arctan2(x, z)
    theta = np.where(on_disk, theta, np.nan)
    phi = np.where(on_disk, phi, np.nan)

    return theta, phi, on_disk


def solar_north_axis(smap) -> np.ndarray:
    """Unit vector along the solar rotation axis, in the observer ``xyz`` frame.

    Obtained by transforming the heliographic north pole rather than assuming
    ``(0, cos B0, sin B0)``, so it stays correct whatever frame conventions
    sunpy uses internally.

    Returns:
        Shape ``(3,)`` array in ``(West, North, toward-observer)`` order.
    """
    _require_sunpy()
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    pole = SkyCoord(
        0 * u.deg,
        90 * u.deg,
        1 * u.R_sun,
        frame=frames.HeliographicStonyhurst(obstime=smap.date),
    )
    hc = pole.transform_to(
        frames.Heliocentric(observer=smap.observer_coordinate, obstime=smap.date)
    )
    axis = np.array(
        [hc.x.to_value(u.R_sun), hc.y.to_value(u.R_sun), hc.z.to_value(u.R_sun)]
    )
    return axis / np.linalg.norm(axis)


def wcs_roll_angle(smap) -> float:
    """Roll of the detector ``+y`` axis away from projected solar north, in radians.

    Read from the WCS rotation matrix (``PC``/``CD``, or ``CROTA2``).  This is
    the geometric roll of the *image*; whether the Stokes ``Q>0`` reference
    direction of the polarimetric calibration shares it is an instrument
    question the WCS cannot answer.  See ``analysis/orientation.ipynb``.

    Returned with the sign of the ``peff = -CROTA2`` used by
    :func:`ISPy.util.remap.remap2cea`, which is what the ``roll`` argument
    of :func:`local_vertical_matrix` expects: ``CROTA2 = 30`` gives -30 deg.
    """
    _require_sunpy()
    m = np.asarray(smap.rotation_matrix, dtype=float)
    # atan2(PC2_1, PC2_2) is +CROTA2; peff is its negative.
    return float(-np.arctan2(m[1, 0], m[1, 1]))


def local_vertical_matrix(
    theta: np.ndarray,
    phi: np.ndarray,
    north_axis: Optional[np.ndarray] = None,
    roll: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Rotation from the observer LOS frame to the local solar vertical frame.

    Args:
        theta: Image-aligned heliocentric latitude in radians (see
            :func:`heliocentric_angles`).
        phi: Image-aligned heliocentric longitude in radians.
        north_axis: Pole of the tangential basis, as a 3-vector in ``xyz``
            (West, North, toward-observer) order.  ``None`` selects the
            image-aligned basis ``(0, 1, 0)``.  Pass :func:`solar_north_axis`
            to get a true heliographic (meridional/zonal) split.
        roll: Roll of the frame the transverse components ``(B_n, B_w)`` are
            referenced to, away from projected solar north, in radians.  The
            components are de-rotated by this angle before the local-vertical
            rotation is applied.  Leave at ``0.0`` when they are already
            referenced to solar north.  This is the same quantity as the
            ``peff`` p-angle of :func:`ISPy.util.remap.vector_transformation`.
        eps: Guard for the degenerate case ``r̂ ∥ north_axis`` (the pole
            itself), where the tangential basis is undefined.

    Returns:
        Array of shape ``theta.shape + (3, 3)`` mapping ``(B_los, B_n, B_w)``
        to ``(B_r, B_tN, B_tW)``.  Rows are ``r̂``, ``ê_N``, ``ê_W``; columns
        are indexed in ``lnw`` order.  ``NaN`` inputs propagate to ``NaN``
        matrices.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)

    # Radial unit vector in xyz = (West, North, toward-observer).
    r_hat = np.stack([ct * sp, st, ct * cp], axis=-1)

    if north_axis is None:
        pole = np.array([0.0, 1.0, 0.0])
    else:
        pole = np.asarray(north_axis, dtype=np.float64)
        pole = pole / np.linalg.norm(pole)
    pole = np.broadcast_to(pole, r_hat.shape)

    # Right-handed cross products, valid because we are in xyz.
    #   ê_W ∝ ẑ_sun × r̂   (points West at disk centre)
    #   ê_N  = r̂ × ê_W    (completes the right-handed triad, points North)
    e_w = np.cross(pole, r_hat)
    norm = np.linalg.norm(e_w, axis=-1, keepdims=True)
    e_w = e_w / np.where(norm < eps, np.nan, norm)
    e_n = np.cross(r_hat, e_w)

    rot_xyz = np.stack([r_hat, e_n, e_w], axis=-2)  # (..., 3, 3)

    if roll:
        # De-rotate the image components about the LOS before the local-vertical
        # rotation.  Gary & Hagyard's p-angle enters the same way: their full
        # matrix factorizes exactly as A(P) = A(0) @ Rz(-P), verified against
        # ISPy.util.remap in tests/test_local_vertical_vs_remap.py.
        c, s = np.cos(roll), np.sin(roll)
        rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        rot_xyz = rot_xyz @ rz

    # Reverse the component axis to convert the columns from xyz to lnw.
    return rot_xyz[..., ::-1].copy()


def local_vertical_matrix_from_map(
    smap,
    heliographic: bool = True,
    roll: float = 0.0,
    fill_offdisk_identity: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the per-pixel rotation directly from a map's WCS.

    Args:
        smap: ``sunpy.map.GenericMap`` describing the inversion grid.
        heliographic: If ``True`` (default) the tangential basis uses the solar
            rotation axis, giving true meridional/zonal components.  If
            ``False`` it uses projected image north, reproducing the classical
            closed-form matrix.  ``B_r`` is identical either way.
        roll: Roll of the transverse components away from solar north, in
            radians — see :func:`local_vertical_matrix`.  Defaults to ``0.0``
            rather than the WCS value, because the reference direction of
            ``(B_n, B_w)`` is set by the polarimetric calibration, not by the
            pointing.  Pass ``wcs_roll_angle(smap)`` if the two coincide.
        fill_offdisk_identity: Replace off-disk (``NaN``) matrices with the
            identity so the result never poisons a loss with ``NaN``.  Use the
            returned mask to discard those pixels.

    Returns:
        ``(rotation, on_disk)`` with shapes ``(ny, nx, 3, 3)`` and ``(ny, nx)``.
    """
    theta, phi, on_disk = heliocentric_angles(smap)
    north = solar_north_axis(smap) if heliographic else None
    rot = local_vertical_matrix(theta, phi, north_axis=north, roll=roll)

    if fill_offdisk_identity:
        rot = np.where(on_disk[..., None, None], rot, np.eye(3))

    return rot, on_disk


def apply_local_vertical(B_los, B_n, B_w, rotation):
    """Rotate a magnetic vector into the local solar vertical frame.

    Differentiable, allocation-light, and exactly norm-preserving (up to
    floating point) because ``rotation`` is orthonormal.

    Args:
        B_los, B_n, B_w: Tensors of matching (broadcastable) shape.
        rotation: Tensor of shape ``(..., 3, 3)`` from
            :func:`local_vertical_matrix`, broadcastable against the field
            components.

    Returns:
        ``(B_r, B_tN, B_tW)`` tensors with the shape of the inputs.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for apply_local_vertical")

    vec = torch.stack(
        torch.broadcast_tensors(B_los, B_n, B_w), dim=-1
    )  # (..., 3)
    rotation = rotation.to(device=vec.device, dtype=vec.dtype)
    out = torch.einsum("...ij,...j->...i", rotation, vec)
    return out[..., 0], out[..., 1], out[..., 2]
