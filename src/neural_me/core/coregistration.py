"""
Co-registration of two instruments on one model grid, from their WCS.

The stereo inversion fits both views on one grid, the *model grid*: a crop of
an Earth-view helioprojective grid (TAN, observer at Earth) that covers the
reference instrument (Hinode/SP). The second instrument (SO/PHI-HRT) is
compared with it in two ways, one the inverse of the other:

* model grid -> instrument pixels (the forward model of a view in its own
  pixels): where each instrument pixel falls on the model grid,
  :func:`~neural_me.core.geometry.instrument_sampling_grid`, read by
  :class:`~neural_me.optimization.degradation.SparseWarpOperator`;
* instrument pixels -> model grid (the data on the model grid):
  :func:`reproject_to_model_grid`, the flux-conserving reprojection of the
  instrument cube onto the model grid.

Both are pure geometry from the two WCS (sunpy places every point on the
solar sphere). The one input that is not geometry is the pointing: the
helioprojective position of each instrument, set by hand when its header is
off. The steps, with the values of the 2022-03-31 Hinode/SP + SO/PHI-HRT run::

    ref = pointing_map(hinode_ic, center=(511.0, 329.0), scale=0.16,
                       date_obs="2022-03-31T19:39:59")
    phi = instrument_map(phi_l2_fits, crval=(-2464.626, 642.2178))
    wcs = model_grid_wcs(ref, obstime=phi.date)
    warp = instrument_sampling_grid(phi, wcs, (512, 512), crop_shape=(512, 512), n_sub=3, margin=2)
    phi_on_model = reproject_to_model_grid(fits.getdata(phi_l2_fits), phi, wcs)
    icont = int(np.argmax(phi_on_model[:, 0].mean((1, 2))))      # continuum wavelength
    save_sampling_grid("phi_native_warp.npz", warp, icont=icont)

These reproduce phi_native_warp.npz and the reprojected PHI cube of that run
bit for bit. ``crval`` moves the PHI L2 pointing by (-12, -77) arcsec (CRVAL1
-2452.626 -> -2464.626, CRVAL2 719.2178 -> 642.2178), a correction set by
hand. Hinode/SP lands on the model grid only to within the pointing of its
hand-made header; its fine alignment onto PHI-on-the-model-grid is a TPS
fitted to the images (:func:`~neural_me.core.warp.align_images`,
:func:`~neural_me.core.warp.apply_warp`), which is not geometry and not part
of this module.

Conventions: arrays are (ny, nx) with origin='lower' (row 0 is the bottom of
the image, as sunpy); pixel indices are 0-based at pixel centres; angles in
arcsec. The sampling grid holds (x, y) normalised with align_corners=True
(-1 and +1 are the centres of the first and last model pixels).
"""

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "pointing_map",
    "instrument_map",
    "model_grid_wcs",
    "save_sampling_grid",
    "reproject_to_model_grid",
]


def _require_sunpy():
    try:
        import astropy.units as u  # noqa: F401
        import sunpy.map  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sunpy and astropy are required for the co-registration helpers "
                          "(pip install 'neural_me[geometry]')") from exc


def pointing_map(image, center: Tuple[float, float], scale: float, date_obs: str,
                 crpix: Optional[Tuple[float, float]] = None, **meta):
    """A sunpy map for an image whose header has no WCS: helioprojective TAN,
    observer at Earth (no observer keywords), square pixels.

    Args:
        image: (ny, nx) array.
        center: (Tx, Ty) in arcsec, the helioprojective position of the reference pixel.
        scale: arcsec per pixel.
        date_obs: ISO date of the observation.
        crpix: (CRPIX1, CRPIX2), 1-based FITS reference pixel. Default (nx / 2, ny / 2),
            the convention of the header the Hinode/SP grid was built with
            (0-based pixel nx / 2 - 1, half a pixel left and below the centre);
            keep it to reproduce that grid.
        **meta: further header keywords (e.g. ``INSTRUME='SP'``).
    """
    _require_sunpy()
    import sunpy.map
    from astropy.io.fits import Header

    ny, nx = np.shape(image)
    crpix = (nx / 2, ny / 2) if crpix is None else crpix
    header = Header({"NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny, "CRPIX1": crpix[0], "CRPIX2": crpix[1],
                     "CRVAL1": center[0], "CRVAL2": center[1], "CDELT1": scale, "CDELT2": scale,
                     "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN", "CUNIT1": "arcsec", "CUNIT2": "arcsec",
                     "DATE-OBS": date_obs})
    header.update(meta)
    return sunpy.map.Map(np.asarray(image), header)


def instrument_map(path, plane: Sequence[int] = (0, 0), crval: Optional[Tuple[float, float]] = None, hdu: int = 0):
    """A sunpy map of one 2-D plane of an instrument cube, with the WCS of its
    header and, optionally, a corrected pointing.

    Args:
        path: FITS file whose header holds the celestial WCS on its last two axes
            (e.g. PHI-HRT L2 Stokes, (n_lambda, 4, ny, nx)).
        plane: indices of the leading axes of the plane (default: first wavelength,
            Stokes I).
        crval: (CRVAL1, CRVAL2) in arcsec replacing the header's: a pointing
            correction, the helioprojective position of the reference pixel CRPIX.
        hdu: index of the HDU.

    The header keeps everything else (observer position, date, rotation, plate
    scale), so the map places the pixels as the instrument saw them.
    """
    _require_sunpy()
    import astropy.units as u
    import sunpy.map
    from astropy.io import fits

    with fits.open(path) as hl:
        header = hl[hdu].header.copy()
        data = np.asarray(hl[hdu].data[tuple(plane)])
    if data.ndim != 2:
        raise ValueError(f"plane {tuple(plane)} leaves a {data.ndim}-D array, not an image")
    for k in range(3, header.get("NAXIS", 2) + 1):
        header.remove(f"NAXIS{k}", ignore_missing=True)
    header["NAXIS"] = 2
    if crval is not None:                               # arcsec -> the units of the header (arcsec for PHI: unchanged)
        for i in (1, 2):
            header[f"CRVAL{i}"] = (float(crval[i - 1]) * u.arcsec).to_value(u.Unit(header.get(f"CUNIT{i}", "deg")))
    return sunpy.map.Map(data, header)


def model_grid_wcs(reference_map, obstime, full_size: int = 8750, full_fov: float = 1400.0):
    """The WCS of the model grid: the crop of an Earth-view grid that covers the
    reference map.

    The full grid is ``full_size`` x ``full_size`` pixels of ``full_fov /
    full_size`` arcsec (0.16'' by default, the Hinode/SP plate scale), TAN,
    centred on the disc centre seen from Earth at ``obstime``. The crop has the
    shape of ``reference_map`` and starts at the full-grid pixel nearest to its
    bottom-left corner, as ``reproject_to(full grid).submap(bottom_left,
    top_right)`` does, so a map reprojected straight onto this WCS equals that
    submap bit for bit.

    Args:
        reference_map: sunpy map of the reference instrument (its shape and
            bottom-left corner fix the crop).
        obstime: time of the Earth observer of the full grid. It fixes the crop:
            the reference corner, at the date of ``reference_map``, is carried to
            this time. The 2022-03-31 grid used the PHI time (``phi.date``,
            19:56:08), 16 min after the Hinode date: with the Hinode date instead
            the crop starts one pixel further left and the sampling grid moves by
            up to 0.12 model px.
        full_size, full_fov: pixels and arcsec of the full grid. Only the crop
            matters; they fix its pixel phase (the full grid has the disc centre at
            pixel (full_size - 1) / 2).

    Returns:
        astropy.wcs.WCS with ``array_shape`` the model grid, observer at Earth.
    """
    _require_sunpy()
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from sunpy.coordinates import frames

    ref = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective(observer="earth", obstime=obstime))
    scale = full_fov / full_size
    full = WCS(sunpy.map.make_fitswcs_header((full_size, full_size), ref, scale=[scale, scale] * u.arcsec / u.pix))
    bl = full.world_to_pixel(reference_map.bottom_left_coord)
    origin = np.round([float(bl[0]), float(bl[1])])
    wcs = full.deepcopy()
    wcs.wcs.crpix = wcs.wcs.crpix - origin
    wcs.array_shape = reference_map.data.shape
    return wcs


def save_sampling_grid(path, sampling: dict, **extra):
    """Write the output of :func:`~neural_me.core.geometry.instrument_sampling_grid`
    as the ``.npz`` that
    :meth:`~neural_me.optimization.degradation.SparseWarpOperator.from_file` reads.

    Keys: ``grid`` (H, W, 2) and ``grid_sub`` (H, W, K, 2) float32, (x, y) on the
    model grid normalised with align_corners=True (2.0 off the grid); ``valid``
    (H, W) bool; ``r0``, ``c0``, the first row and column of the instrument crop
    in the full instrument image; and ``extra`` (e.g. ``icont``, the wavelength
    index of the continuum, which phi_native_warp.npz holds).
    """
    np.savez(path, grid=sampling["grid"], grid_sub=sampling["grid_sub"], valid=sampling["valid"],
             r0=sampling["r0"], c0=sampling["c0"], **extra)


def reproject_to_model_grid(cube, instrument_map, model_wcs, algorithm: str = "exact"):
    """The instrument cube on the model grid: the inverse of the sampling grid.

    Each plane is reprojected with sunpy (``GenericMap.reproject_to``) from the
    instrument WCS to the model WCS, both with their own observer, so the
    change of viewpoint (foreshortening, rotation) is included.
    ``algorithm='exact'`` integrates the overlap of the pixels on the sphere:
    it conserves the surface brightness and averages over the larger PHI
    pixels; 'interpolation' is faster and samples instead.

    Args:
        cube: (..., ny, nx) array in the instrument pixels (e.g. PHI L2 Stokes,
            (6, 4, 1152, 1152)).
        instrument_map: sunpy map with the instrument WCS (:func:`instrument_map`);
            its data are not used.
        model_wcs: WCS of the model grid (:func:`model_grid_wcs`), with
            ``array_shape``.
        algorithm: 'exact', 'interpolation' or 'adaptive'.

    Returns:
        (..., ny_m, nx_m) float64, NaN where no instrument pixel covers the model pixel.
    """
    _require_sunpy()
    import sunpy.map

    cube = np.asarray(cube)
    lead = cube.shape[:-2]
    out = np.empty(lead + tuple(model_wcs.array_shape))
    for idx in np.ndindex(*lead):
        plane = sunpy.map.Map(cube[idx], instrument_map.meta)
        out[idx] = plane.reproject_to(model_wcs, algorithm=algorithm).data
    return out
