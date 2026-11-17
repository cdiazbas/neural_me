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
    "sampling_grid_from_wcs",
    "sampling_grid_from_images",
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


def sampling_grid_from_wcs(instrument_map, model_wcs, model_shape, crop_shape=None, n_sub: int = 3, margin: float = 2):
    """Sampling grid of an instrument on the model grid when both have a WCS: pure geometry.

    Where each instrument pixel falls on the model grid, through the solar sphere (sunpy), with ``n_sub`` x ``n_sub``
    samples inside each pixel to integrate its footprint. The pointing of each WCS is taken as it is: correct it before
    (:func:`instrument_map` ``crval``, :func:`pointing_map`). Same as
    :func:`~neural_me.core.geometry.instrument_sampling_grid`; the dict it returns is read by
    :meth:`~neural_me.optimization.degradation.SparseWarpOperator.from_sampling` and written by
    :func:`save_sampling_grid`.
    """
    from neural_me.core.geometry import instrument_sampling_grid

    return instrument_sampling_grid(instrument_map, model_wcs, model_shape, crop_shape=crop_shape, n_sub=n_sub,
                                    margin=margin)


def _to_uint8(image, percentiles=(1.0, 99.0)):
    """A float image as uint8 for ORB: clipped at two percentiles of its finite values, NaN as the lower one."""
    img = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(img, percentiles)
    img = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.round(255 * img).astype(np.uint8)


def _stretch_grid():
    """Default stretches (t, theta deg) of the instrument image: none, and t = 1.5-3.5 every 15 deg."""
    return [(1.0, 0.0)] + [(t, float(th)) for t in (1.5, 2.0, 2.5, 3.0, 3.5) for th in range(0, 180, 15)]


def _orb_matches_per_stretch(image_instrument, image_model, stretches, n_features, percentiles):
    """ORB matches (Hamming, cross-check) of the instrument image, stretched by t along theta, against the model image,
    one (src, dst) pair of arrays per stretch, with the instrument keypoints mapped back to its original pixels (tilt
    simulation, as in ASIFT: ORB is not invariant to the anisotropic scale of a tilted view). NaN pixels are filled
    with the mean; each stretched image is mapped onto 0-255 with its own percentiles."""
    import cv2

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_m, des_m = orb.detectAndCompute(_to_uint8(image_model, percentiles), None)
    if des_m is None:
        raise ValueError("ORB found no keypoints in the model image")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    inst = np.asarray(image_instrument, dtype=np.float64)
    ny, nx = inst.shape
    filled = np.nan_to_num(inst, nan=float(np.nanmean(inst))).astype(np.float32)
    out = []
    for t, th in stretches:
        a = np.radians(th)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        S = R @ np.diag([t, 1.0]) @ R.T
        c = S @ np.array([[0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1]], dtype=np.float64)
        off = -c.min(axis=1)
        size = tuple(int(np.ceil(v)) + 1 for v in (c.max(axis=1) + off))
        img = cv2.warpAffine(filled, np.c_[S, off].astype(np.float32), size, flags=cv2.INTER_LINEAR)
        kp, des = orb.detectAndCompute(_to_uint8(img, percentiles), None)
        src, dst = np.zeros((0, 2)), np.zeros((0, 2))
        if des is not None:
            mt = bf.match(des, des_m)
            if mt:
                src = np.array([np.linalg.solve(S, np.asarray(kp[m.queryIdx].pt) - off) for m in mt])
                dst = np.array([kp_m[m.trainIdx].pt for m in mt], dtype=np.float64)
                keep = (src[:, 0] >= 0) & (src[:, 0] <= nx - 1) & (src[:, 1] >= 0) & (src[:, 1] <= ny - 1)
                src, dst = src[keep], dst[keep]
        out.append((src, dst))
    return out


def sampling_grid_from_images(image_instrument, image_model, crop_shape=None, n_sub: int = 3, margin: float = 2,
                              model: str = "projective", refine: Optional[str] = "ecc", stretches="auto",
                              n_best_stretches: int = 4, ransac_threshold: float = 15.0, node_clamp: float = 0.05,
                              node_levels=(0, 1, 2), node_niter: int = 400, ecc_passes: int = 2, n_features: int = 5000,
                              percentiles=(1.0, 99.0), min_inliers: int = 20, seed: int = 0):
    """Sampling grid of an instrument on the model grid when there is no WCS: a mapping fitted to two images.

    1. Matches: ORB keypoints of the instrument image stretched by each (t, theta) of ``stretches`` against the model
       image (a view tilted by ~100 deg is compressed ~2.8 times along one axis, and ORB on the image as it is found 10
       consistent matches on SO/PHI-HRT against Hinode/SP).
    2. Stretches: a RANSAC homography per stretch, and the ``n_best_stretches`` with most inliers; their matches are
       pooled. Pooling all the stretches of the broad default grid let wrong matches win RANSAC.
    3. Global map: a homography (``model='projective'``) or an affine map fitted to the pooled matches with RANSAC
       (``ransac_threshold``, model pixels; loose because no homography follows the sphere at large angles).
    4. Refinement. ``refine='ecc'`` (default): passes of OpenCV ``findTransformECC`` (intensity based) on the
       residual of the global map, for images that look alike. ``refine='nodes'``: the TPS of
       :func:`~neural_me.core.warp.align_images` between the instrument image and the model image resampled by the
       global map, control points moving at most ``node_clamp``, composed with the global map by evaluating its stages
       on the points; for very different views, where no homography follows the sphere. ``refine=None``: the global
       map.

    Measured distance to the true mapping (median / p95 or max, in model pixels):

    * similar images (a textured image and its sampling at a known homography, tests/test_sampling_grid_two_ways.py):
      ``'ecc'`` 0.013 / 0.026 (max); ``'nodes'`` 1.9 / 5.6 (max): the TPS adds noise where the homography is exact.
    * SO/PHI-HRT (native pixels) against Hinode/SP (model grid), 2022-03-31, ~100 deg apart, total polarization times
      continuum as images, against the WCS mapping: ``'nodes'`` 2.4 / 6.9 / 23 (max); ``'ecc'`` 5.6 / 20 / 42;
      ``None`` 5.0 / 24 / 50; ``node_clamp=0.02`` 3.5 / 23 / 56. With the continuum as images, ``'nodes'`` gave
      8.1 / 24 / 33. ECC followed by the nodes was worse than either alone in both cases.

    Use :func:`sampling_grid_from_wcs` whenever both instruments have a WCS: it is exact. The figures for different
    views come from one pair of observations.

    Parameters
    ----------
    image_instrument, image_model : (ny, nx) arrays
        Instrument image in its own pixels (NaN where it has no data) and the image on the model grid.
    crop_shape, n_sub, margin
        As in :func:`sampling_grid_from_wcs`.
    model : 'projective' or 'affine'
    refine : 'ecc', 'nodes' or None
    stretches : 'auto' or list of (t, theta_deg)
        Stretches of the instrument image for the matching; ``[(1, 0)]`` uses the image as it is.
    n_best_stretches : int
        Stretches whose matches are pooled.
    ransac_threshold : float
        Largest distance of an inlier to the RANSAC map, in model pixels.
    node_clamp, node_levels, node_niter
        Largest displacement of the TPS control points (normalised units), control-point levels, Adam steps per level.
    ecc_passes : int
    n_features, percentiles, min_inliers
        ORB keypoints per image, intensity range mapped onto 0-255, fewest RANSAC inliers accepted.
    seed : int
        Seed of the OpenCV random generator (RANSAC), for reproducible results.

    Returns
    -------
    dict
        The keys of :func:`sampling_grid_from_wcs`, plus ``model``, ``refine``, ``transform`` (3, 3, instrument pixels
        to model pixels, the global map after any ECC passes), ``stretches`` (the ones pooled), ``n_matches``,
        ``n_inliers`` and ``ecc``.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sampling_grid_from_images needs OpenCV: pip install opencv-python-headless") from exc
    from scipy.ndimage import map_coordinates
    from neural_me.core.geometry import sampling_grid_from_mapping

    if model not in ("projective", "affine"):
        raise ValueError(f"model must be 'projective' or 'affine', got {model!r}")
    if refine not in (None, "ecc", "nodes"):
        raise ValueError(f"refine must be None, 'ecc' or 'nodes', got {refine!r}")
    inst = np.asarray(image_instrument, dtype=np.float64)
    modl = np.asarray(image_model, dtype=np.float64)
    ny, nx = inst.shape
    stretches = _stretch_grid() if isinstance(stretches, str) and stretches == "auto" else list(stretches)

    def fit(src, dst, method, threshold, iters=20000):
        cv2.setRNGSeed(seed)
        if model == "projective":
            return cv2.findHomography(np.float32(src), np.float32(dst), method, threshold, maxIters=iters, confidence=0.999)
        A, inl = cv2.estimateAffine2D(np.float32(src), np.float32(dst), method=method, ransacReprojThreshold=threshold,
                                      maxIters=iters, confidence=0.999)
        return (None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])), inl

    per = _orb_matches_per_stretch(inst, modl, stretches, n_features, percentiles)
    counts = []
    for s, d in per:
        _, inl = fit(s, d, cv2.RANSAC, ransac_threshold, 5000) if len(s) >= 8 else (None, None)
        counts.append(0 if inl is None else int(inl.sum()))
    best = [i for i in np.argsort(counts)[::-1][:n_best_stretches] if counts[i] > 0]
    if not best:
        raise ValueError("no stretch of the instrument image gave consistent matches")
    src = np.concatenate([per[i][0] for i in best])
    dst = np.concatenate([per[i][1] for i in best])
    T, inl = fit(src, dst, cv2.RANSAC, ransac_threshold)
    n_inliers = 0 if inl is None else int(inl.sum())
    if T is None or n_inliers < min_inliers:
        raise ValueError(f"RANSAC found {n_inliers} inliers among {len(src)} matches (min_inliers={min_inliers})")
    T = np.asarray(T, dtype=np.float64)

    def apply_T(x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        w = T[2, 0] * x + T[2, 1] * y + T[2, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = (T[0, 0] * x + T[0, 1] * y + T[0, 2]) / w
            my = (T[1, 0] * x + T[1, 1] * y + T[1, 2]) / w
        return np.where(w <= 0, np.nan, mx), np.where(w <= 0, np.nan, my)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    good = np.isfinite(inst)
    ecc = []
    if refine == "ecc":
        fill = float(np.nanmean(inst))
        template = np.nan_to_num(inst, nan=fill).astype(np.float32)
        motion = cv2.MOTION_HOMOGRAPHY if model == "projective" else cv2.MOTION_AFFINE
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
        for _ in range(ecc_passes):
            mx, my = apply_T(xx, yy)
            warped = map_coordinates(modl, [my, mx], order=3, mode="constant", cval=np.nan)
            ok = (np.isfinite(warped) & good).astype(np.uint8)
            W0 = np.eye(3, dtype=np.float32) if model == "projective" else np.eye(2, 3, dtype=np.float32)
            try:
                cc, W = cv2.findTransformECC(template, np.nan_to_num(warped, nan=fill).astype(np.float32), W0, motion,
                                             criteria, ok, 3)
            except cv2.error as exc:
                raise ValueError(f"ECC refinement did not converge: {exc}") from exc
            T = T @ (W.astype(np.float64) if model == "projective" else np.vstack([W, [0.0, 0.0, 1.0]]))
            ecc.append(float(cc))
    to_model = apply_T
    if refine == "nodes":
        import torch
        from kornia.geometry.transform import get_tps_transform, warp_points_tps
        from neural_me.core.warp import align_images

        mx, my = apply_T(xx, yy)
        src_img = np.where(good, np.nan_to_num(map_coordinates(modl, [my, mx], order=1, mode="constant", cval=0.0)),
                           0.0)
        tgt_img = np.where(good, inst, 0.0)

        def as_t(a):
            return torch.from_numpy((a / np.mean(a[good])).astype(np.float32))[None, None]

        nl = len(node_levels)
        _, ps, pd = align_images(as_t(src_img), as_t(tgt_img), levels=list(node_levels), clamp=[node_clamp] * nl,
                                 lr=[5e-4] * nl, niter=node_niter, optimizer="adam", verbose=False, composition=True,
                                 padding="zeros", weight_wrap=True)
        ps, pd = (ps, pd) if isinstance(ps, (list, tuple)) else ([ps], [pd])
        stages = []
        for a, b in zip(ps, pd):
            a, b = a.detach().double(), b.detach().double()
            kw, aw = get_tps_transform(b, a)
            stages.append((kw.detach(), aw.detach(), a))

        def to_model(x, y):
            # the composed warp samples stage 1 at stage 2 at ... : apply the stages from the last to the first
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            g = torch.from_numpy(np.stack([2 * x.ravel() / (nx - 1) - 1, 2 * y.ravel() / (ny - 1) - 1], -1))[None]
            with torch.no_grad():
                for kw, aw, a in reversed(stages):
                    g = warp_points_tps(g, a, kw, aw)
            g = g[0].numpy()
            mx_, my_ = apply_T((g[:, 0] + 1) / 2 * (nx - 1), (g[:, 1] + 1) / 2 * (ny - 1))
            return mx_.reshape(x.shape), my_.reshape(y.shape)

    out = sampling_grid_from_mapping(to_model, (ny, nx), modl.shape, crop_shape, n_sub, margin)
    out.update(model=model, refine=refine, transform=T, stretches=[stretches[i] for i in best], n_matches=len(src),
               n_inliers=n_inliers, ecc=ecc)
    return out


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
