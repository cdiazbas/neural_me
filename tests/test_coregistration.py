"""core.coregistration: the model grid, the instrument map with its pointing, and the two directions between them.
The last tests rebuild the products of the 2022-03-31 run from the raw files and compare them bit for bit."""
import contextlib
import io

import numpy as np
import pytest
import torch

sunpy_map = pytest.importorskip("sunpy.map")
pytest.importorskip("reproject")
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.coordinates import frames

from conftest import HINODE_SP_CRVAL, HINODE_SP_CDELT, HINODE_SP_DATE_OBS, PHI_CRVAL, PHI_L2
from neural_me.core.coregistration import (
    instrument_map,
    model_grid_wcs,
    pointing_map,
    reproject_to_model_grid,
    save_sampling_grid,
)
from neural_me.core.geometry import instrument_sampling_grid
from neural_me.optimization import SparseWarpOperator

DATE = "2022-03-31T19:39:59"


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def wcs_map(shape, centre, scale, date=DATE, observer="earth"):
    ref = SkyCoord(centre[0] * u.arcsec, centre[1] * u.arcsec,
                   frame=frames.Helioprojective(observer=observer, obstime=date))
    header = sunpy_map.make_fitswcs_header(np.zeros(shape), ref, scale=[scale, scale] * u.arcsec / u.pix)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    data = 1 + 0.5 * np.cos(x / 1.7) * np.sin(y / 2.3)                 # structure at a few pixels
    return sunpy_map.Map(data, header)


def test_pointing_map_reference_pixel():
    """CRPIX = n / 2 (FITS, 1-based): the reference value sits at 0-based pixel n / 2 - 1."""
    m = pointing_map(np.zeros((20, 30)), (300.0, 200.0), 0.5, DATE, INSTRUME="SP")
    c = m.wcs.pixel_to_world(14, 9)
    assert c.Tx.to_value(u.arcsec) == pytest.approx(300.0, abs=1e-9)
    assert c.Ty.to_value(u.arcsec) == pytest.approx(200.0, abs=1e-9)
    assert m.meta["instrume"] == "SP" and m.scale.axis1.to_value(u.arcsec / u.pix) == 0.5
    m = pointing_map(np.zeros((20, 30)), (300.0, 200.0), 0.5, DATE, crpix=(1, 1))
    assert m.wcs.pixel_to_world(0, 0).Tx.to_value(u.arcsec) == pytest.approx(300.0, abs=1e-9)


def test_instrument_map_plane_and_pointing(tmp_path):
    """One plane of a 4-D cube, with the header's WCS and a new CRVAL; the rest of the header is kept."""
    m = wcs_map((16, 18), (100.0, 50.0), 1.0)
    cube = np.arange(2 * 3 * 16 * 18, dtype=np.float32).reshape(2, 3, 16, 18)
    hdr = fits.Header(m.wcs.to_header())
    hdr["DATE-OBS"] = DATE
    hdu = fits.PrimaryHDU(cube, header=hdr)
    path = tmp_path / "cube.fits"
    hdu.writeto(path)
    a = instrument_map(path, plane=(1, 2))
    assert np.array_equal(a.data, cube[1, 2])
    assert a.wcs.pixel_to_world(5, 7).separation(m.wcs.pixel_to_world(5, 7)).arcsec < 1e-6
    b = instrument_map(path, plane=(1, 2), crval=(90.0, 60.0))
    assert b.reference_coordinate.Tx.to_value(u.arcsec) == pytest.approx(90.0)
    assert b.reference_coordinate.Ty.to_value(u.arcsec) == pytest.approx(60.0)
    assert b.meta["naxis"] == 2 and "naxis3" not in b.meta
    with pytest.raises(ValueError, match="not an image"):
        instrument_map(path, plane=(1,))


def test_model_grid_is_a_full_grid_crop_at_the_reference_corner():
    ref = pointing_map(np.zeros((40, 50)), (300.0, 200.0), 1.6, DATE)
    wcs = model_grid_wcs(ref, DATE, full_size=875, full_fov=1400.0)
    assert wcs.array_shape == (40, 50)
    corner = wcs.world_to_pixel(ref.bottom_left_coord)
    assert all(abs(float(c)) <= 0.5 for c in corner)                  # the full-grid pixel nearest to the corner
    assert np.allclose(wcs.wcs.cdelt * 3600, 1.6)
    # pixel phase of the full grid: disc centre at its pixel (full_size - 1) / 2, the crop starts at an integer pixel
    for n in (875, 876):
        x0, y0 = model_grid_wcs(ref, DATE, full_size=n, full_fov=1.6 * n).world_to_pixel(
            SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective(observer="earth", obstime=DATE)))
        for c in (float(x0), float(y0)):
            assert abs((c - (n - 1) / 2 + 0.5) % 1 - 0.5) < 1e-6


def test_reprojection_equals_the_submap_of_the_full_grid():
    """Reprojecting onto the crop gives the notebook product, reproject_to(full grid).submap(corner, corner), bit for
    bit, for an instrument seen from another observer."""
    obs = SkyCoord(60 * u.deg, 3 * u.deg, 0.4 * u.AU, frame=frames.HeliographicStonyhurst(obstime=DATE))
    spot = SkyCoord(25 * u.deg, 18 * u.deg, 695700 * u.km, frame=frames.HeliographicStonyhurst(obstime=DATE))
    seen = [spot.transform_to(frames.Helioprojective(observer=o, obstime=DATE)) for o in ("earth", obs)]
    ref = pointing_map(np.zeros((30, 34)), (seen[0].Tx.to_value(u.arcsec), seen[0].Ty.to_value(u.arcsec)), 1.6, DATE)
    inst = wcs_map((48, 48), (seen[1].Tx.to_value(u.arcsec), seen[1].Ty.to_value(u.arcsec)), 1.0, observer=obs)
    wcs = model_grid_wcs(ref, DATE, full_size=875, full_fov=1400.0)
    ours = reproject_to_model_grid(inst.data, inst, wcs)
    full = sunpy_map.make_fitswcs_header((875, 875), SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective(
        observer="earth", obstime=DATE)), scale=[1.6, 1.6] * u.arcsec / u.pix)
    sub = inst.reproject_to(full, algorithm="exact").submap(ref.bottom_left_coord, top_right=ref.top_right_coord)
    assert sub.data.shape == ours.shape == (30, 34)
    assert np.isfinite(ours).sum() > 100
    assert np.array_equal(ours, sub.data, equal_nan=True)


def test_reproject_cube_shape_and_planes():
    ref = pointing_map(np.zeros((12, 14)), (300.0, 200.0), 1.6, DATE)
    inst = wcs_map((20, 20), (305.0, 205.0), 2.0)
    wcs = model_grid_wcs(ref, DATE, full_size=875, full_fov=1400.0)
    cube = np.stack([np.stack([inst.data * (1 + i + 3 * j) for i in range(3)]) for j in range(2)])
    out = reproject_to_model_grid(cube, inst, wcs, algorithm="interpolation")
    assert out.shape == (2, 3, 12, 14) and out.dtype == np.float64
    one = inst.reproject_to(wcs, algorithm="interpolation").data
    for j in range(2):
        for i in range(3):
            assert np.allclose(out[j, i], one * (1 + i + 3 * j), equal_nan=True)


def test_saved_sampling_grid_is_read_by_the_operator(tmp_path):
    model = wcs_map((24, 24), (100.0, 50.0), 1.0)
    inst = wcs_map((12, 12), (100.0, 50.0), 1.5)
    g = instrument_sampling_grid(inst, model.wcs, (24, 24), n_sub=2)
    save_sampling_grid(tmp_path / "w.npz", g, icont=3)
    with np.load(tmp_path / "w.npz") as f:
        assert sorted(f.files) == ["c0", "grid", "grid_sub", "icont", "r0", "valid"]
        assert int(f["icont"]) == 3 and np.array_equal(f["grid_sub"], g["grid_sub"])
    S = torch.randn(24 * 24, 2, dtype=torch.float64)
    a = quiet(SparseWarpOperator.from_file, tmp_path / "w.npz", input_shape=(24, 24))((S,))[0]
    b = quiet(SparseWarpOperator, g["grid_sub"], g["valid"], input_shape=(24, 24))((S,))[0]
    assert torch.equal(a, b)


# --- the 2022-03-31 products, rebuilt from the raw files -------------------------------------------------------------

def _run_2022(data_path):
    phi = instrument_map(data_path(PHI_L2 + ".fits"), crval=PHI_CRVAL)
    ref = pointing_map(np.zeros((512, 512)), HINODE_SP_CRVAL, HINODE_SP_CDELT, HINODE_SP_DATE_OBS)
    return phi, model_grid_wcs(ref, phi.date)


def test_phi_pointing_equals_the_fixed_file(data_path):
    """The hand-set pointing on the L2 header gives the WCS of the ``_fixed.fits`` the notebooks read."""
    phi = instrument_map(data_path(PHI_L2 + ".fits"), crval=PHI_CRVAL)
    fixed = sunpy_map.Map(data_path(PHI_L2 + "_fixed.fits"))
    assert phi.wcs.to_header() == fixed.wcs.to_header()
    assert np.array_equal(phi.data, fixed.data)


def test_phi_native_warp_is_reproduced(data_path, tmp_path):
    """data/phi_solo_data/phi_native_warp.npz (03_coalignment/phi_reverse_warp.ipynb), every key bit for bit."""
    stored = np.load(data_path("phi_solo_data/phi_native_warp.npz"))
    rep = fits.getdata(data_path(PHI_L2 + "_fixed_reprojected.fits"))
    phi, wcs = _run_2022(data_path)
    warp = instrument_sampling_grid(phi, wcs, (512, 512), crop_shape=(512, 512), n_sub=3, margin=2)
    save_sampling_grid(tmp_path / "w.npz", warp, icont=int(np.argmax(rep[:, 0].mean((1, 2)))))
    new = np.load(tmp_path / "w.npz")
    assert sorted(new.files) == sorted(stored.files)
    for k in stored.files:
        assert new[k].dtype == stored[k].dtype and np.array_equal(new[k], stored[k]), k


@pytest.mark.parametrize("plane", [(0, 0), (2, 3)])
def test_phi_reprojection_is_reproduced(data_path, plane):
    """The PHI cube on the model grid (03_coalignment/phi_reproject_to_earth.ipynb, ``_fixed_reprojected.fits``):
    two of the 24 planes, bit for bit (each takes ~5 s)."""
    stored = fits.getdata(data_path(PHI_L2 + "_fixed_reprojected.fits"))[plane]
    phi, wcs = _run_2022(data_path)
    cube = fits.getdata(data_path(PHI_L2 + ".fits"))[plane]
    assert np.array_equal(reproject_to_model_grid(cube, phi, wcs), stored, equal_nan=True)
