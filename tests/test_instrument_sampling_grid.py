"""instrument_sampling_grid: geometry of one instrument on the model grid, from the WCS alone."""
import numpy as np
import pytest
import torch

sunpy_map = pytest.importorskip("sunpy.map")
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import frames

from neural_me.core.geometry import instrument_sampling_grid
from neural_me.optimization import SparseWarpOperator

DATE = "2022-03-31T19:39:59"


def make_map(shape, centre, scale):
    ref = SkyCoord(centre[0] * u.arcsec, centre[1] * u.arcsec, frame=frames.Helioprojective(observer="earth", obstime=DATE))
    header = sunpy_map.make_fitswcs_header(np.zeros(shape), ref, scale=[scale, scale] * u.arcsec / u.pix)
    return sunpy_map.Map(np.zeros(shape), header)


def test_same_wcs_is_the_identity():
    m = make_map((20, 24), (100, 50), 1.0)
    g = instrument_sampling_grid(m, m.wcs, (20, 24), margin=0.5)
    y, x = np.mgrid[0:20, 0:24]
    assert np.allclose((g["grid"][..., 0] + 1) / 2 * 23, x, atol=1e-3)
    assert np.allclose((g["grid"][..., 1] + 1) / 2 * 19, y, atol=1e-3)
    assert g["footprint"] == pytest.approx([1.0, 1.0], abs=1e-3)
    assert g["valid"].sum() == 18 * 22 and g["r0"] == 0 and g["c0"] == 0      # margin 0.5: centres in 1..18, 1..22


def test_coarser_instrument_footprint_and_subsamples():
    """Instrument pixels of 2'' on a 1'' model: footprint 2 x 2 and samples at +-2/3 model px around the centre."""
    model = make_map((60, 60), (100, 50), 1.0)
    inst = make_map((20, 20), (100, 50), 2.0)
    g = instrument_sampling_grid(inst, model.wcs, (60, 60))
    assert g["footprint"] == pytest.approx([2.0, 2.0], abs=1e-2)
    assert g["valid"].all()
    sub = (g["grid_sub"] + 1) / 2 * 59
    centre = (g["grid"] + 1) / 2 * 59
    assert np.allclose(sub.mean(2), centre, atol=1e-3)
    assert sub.shape == (20, 20, 9, 2)
    assert np.allclose(sub[10, 10, 8] - sub[10, 10, 0], [4 / 3, 4 / 3], atol=1e-3)


def test_crop_is_centred_and_inside_and_the_operator_reads_it():
    model = make_map((16, 16), (100, 50), 1.0)
    inst = make_map((64, 64), (100, 50), 1.0)
    g = instrument_sampling_grid(inst, model.wcs, (16, 16), crop_shape=(24, 24), n_sub=1)
    assert (g["r0"], g["c0"]) == (19, 19)                     # model on pixels 24..39, centre 31, minus 12
    # a shift between instrument and model: an output pixel holds the model value it falls on
    op = SparseWarpOperator(g["grid_sub"], g["valid"], input_shape=(16, 16))
    S = torch.arange(16 * 16, dtype=torch.float64).reshape(-1, 1)
    out = op((S,))[0].reshape(24, 24).numpy()
    ok = g["valid"]
    cy, cx = (g["grid"][..., 1] + 1) / 2 * 15, (g["grid"][..., 0] + 1) / 2 * 15
    assert ok.sum() > 0 and np.allclose(out[ok], (np.round(cy) * 16 + np.round(cx))[ok], atol=1e-2)
    assert (out[~ok] == 0).all()


def test_pixels_off_the_disc_are_not_valid():
    """An instrument over the limb: NaN in the WCS chain never reaches the grid."""
    model = make_map((30, 30), (900, 0), 4.0)
    # transform via another observer so that off-disc pixels have no point on the sphere
    header = sunpy_map.make_fitswcs_header(np.zeros((60, 60)),
                                           SkyCoord(900 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective(
                                               observer=SkyCoord(30 * u.deg, 0 * u.deg, 1 * u.AU, frame=frames.HeliographicStonyhurst, obstime=DATE),
                                               obstime=DATE)),
                                           scale=[4, 4] * u.arcsec / u.pix)
    inst = sunpy_map.Map(np.zeros((60, 60)), header)
    g = instrument_sampling_grid(inst, model.wcs, (30, 30))
    assert np.isfinite(g["grid"]).all() and np.isfinite(g["grid_sub"]).all()
    assert 0 < g["valid"].sum() < 60 * 60


def test_no_overlap_raises():
    with pytest.raises(ValueError, match="no pixel"):
        instrument_sampling_grid(make_map((8, 8), (800, 0), 1.0), make_map((8, 8), (-800, 0), 1.0).wcs, (8, 8))
