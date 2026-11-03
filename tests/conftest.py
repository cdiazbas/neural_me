"""
Shared test fixtures.

The local-vertical tests need a concrete on-disk pointing to exercise.  The
Hinode/SP scan of 2022-03-31 is used because it is the reference view of the
stereoscopic run these modules were written against — it is a *test fixture*,
not a supported configuration.  :mod:`neural_me.core.geometry` is agnostic to
instrument and pointing; anything specific to one dataset belongs here.
"""

import numpy as np
import pytest


# Pointing used to build the shared (warped) grid of the 2022-03-31 run, taken
# from analysis/orientation.ipynb.  The FITS files on disk have had their
# headers stripped, so these are the numbers that define the grid.
HINODE_SP_CRVAL = (520.0 - 9.0, 320.0 + 9.0)  # arcsec, helioprojective
HINODE_SP_CDELT = 0.16  # arcsec / pixel
HINODE_SP_DATE_OBS = "2022-03-31T19:39:59"
HINODE_SP_B0_DEG = -6.57  # solar B0 at that date


def make_hinode_sp_map(
    grid_shape,
    crval=HINODE_SP_CRVAL,
    cdelt=HINODE_SP_CDELT,
    binning=1.0,
    date_obs=HINODE_SP_DATE_OBS,
):
    """Rebuild the sunpy Map for the 2022-03-31 Hinode/SP grid.

    Constructs the header literally rather than via ``make_fitswcs_header`` so
    that the ``CRPIX = n / 2`` convention of analysis/orientation.ipynb is
    reproduced exactly.  No observer keywords are given, so sunpy places the
    observer at Earth; Hinode is in low Earth orbit, which is good to well
    under an arcsecond.

    Args:
        grid_shape: ``(ny, nx)``.
        crval: Helioprojective ``(Tx, Ty)`` in arcsec at the reference pixel.
        cdelt: Native plate scale in arcsec/pixel, before decimation.
        binning: Decimation factor (the drivers' ``skip_factor``); scales cdelt.
        date_obs: Observation date, ISO format.
    """
    import sunpy.map
    from astropy.io.fits import Header

    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    meta = Header()
    meta["NAXIS"] = 2
    meta["NAXIS1"] = nx
    meta["NAXIS2"] = ny
    meta["CRPIX1"] = nx / 2  # FITS is 1-based; matches orientation.ipynb
    meta["CRPIX2"] = ny / 2
    meta["CRVAL1"] = crval[0]
    meta["CRVAL2"] = crval[1]
    meta["CDELT1"] = cdelt * binning
    meta["CDELT2"] = cdelt * binning
    meta["CTYPE1"] = "HPLN-TAN"
    meta["CTYPE2"] = "HPLT-TAN"
    meta["CUNIT1"] = "arcsec"
    meta["CUNIT2"] = "arcsec"
    meta["INSTRUME"] = "SP"
    meta["TELESCOP"] = "SP/Hinode"
    meta["WAVELNTH"] = 6302.0
    meta["WAVEUNIT"] = "angstrom"
    meta["DATE-OBS"] = date_obs

    return sunpy.map.Map(np.zeros((ny, nx)), meta)


@pytest.fixture
def hinode_sp_map():
    """Factory fixture: ``hinode_sp_map(grid_shape, **kwargs)``."""
    return make_hinode_sp_map
