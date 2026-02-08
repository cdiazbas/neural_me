"""
Neural ME I/O Utilities.

Functions for reading and writing FITS files.
"""

import os
import numpy as np
from astropy.io import fits


def writefits(filename: str, data: np.ndarray, header=None, overwrite: bool = True):
    """
    Writes a NumPy array to a FITS file.
    
    Args:
        filename: Output path.
        data: Data to save.
        header: Optional FITS header.
        overwrite: Whether to overwrite existing file.
    """
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"File {filename} already exists.")
        
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(filename, overwrite=overwrite)


def readfits(filename: str) -> np.ndarray:
    """
    Reads data from a FITS file.
    
    Args:
        filename: Path to FITS file.
        
    Returns:
        NumPy array of data from primary HDU.
    """
    with fits.open(filename) as hdul:
        return hdul[0].data.copy()
