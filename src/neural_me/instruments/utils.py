import os
import torch
import numpy as np
from astropy.io import fits

# Find the path of the resources folder
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")

def get_resource_path(name: str) -> str:
    """Returns the absolute path of a resource file in the package."""
    path = os.path.join(RESOURCES_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resource {name} not found in {RESOURCES_DIR}")
    return path

def load_spectral_psf(name: str) -> torch.Tensor:
    """
    Loads a 1D spectral PSF profile from the package resources.
    Returns a PyTorch tensor with the PSF values (2nd column of the file).
    """
    path = get_resource_path(name)
    data = np.loadtxt(path)
    # The files contain 2 columns: [wavelength offset, intensity]
    # We return the intensity/profile values (the second column)
    if data.ndim == 2:
        return torch.from_numpy(data[:, 1])
    else:
        return torch.from_numpy(data)

def load_spatial_psf(name: str) -> torch.Tensor:
    """
    Loads a 2D spatial PSF from a FITS file in the package resources.
    Returns a 2D PyTorch tensor.
    """
    path = get_resource_path(name)
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
    return torch.from_numpy(data)
