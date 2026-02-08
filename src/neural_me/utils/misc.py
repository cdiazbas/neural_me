
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import astropy.io.fits as fits

class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def writefits(name, d):
    io = fits.PrimaryHDU(d)
    io.writeto(name, overwrite=True)


def add_colorbar(im, aspect=20, pad_fraction=0.5, nbins=5, **kwargs):
    """Add a vertical color bar to an image plot."""
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    from matplotlib import ticker
    cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    tick_locator = ticker.MaxNLocator(nbins)
    cb.locator = tick_locator
    cb.update_ticks()
    return cb

from neural_me.regularization.spatial import (
    smoothness_loss, 
    target_value_regularization
)

def regu(out, ii, img):
    """
    Legacy wrapper for L1 smoothness regularization.
    
    Args:
        out: Tensor of shape (N_pixels, N_params)
        ii: Parameter index to regularize
        img: Reference image tensor (used for shape information: (Any, Ny, Nx, ...))
    """
    # Infer grid shape from img: (Ny, Nx)
    # img is typically (samples, Ny, Nx, ...) in legacy code
    ny, nx = img.shape[1], img.shape[2]
    
    # Extract parameter map and reshape to (1, 1, Ny, Nx) for convolution
    # out is (Ny*Nx, N_params)
    param_map = out[:, ii].view(1, 1, ny, nx)
    
    return smoothness_loss(param_map, penalty='l1')


def regu2(out, ii, img):
    """
    Legacy wrapper for L2 smoothness regularization.
    """
    ny, nx = img.shape[1], img.shape[2]
    param_map = out[:, ii].view(1, 1, ny, nx)
    return smoothness_loss(param_map, penalty='l2')

def regu_value(out, ii, img, value):
    """
    Legacy wrapper for value target regularization.
    """
    # No reshaping needed for simple differences, but keeping logic consistent
    param_vals = out[:, ii]
    return target_value_regularization(param_vals, value)
