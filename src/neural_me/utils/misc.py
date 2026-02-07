
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

def regu(out, ii, img):
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[-1]))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(mout[:,:,:,ii]-output_smooth))


def regu2(out, ii,img):
    # Squared diference
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[-1]))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(output_smooth-mout[:,:,:,ii])**2.0)

def regu_value(out, ii,img, value):
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],out.shape[-1]))
    return torch.sum(torch.abs(mout[:,:,:,ii]-value))
