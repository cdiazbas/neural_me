"""
Neural ME Visualization Utilities.

Plotting functions for Milne-Eddington inversion results.
Ported from neural_wfa with additions for ME-specific visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import torch
from typing import Optional, Tuple, List, Union

# =============================================================================
# Legacy Plot Parameters (for visual parity with reference implementations)
# =============================================================================

LEGACY_PARAMS = {
    "axes.labelsize": 19.0,
    "axes.linewidth": 1.8,
    "axes.titlesize": 18.0,
    "figure.figsize": [6.15, 5.0],
    "font.size": 10.0,
    "legend.fontsize": 17.0,
    "legend.frameon": False,
    "lines.linewidth": 2.5,
    "savefig.bbox": "tight",
    "text.usetex": False,  # Disable by default for compatibility
    "xtick.labelsize": 17.5,
    "xtick.major.size": 14.0,
    "xtick.minor.visible": True,
    "ytick.labelsize": 17.5,
    "ytick.major.size": 14.0,
    "ytick.minor.visible": True,
    "image.interpolation": "nearest",
    "image.origin": "lower",
}


def set_plot_params(params: dict = None, use_tex: bool = False):
    """
    Sets default matplotlib parameters.
    
    Args:
        params: Custom parameters dict. If None, uses LEGACY_PARAMS.
        use_tex: If True, enables LaTeX rendering.
    """
    if params is None:
        params = LEGACY_PARAMS.copy()
    
    if use_tex:
        params["text.usetex"] = True
        params["text.latex.preamble"] = r"\usepackage[T1]{fontenc} \usepackage{newcent} \boldmath"
        
    pylab.rcParams.update(params)


# Alias for legacy compatibility
set_params = set_plot_params


# =============================================================================
# Helper Functions
# =============================================================================

def torch2numpy(tensor: Union[torch.Tensor, np.ndarray], shape=None) -> np.ndarray:
    """
    Converts a torch tensor to a numpy array and optionally reshapes it.
    
    Args:
        tensor: Input tensor or array.
        shape: Optional new shape.
        
    Returns:
        NumPy array.
    """
    if hasattr(tensor, 'detach'):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)
        
    if shape:
        return arr.reshape(shape)
    return arr


def format_scientific(value: float, precision: int = 2) -> str:
    """
    Formats a value into LaTeX scientific notation.
    
    Args:
        value: Number to format.
        precision: Decimal places.
        
    Returns:
        LaTeX string.
    """
    if value == 0:
        return "0"
    formatted = f"{value:.{precision}e}".replace("e", r" \times 10^{") + "}"
    return f"${formatted}$"


def add_colorbar(im, aspect=20, pad_fraction=0.5, nbins=5, 
                 orientation='vertical', **kwargs):
    """
    Add a color bar to an image plot.
    
    Args:
        im: Matplotlib image object.
        aspect: Aspect ratio.
        pad_fraction: Padding fraction.
        nbins: Number of tick bins.
        orientation: 'vertical' or 'horizontal'.
        
    Returns:
        Colorbar object.
    """
    from mpl_toolkits import axes_grid1
    from matplotlib import ticker

    divider = axes_grid1.make_axes_locatable(im.axes)
    
    if orientation.lower() == 'horizontal':
        width = axes_grid1.axes_size.AxesX(im.axes, aspect=aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("bottom", size=width, pad=pad)
        plt.sca(current_ax)
    else:  # vertical
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
    
    cb = im.axes.figure.colorbar(im, cax=cax, orientation=orientation, **kwargs)
    tick_locator = ticker.MaxNLocator(nbins)
    cb.locator = tick_locator
    cb.update_ticks()
    return cb


# =============================================================================
# Training / Loss Plotting
# =============================================================================

def plot_loss(losses: Union[List[float], np.ndarray], lr_values: List[float] = None,
              save_name: str = None, show: bool = True):
    """
    Plot training loss and optionally learning rate.
    
    Args:
        losses: List/array of loss values per iteration.
        lr_values: Optional list of learning rates.
        save_name: If provided, saves to this path.
        show: If True, displays the plot.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    losses = np.array(losses)
    ax1.semilogy(losses, alpha=0.5, color='C0')
    
    # Savgol smoothing
    if len(losses) > 10:
        try:
            from scipy.signal import savgol_filter
            window = max(5, int(len(losses) / 10))
            if window % 2 == 0:
                window += 1
            smoothed = savgol_filter(losses, window, min(3, window - 1))
            ax1.semilogy(smoothed, 'C0-', alpha=0.8)
        except ImportError:
            pass
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Final loss: {losses[-1]:.2e}')
    ax1.minorticks_on()
    
    # LR on secondary axis
    if lr_values is not None and len(lr_values) > 0:
        ax2 = ax1.twinx()
        ax2.semilogy(lr_values, 'k--', alpha=0.5)
        ax2.set_ylabel('Learning rate')
    
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


plot_training_progress = plot_loss  # Alias


# =============================================================================
# WFA Results Plotting
# =============================================================================

def plot_wfa_results(blos: np.ndarray, btrans: np.ndarray, phi: np.ndarray,
                     save_name: str = None, show: bool = True):
    """
    Plots Blos, Btrans, and Azimuth maps.
    
    Args:
        blos: (ny, nx) LOS field in Gauss.
        btrans: (ny, nx) Transverse field in Gauss.
        phi: (ny, nx) Azimuth in radians.
        save_name: Optional save path.
        show: If True, displays plot.
    """
    ny, nx = blos.shape
    extent = (0, nx, 0, ny)
    
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.75))
    
    im0 = axes[0].imshow(blos, vmin=-800, vmax=800, cmap='RdGy', extent=extent)
    im1 = axes[1].imshow(btrans, vmin=0, vmax=800, cmap='gist_gray', extent=extent)
    im2 = axes[2].imshow(phi, vmin=0, vmax=np.pi, cmap='twilight', extent=extent)
    
    fig.colorbar(im0, ax=axes[0], orientation='horizontal', label=r'B$_\parallel$ [G]', pad=0.15)
    fig.colorbar(im1, ax=axes[1], orientation='horizontal', label=r'B$_\bot$ [G]', pad=0.15)
    fig.colorbar(im2, ax=axes[2], orientation='horizontal', label=r'$\Phi_B$ [rad]', pad=0.15)
    
    for i in range(3):
        axes[i].set_xlabel('x [pixels]')
        axes[i].minorticks_on()
    axes[0].set_ylabel('y [pixels]')
    for i in range(1, 3):
        axes[i].set_yticklabels([])
    
    fig.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


# =============================================================================
# Stokes Profile Comparison
# =============================================================================

def plot_stokes_profiles(wavelengths: np.ndarray, 
                         obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                         mod: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                         pixel_idx: int = 0,
                         save_name: str = None, show: bool = True):
    """
    Plots observed vs modeled Stokes profiles for a single pixel.
    
    Args:
        wavelengths: Wavelength array.
        obs: Tuple (I, Q, U, V) of observed profiles, each (n_pixels, n_wav).
        mod: Tuple (I, Q, U, V) of modeled profiles.
        pixel_idx: Which pixel to plot.
        save_name: Optional save path.
        show: If True, displays plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    labels = ['I', 'Q', 'U', 'V']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        obs_data = torch2numpy(obs[i])
        mod_data = torch2numpy(mod[i])
        
        if obs_data.ndim > 1:
            obs_data = obs_data[pixel_idx]
            mod_data = mod_data[pixel_idx]
        
        ax.plot(wavelengths, obs_data, 'ko', markersize=4, label='Observed')
        ax.plot(wavelengths, mod_data, 'C0-', linewidth=1.5, label='Model')
        ax.set_ylabel(f'Stokes {label}')
        ax.set_xlabel(r'$\Delta\lambda$ [Å]')
        ax.legend()
        ax.minorticks_on()
    
    fig.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


# =============================================================================
# ME Inversion Results - 8-Panel Plot (Legacy Parity)
# =============================================================================

def plot_me_results(outplot: np.ndarray, 
                    save_name: str = None, 
                    show: bool = True,
                    interactive: bool = False,
                    stokes_data=None,
                    stokes_syn=None):
    """
    8-panel Milne-Eddington inversion result plot matching legacy exactly.
    
    Args:
        outplot: Array of shape (ny, nx, 8) with parameters:
                 [0] B (kG), [1] theta_B (rad), [2] phi_B (rad), [3] vlos (km/s)
                 [4] eta0, [5] damp, [6] Ddop, [7] S0+S1
        save_name: Optional save path.
        show: If True, displays plot.
        interactive: If True, enables onclick for profile inspection.
        stokes_data: Original Stokes data for interactive mode (ny, nx, 4, nw).
        stokes_syn: Synthetic Stokes for interactive mode.
        
    Returns:
        Figure and axes for further customization.
    """
    ny, nx = outplot.shape[:2]
    extent = (0, nx, 0, ny)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.5))
    axes = axes.flatten()
    
    # Parameter names and colormaps (matching legacy exactly)
    names = [
        r'B [kG]', r'$\Theta_B$ [rad]', r'$\Phi_B$ [rad]', r'v$_{\rm los}$ [km/s]',
        r'$\eta_0$ [1]', r'damp [1]', r'Ddop [1]', r'S0+S1 [au]'
    ]
    cmaps = [
        'cubehelix', 'PRGn', 'twilight', 'seismic',
        'cubehelix', 'cubehelix', 'cubehelix', 'cubehelix'
    ]
    
    for i in range(8):
        im = axes[i].imshow(outplot[:, :, i], cmap=cmaps[i], 
                           interpolation='nearest', extent=extent, origin='lower')
        fig.colorbar(im, ax=axes[i], orientation='horizontal', label=names[i])
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
    
    fig.tight_layout()
    
    # Interactive onclick handler
    if interactive and stokes_data is not None:
        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            ix, iy = int(np.floor(event.xdata)), int(np.floor(event.ydata))
            if 0 <= ix < nx and 0 <= iy < ny:
                _plot_pixel_profiles(stokes_data, stokes_syn, iy, ix, nx)
        
        fig.canvas.mpl_connect('button_press_event', onclick)
    
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, axes


def _plot_pixel_profiles(data: np.ndarray, syn: np.ndarray, iy: int, ix: int, nx: int):
    """Helper: Plot Stokes profiles for a clicked pixel."""
    plt.figure(figsize=(10, 6))
    axes = [plt.subplot(2, 2, i+1) for i in range(4)]
    
    labels = ['I', 'Q', 'U', 'V']
    colors = ['C0', 'C1', 'C2', 'C3']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(data[iy, ix, i, :], 'o-', label=f'Obs {label}', color='k', markersize=3)
        if syn is not None:
            pixel_idx = iy * nx + ix
            ax.plot(syn[i][pixel_idx].detach().numpy() if hasattr(syn[i], 'detach') else syn[i][pixel_idx], 
                   '-', label=f'Model {label}', color=color)
        ax.set_ylabel(f'Stokes {label}')
        ax.legend(fontsize=8)
        ax.minorticks_on()
    
    plt.suptitle(f'Pixel (x={ix}, y={iy})')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Chi2 / Residual Maps
# =============================================================================

def plot_chi2_maps(chi2_q: np.ndarray, chi2_u: np.ndarray, chi2_v: np.ndarray,
                   chi2_total: np.ndarray = None,
                   save_name: str = None, show: bool = True):
    """
    Plots spatial Chi2 maps for Q, U, V components.
    
    Args:
        chi2_q, chi2_u, chi2_v: (ny, nx) arrays.
        chi2_total: Optional total Chi2 map.
        save_name: Optional save path.
        show: If True, displays plot.
    """
    ny, nx = chi2_q.shape
    extent = (0, nx, 0, ny)
    
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 7), sharex=True, sharey=True)
    
    for ax, data, label in zip(axes, [chi2_q, chi2_u, chi2_v], [r'$\chi^2_Q$', r'$\chi^2_U$', r'$\chi^2_V$']):
        im = ax.imshow(data, cmap='magma', extent=extent)
        add_colorbar(im, orientation='horizontal', pad_fraction=7.5, nbins=3, label=f'{label} [a.u.]')
        ax.set_xlabel('x [pixels]')
        ax.minorticks_on()
    
    axes[0].set_ylabel('y [pixels]')
    fig.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


# =============================================================================
# Uncertainty Maps
# =============================================================================

def plot_uncertainties(unc_blos: np.ndarray, unc_btrans: np.ndarray, unc_phi: np.ndarray,
                       save_name: str = None, show: bool = True):
    """
    Plots uncertainty maps for magnetic field components.
    
    Args:
        unc_blos, unc_btrans, unc_phi: (ny, nx) uncertainty arrays.
        save_name: Optional save path.
        show: If True, displays plot.
    """
    ny, nx = unc_blos.shape
    extent = (0, nx, 0, ny)
    
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.75))
    
    im0 = axes[0].imshow(unc_blos, extent=extent, cmap='gray', vmax=100)
    im1 = axes[1].imshow(unc_btrans, extent=extent, cmap='gray', vmax=1000)
    im2 = axes[2].imshow(unc_phi, extent=extent, cmap='gray', vmax=3)
    
    fig.colorbar(im0, ax=axes[0], orientation='horizontal', label=r'$\Delta$B$_\parallel$ [G]', pad=0.15)
    fig.colorbar(im1, ax=axes[1], orientation='horizontal', label=r'$\Delta$B$_\bot$ [G]', pad=0.15)
    fig.colorbar(im2, ax=axes[2], orientation='horizontal', label=r'$\Delta\Phi_B$ [rad]', pad=0.15)
    
    for i in range(3):
        axes[i].set_xlabel('x [pixels]')
        axes[i].minorticks_on()
    axes[0].set_ylabel('y [pixels]')
    for i in range(1, 3):
        axes[i].set_yticklabels([])
    
    fig.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
