"""
Center of Gravity (COG) and Moment method utilities.

Provides initial condition estimation for Milne-Eddington inversions.
"""

import numpy as np
from typing import Tuple

from neural_me.physics.lineas import get_line, get_line_by_wavelength

# Larmor constant
LARM = 4.668645048281451e-13


def cog_initial_conditions(
    wavelengths: np.ndarray,
    stokes_data: np.ndarray,
    line_id: int = 7,
    use_wavelength_lookup: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute initial Blos, BtQ, BtU estimates using the Center of Gravity
    and Moment methods.

    Args:
        wavelengths: 1D array of wavelengths (Å, relative to line center)
        stokes_data: 4D array (Ny, Nx, 4, Nw) of Stokes [I, Q, U, V]
        line_id: Spectral line identifier (1-7, or wavelength if use_wavelength_lookup=True)
        use_wavelength_lookup: If True, treat line_id as wavelength (e.g., 6173)

    Returns:
        Tuple of (Blos, BtQ, BtU): Each is 2D array (Ny, Nx) in Gauss
    """
    # Get line data
    if use_wavelength_lookup:
        line = get_line_by_wavelength(line_id)
    else:
        line = get_line(line_id)

    spatial_shape = stokes_data.shape[:-2]
    ns = stokes_data.shape[-2]
    nw = stokes_data.shape[-1]
    assert ns == 4, f"Expected 4 Stokes parameters, got {ns}"

    # Flatten spatial dimensions first
    stokes_flat = stokes_data.reshape(-1, ns, nw)
    I_flat = stokes_flat[:, 0, :]
    Q_flat = stokes_flat[:, 1, :]
    U_flat = stokes_flat[:, 2, :]
    V_flat = stokes_flat[:, 3, :]

    # Constants
    C = -4.67e-13 * line.cw**2
    lB = LARM * line.cw**2

    # Continuum intensity estimate per pixel (max intensity along wavelength)
    Ic = np.max(I_flat, axis=-1, keepdims=True)

    # 1. COG for Blos
    term_plus = Ic - (I_flat + V_flat)
    term_minus = Ic - (I_flat - V_flat)
    
    # Avoid division by zero in COG
    sum_plus = np.sum(term_plus, axis=-1)
    sum_minus = np.sum(term_minus, axis=-1)
    
    lambda_plus = np.sum(wavelengths * term_plus, axis=-1) / (sum_plus + 1e-12)
    lambda_minus = np.sum(wavelengths * term_minus, axis=-1) / (sum_minus + 1e-12)
    Blos = (lambda_plus - lambda_minus) / (2.0 * line.geff * lB + 1e-12)

    # 2. Moment method for Btrans (BtQ, BtU)
    Q_int = np.sum(wavelengths**2 * Q_flat, axis=-1)
    U_int = np.sum(wavelengths**2 * U_flat, axis=-1)
    EqW = np.sum(Ic - I_flat, axis=-1)

    # Factor: 1.5 * C**2 * Gg
    factor = 1.5 * C**2 * line.Gg

    BtQ_sq = Q_int / (factor * EqW + 1e-12)
    BtU_sq = U_int / (factor * EqW + 1e-12)

    # Compute transverse field magnitude
    Bt_trans = (BtQ_sq**2 + BtU_sq**2)**0.25

    # Compute BtQ and BtU (in Gauss)
    BtQ = BtQ_sq / (Bt_trans + 1e-12)
    BtU = BtU_sq / (Bt_trans + 1e-12)

    # Reshape back to original spatial shape
    Blos = Blos.reshape(spatial_shape).astype(np.float32)
    BtQ = BtQ.reshape(spatial_shape).astype(np.float32)
    BtU = BtU.reshape(spatial_shape).astype(np.float32)

    return Blos, BtQ, BtU
