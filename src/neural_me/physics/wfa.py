"""
Weak Field Approximation (WFA) utilities.

Provides initial condition estimation for Milne-Eddington inversions
using the Weak Field Approximation formulas.
"""

import numpy as np
from typing import Tuple

from neural_me.physics.lineas import get_line, get_line_by_wavelength

# Larmor constant
LARM = 4.668645048281451e-13


def _centered_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute centered finite difference derivative dI/dλ.

    Args:
        x: 1D wavelength array (Nw,)
        y: 3D or 4D Stokes I array (..., Nw)

    Returns:
        Derivative array with same shape as y
    """
    yp = np.zeros_like(y)
    nw = y.shape[-1]

    # Forward difference at start
    yp[..., 0] = (y[..., 1] - y[..., 0]) / (x[1] - x[0] + 1e-10)

    # Centered difference in middle
    for i in range(1, nw - 1):
        yp[..., i] = (y[..., i + 1] - y[..., i - 1]) / (x[i + 1] - x[i - 1] + 1e-10)

    # Backward difference at end
    yp[..., -1] = (y[..., -1] - y[..., -2]) / (x[-1] - x[-2] + 1e-10)

    return yp


def wfa_initial_conditions(
    wavelengths: np.ndarray,
    stokes_data: np.ndarray,
    line_id: int = 7,
    use_wavelength_lookup: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute initial Blos, BtQ, BtU estimates using Weak Field Approximation.

    This provides a physics-based initialization for the magnetic field
    components, which typically leads to faster convergence than random init.

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

    ny, nx, ns, nw = stokes_data.shape
    assert ns == 4, f"Expected 4 Stokes parameters, got {ns}"

    # Extract Stokes components
    I = stokes_data[:, :, 0, :]  # (Ny, Nx, Nw)
    Q = stokes_data[:, :, 1, :]
    U = stokes_data[:, :, 2, :]
    V = stokes_data[:, :, 3, :]

    # Compute dI/dλ
    dIdw = _centered_derivative(wavelengths, I)

    # WFA constant: C = -4.67e-13 * λ₀²
    C = -4.67e-13 * line.cw**2

    # Flatten spatial dimensions for vectorized computation
    V_flat = V.reshape(ny * nx, nw)
    Q_flat = Q.reshape(ny * nx, nw)
    U_flat = U.reshape(ny * nx, nw)
    dIdw_flat = dIdw.reshape(ny * nx, nw)

    # === Blos from Stokes V ===
    # Blos = Σ(V * dI/dλ) / (C * geff * Σ(dI/dλ)²)
    numerator = np.sum(V_flat * dIdw_flat, axis=-1)
    denominator = C * line.geff * np.sum(dIdw_flat**2, axis=-1)
    Blos = numerator / (denominator + 1e-12)

    # === Btrans from Stokes Q, U ===
    # Need to avoid line core where dI/dλ ~ 0
    # Use scaling: scl = 1/|λ| for |λ| > vdop, else 0
    vdop = 0.035  # Typical Doppler width in Å
    scl = np.where(np.abs(wavelengths) > vdop, 1.0 / (np.abs(wavelengths) + 1e-10), 0.0)
    dIdw_scl = dIdw_flat * scl

    # BtQ² ∝ Σ(Q * dI/dλ_scaled) / (C² * Gg * Σ(dI/dλ_scaled)²)
    denom_t = 0.75 * C**2 * line.Gg * np.sum(dIdw_scl**2, axis=-1)

    BtQ_sq = np.sum(Q_flat * dIdw_scl, axis=-1) / (denom_t + 1e-12)
    BtQ = np.sign(BtQ_sq) * np.sqrt(np.abs(BtQ_sq))

    BtU_sq = np.sum(U_flat * dIdw_scl, axis=-1) / (denom_t + 1e-12)
    BtU = np.sign(BtU_sq) * np.sqrt(np.abs(BtU_sq))

    # Reshape back to (Ny, Nx)
    Blos = Blos.reshape(ny, nx).astype(np.float32)
    BtQ = BtQ.reshape(ny, nx).astype(np.float32)
    BtU = BtU.reshape(ny, nx).astype(np.float32)

    return Blos, BtQ, BtU


def wfa_thermodynamic_estimates(
    stokes_data: np.ndarray, continuum_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate thermodynamic parameters S0 and S1 from Stokes I.

    Args:
        stokes_data: 4D array (Ny, Nx, 4, Nw) of Stokes [I, Q, U, V]
        continuum_idx: Index of continuum wavelength point

    Returns:
        Tuple of (S0, S1): Each is 2D array (Ny, Nx)
    """
    I = stokes_data[:, :, 0, :]  # (Ny, Nx, Nw)

    # S0 ≈ min(I) (line core)
    S0 = np.min(I, axis=-1).astype(np.float32)

    # S1 ≈ max(I) - min(I) (continuum - core)
    S1 = (np.max(I, axis=-1) - S0).astype(np.float32)

    return S0, S1
