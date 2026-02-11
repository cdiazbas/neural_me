"""
Centralized spectral line database.

This module provides the single source of truth for spectral line
parameters used across the neural_me library. All line-related lookups
should import from this module.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass


# Lande factor calculation
def _g(j: float, l: float, s: float) -> float:
    """
    Calculate Landé g-factor.
    """
    if j == 0:
        return 0.0
    return 1.5 + (s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


def _geff(ju: float, gu: float, jl: float, gl: float) -> float:
    """
    Calculate effective Landé factor.
    """
    return 0.5 * (gu + gl) + 0.25 * (gl - gu) * (jl * (jl + 1) - ju * (ju + 1))


def _Gg(geff: float, gu: float, gl: float, ju: float, jl: float) -> float:
    """
    Calculate Gg factor for transverse field WFA.
    """
    ss = ju * (ju + 1) + jl * (jl + 1)
    dd = ju * (ju + 1) - jl * (jl + 1)
    gd = gu - gl
    return geff**2 - 0.0125 * gd**2 * (16 * ss - 7 * dd**2 - 4)


@dataclass
class SpectralLineData:
    """
    Container for spectral line data.
    """

    line_id: int
    element: str
    cw: float  # Central wavelength (Å)
    ju: float  # Upper level J
    jl: float  # Lower level J
    lu: float  # Upper level L
    ll: float  # Lower level L
    su: float  # Upper level S
    sl: float  # Lower level S
    gu: float  # Upper level g-factor
    gl: float  # Lower level g-factor
    geff: float  # Effective Landé factor
    Gg: float  # Gg factor for WFA


def _parse_lineas_file() -> Dict[int, SpectralLineData]:
    """
    Parse LINEAS.txt and return a dictionary of SpectralLineData.

    LINEAS.txt format:
    ID = ELEM  WAVELENGTH  ...  TERM_L- TERM_U  ...
    Example: 7 = FE 1   6173.3356  1.0  2.223    -2.880   5P 1.0- 5D 0.0  ...
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lineas_path = os.path.join(current_dir, "LINEAS.txt")

    lines_dict = {}
    l_notation = ["S", "P", "D", "F"]

    with open(lineas_path, "r") as f:
        for line in f:
            parts = line.split()
            line_id = int(parts[0])

            # Element
            element = "".join(parts[2:4])

            # Central wavelength
            cw = float(parts[4])

            # Parse quantum numbers from term symbols
            # Format: "5P 1.0- 5D 0.0" -> upper=5D 0.0, lower=5P 1.0
            # parts[8] = lower term (e.g., "5P"), parts[9] = lower J (e.g., "1.0-")
            # parts[10] = upper term (e.g., "5D"), parts[11] = upper J (e.g., "0.0")

            jl = float(parts[9].rstrip("-"))
            ju = float(parts[11].rstrip("-"))

            ll = l_notation.index(parts[8][1])
            lu = l_notation.index(parts[10][1])

            sl = (int(parts[8][0]) - 1) / 2
            su = (int(parts[10][0]) - 1) / 2

            # Calculate derived quantities
            gu = _g(ju, lu, su)
            gl = _g(jl, ll, sl)
            geff = _geff(ju, gu, jl, gl)
            Gg = _Gg(geff, gu, gl, ju, jl)

            lines_dict[line_id] = SpectralLineData(
                line_id=line_id,
                element=element,
                cw=cw,
                ju=ju,
                jl=jl,
                lu=lu,
                ll=ll,
                su=su,
                sl=sl,
                gu=gu,
                gl=gl,
                geff=geff,
                Gg=Gg,
            )

    return lines_dict


# Global line database (lazy loaded)
_LINEAS_CACHE: Optional[Dict[int, SpectralLineData]] = None


def get_lineas() -> Dict[int, SpectralLineData]:
    """
    Get the spectral line database (cached).
    """
    global _LINEAS_CACHE
    if _LINEAS_CACHE is None:
        _LINEAS_CACHE = _parse_lineas_file()
    return _LINEAS_CACHE


def get_line(line_id: int) -> SpectralLineData:
    """
    Get data for a specific spectral line.

    Args:
        line_id: Line identifier (1-7 for current LINEAS.txt)

    Returns:
        SpectralLineData object with all line parameters

    Raises:
        KeyError: If line_id is not found
    """
    lineas = get_lineas()
    if line_id not in lineas:
        available = list(lineas.keys())
        raise KeyError(f"Line ID {line_id} not found. Available: {available}")
    return lineas[line_id]


# Convenience aliases mapping common wavelengths to line IDs
LINE_ID_BY_WAVELENGTH = {
    6301: 1,  # Fe I 6301.5
    6302: 2,  # Fe I 6302.5
    10827: 3,  # Si I 10827
    10833: 4,  # Ca I 10833
    10839: 5,  # Ca I 10839
    15648: 6,  # Fe I 15648.5
    6173: 7,  # Fe I 6173.3 (SDO/HMI, SolO/PHI)
}


def get_line_by_wavelength(wavelength: int) -> SpectralLineData:
    """
    Get line data by approximate central wavelength.

    Args:
        wavelength: Approximate wavelength in Å (e.g., 6173 for Fe I 6173.3)

    Returns:
        SpectralLineData object
    """
    if wavelength not in LINE_ID_BY_WAVELENGTH:
        raise KeyError(
            f"Wavelength {wavelength} not recognized. Available: {list(LINE_ID_BY_WAVELENGTH.keys())}"
        )
    return get_line(LINE_ID_BY_WAVELENGTH[wavelength])


# For backward compatibility, expose as module-level dict
# Usage: from neural_me.physics.lineas import LINEAS; LINEAS[7]
class _LineasDict(dict):
    """
    Lazy-loading dict wrapper for LINEAS.
    """

    def __getitem__(self, key):
        return get_line(key)

    def __contains__(self, key):
        return key in get_lineas()

    def keys(self):
        return get_lineas().keys()


LINEAS = _LineasDict()
