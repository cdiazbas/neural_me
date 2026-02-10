"""
Physical Constants and Normalization Factors for Milne-Eddington Inversion.

This module provides the single source of truth for all normalization factors
used throughout the neural_me package.
"""

# Default normalization factors for Milne-Eddington parameters
DEFAULT_NORMALIZATION = {
    # Magnetic field components (Gauss)
    'B': 500.0,      # Total magnetic field strength
    'Blos': 500.0,   # Line-of-sight component
    'BQ': 500.0,     # Stokes Q component
    'BU': 500.0,     # Stokes U component
    
    # Magnetic field angles (radians)
    'gamma': 1.0,    # Inclination angle [0, π]
    'xi': 1.0,       # Azimuth angle [0, 2π]
    
    # Thermodynamic parameters
    'vlos': 2.0,     # Line-of-sight velocity (km/s)
    'eta0': 100.0,   # Line-to-continuum absorption ratio
    'a': 0.2,        # Damping parameter
    'ddop': 1.0,     # Doppler width (Angstrom)
    
    # Source function parameters
    'S0': 1.0,       # Source function constant
    'S1': 1.0        # Source function gradient
}

# Initial values for parameters
DEFAULT_PARAMETERS = {
    # Magnetic field 
    'Blos': 500., 
    'BQ': 500., 
    'BU': 500.,
    'B': 500.,
    'gamma': 1.0,
    'xi': 1.0,
    
    # Thermodynamic
    'vlos': 0.0,
    'eta0': 50.0,
    'a': 0.2,
    'ddop': 0.02,
    
    # Source function
    'S0': 0.5,
    'S1': 0.5
}
