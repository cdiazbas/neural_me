"""
Milne-Eddington synthesis with multiple backend support.

This module provides a unified interface for Milne-Eddington synthesis
with automatic backend selection (PyTorch or C++).
"""


def MilneEddington(nlinea=None, backend='torch', **kwargs):
    """
    Create a Milne-Eddington synthesis model.
    
    Factory function that returns the appropriate backend implementation.
    
    Args:
        nlinea: Line identifier (e.g., 1 for Fe I 6301.5)
        backend: 'torch' (default) or 'cpp'
        **kwargs: Additional arguments for C++ backend:
            - lambda_axis: wavelength grid (required for C++)
            - mu: viewing angle cosine (default: 1.0)
            - spectralPSF: spectral PSF for convolution
            - nthreads: number of threads
    
    Returns:
        MilneEddingtonTorch or MilneEddingtonCpp instance
    
    Examples:
        >>> # PyTorch backend (default)
        >>> model = MilneEddington(nlinea=1)
        >>> stokes = model(params, wl)
        
        >>> # C++ backend
        >>> model = MilneEddington(nlinea=1, backend='cpp', lambda_axis=wl)
        >>> stokes = model(params, wl)
    """
    if backend == 'torch':
        from .milne_eddington_torch import MilneEddingtonTorch
        return MilneEddingtonTorch(nlinea=nlinea)
    
    elif backend == 'cpp':
        try:
            from .milne_eddington_cpp import MilneEddingtonCPP
            return MilneEddingtonCPP(nlinea=nlinea, **kwargs)
        except ImportError as e:
            raise ImportError(
                f"C++ backend not available: {e}\n"
                "Make sure the C++ extension is compiled and installed."
            )
    
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Choose 'torch' or 'cpp'"
        )


def get_available_backends():
    """
    Check which backends are available.
    
    Returns:
        dict: {'torch': bool, 'cpp': bool}
    """
    backends = {'torch': True}  # PyTorch always available
    
    try:
        from .milne_eddington_cpp import MilneEddingtonCpp
        backends['cpp'] = True
    except ImportError:
        backends['cpp'] = False
    
    return backends


# Public API
__all__ = ['MilneEddington', 'get_available_backends']
