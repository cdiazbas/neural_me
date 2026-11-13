"""
Milne-Eddington synthesis with multiple backend support.

This module provides a unified interface for Milne-Eddington synthesis
with automatic backend selection (PyTorch or C++).
"""


def MilneEddington(line=None, backend="torch", compile=False, nlinea=None, **kwargs):
    """
    Create a Milne-Eddington synthesis model.

    Factory function that returns the appropriate backend implementation.

    Args:
        line: Spectral line — wavelength in Å (e.g. 6302, 6173) or line ID (1-7).
        backend: 'torch' (default) or 'cpp'
        compile: If True and backend='torch', use torch.compile for speedup.
                 Recommended for large batches. Skip for small cases to avoid overhead.
        nlinea: Deprecated alias for line (kept for backward compatibility).
        **kwargs: Additional arguments for C++ backend:
            - lambda_axis: wavelength grid (required for C++)
            - mu: viewing angle cosine (default: 1.0)
            - spectralPSF: (n, 2) array, offset (mA) and transmission of the instrumental profile
            - spectral_step: step (mA) of the fine grid the profile is applied on (default: the profile's)
            - nthreads: number of threads
            - precision: 'float32' or 'float64' (default: 'float64')
        **kwargs: Additional arguments for Torch backend:
            - checkpoint_gradients: If True, use gradient checkpointing (default: False)
            - mu: only 1.0 is supported (ValueError otherwise)

    Returns:
        MilneEddingtonTorch or MilneEddingtonCpp instance

    Examples:
        >>> model = MilneEddington(line=6302)
        >>> model = MilneEddington(line=6173, compile=True)
        >>> model = MilneEddington(line=6302, backend='cpp', lambda_axis=wl)
    """
    # Backward compat: nlinea= still accepted
    if line is None and nlinea is not None:
        line = nlinea
    # Resolve wavelength → line ID
    from neural_me.physics.lineas import get_lineas, get_line_by_wavelength
    if line is not None and line not in set(get_lineas().keys()):
        line_info = get_line_by_wavelength(line)
        line = line_info.line_id
    nlinea = line
    if backend == "torch":
        from .milne_eddington_torch import MilneEddingtonTorch

        if kwargs.get("mu", 1.0) != 1.0:
            raise ValueError(
                "The torch backend synthesises at disc centre only (mu = 1); "
                f"got mu={kwargs['mu']}. Use backend='cpp' for mu != 1."
            )

        checkpoint_gradients = kwargs.get("checkpoint_gradients", False)
        return MilneEddingtonTorch(
            nlinea=nlinea, compile=compile, checkpoint_gradients=checkpoint_gradients
        )

    elif backend == "cpp":
        try:
            from .milne_eddington_cpp import MilneEddingtonCPP

            return MilneEddingtonCPP(nlinea=nlinea, **kwargs)
        except ImportError as e:
            raise ImportError(
                f"C++ backend not available: {e}\n"
                "Make sure the C++ extension is compiled and installed."
            )

    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose 'torch' or 'cpp'")


def get_available_backends():
    """
    Check which backends are available.

    Returns:
        dict: {'torch': bool, 'cpp': bool}
    """
    backends = {"torch": True}  # PyTorch always available

    try:
        from .milne_eddington_cpp import MilneEddingtonCPP
        backends["cpp"] = True
    except ImportError:
        backends["cpp"] = False

    return backends


# Public API
__all__ = ["MilneEddington", "get_available_backends"]
