"""cpp backend with a spectral PSF: fine-grid synthesis plus a fixed matrix (spectral_psf_matrix).

The matrix must cover the whole profile for every observed point, reproduce pyMilne's own PSF path where that path
is right (a symmetric profile centred on its middle sample, observed points on the grid, a wide margin), apply the
profile as a correlation, and give the same Stokes and gradients in the VJP path and in the response-function path
the LM solver uses.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpp

LINE = {"cw": 6173.334, "jl": 0.0, "ju": 1.0, "gl": 0.0, "gu": 2.5}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _gaussian(fwhm=100.0, step=10.0, half=310.0):
    x = np.arange(-half, half + step / 2, step)
    return np.stack([x, np.exp(-4 * np.log(2) * (x / fwhm) ** 2)], 1)


def _params(n=32, seed=0, dtype=torch.float64):
    g = np.random.default_rng(seed)
    p = np.stack([g.uniform(0, 2500, n), g.uniform(0.1, 3.0, n), g.uniform(0.05, 3.0, n), g.uniform(-3, 3, n),
                  g.uniform(2, 80, n), g.uniform(0.01, 1.0, n), g.uniform(0.02, 0.05, n), g.uniform(0.1, 0.4, n),
                  g.uniform(0.5, 0.9, n)], 1)
    return torch.tensor(p, dtype=dtype)


def test_matrix_covers_the_profile():
    from neural_me.physics.milne_eddington_cpp.milne_cpp import spectral_psf_matrix

    psf = _gaussian()
    w = np.array([-0.35, -0.21, -0.07, 0.07, 0.21, 0.35])
    wf, M = spectral_psf_matrix(w, psf, 15.0)
    assert np.allclose(np.diff(wf), 0.015)
    assert wf[0] <= w.min() - 0.31 and wf[-1] >= w.max() + 0.31
    assert np.allclose(M.sum(1), 1.0)
    # every observed point sees the whole profile: no weight in the first and last fine points
    assert np.all(M[:, 0] == 0) and np.all(M[:, -1] == 0)


def test_correlation_with_exact_offsets():
    """A profile with all its weight at +20 mA samples the spectrum 20 mA redwards (a correlation), whatever the
    position of the zero offset inside the array (pyMilne's path takes the middle sample as zero)."""
    from neural_me.physics.milne_eddington_cpp.milne_cpp import spectral_psf_matrix

    psf = np.array([[-30.0, 0.0], [-20.0, 0.0], [-10.0, 0.0], [0.0, 0.0], [10.0, 0.0], [20.0, 1.0], [30.0, 0.0]])
    w = np.array([-0.1, 0.0, 0.1])
    wf, M = spectral_psf_matrix(w, psf[1:], 10.0)          # 6 samples: the middle one is not the zero offset
    assert np.allclose(wf[M.argmax(1)], w + 0.02)


def test_against_pymilne_psf_path():
    """Symmetric profile centred on its middle sample, observed points on the fine grid: the matrix and pyMilne's
    region PSF on a wide grid give the same Stokes."""
    from neural_me.physics.milne_eddington import MilneEddington
    from neural_me.physics.milne_eddington_cpp.milne_cpp import _LineasMilneEddington

    psf = _gaussian()
    w = np.arange(-6, 7) * 0.06                              # on the 10 mA grid
    me = quiet(MilneEddington, line=6173, backend="cpp", lambda_axis=torch.tensor(w), precision="float64",
               spectralPSF=psf)
    p = _params()
    with torch.no_grad():
        S = torch.stack(me(p)).numpy()                        # (4, n, n_obs)

    wide = np.arange(-120, 121) * 0.01
    ref = quiet(_LineasMilneEddington, [[wide + LINE["cw"], psf[:, 1] / psf[:, 1].sum()]], LINE, nthreads=1,
                precision="float64")
    q = p.numpy().copy()
    model = np.stack([q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 6], q[:, 4] / np.sqrt(np.pi), q[:, 5], q[:, 7],
                      q[:, 8]], 1)[None]
    R = quiet(ref.synthesize, model, mu=1.0)[0]               # (n, 4, n_wide)
    idx = np.rint((w - wide[0]) / 0.01).astype(int)
    assert np.abs(S - R[:, :, idx].transpose(1, 0, 2)).max() < 1e-6


@pytest.mark.parametrize("step", [None, 15.0])
def test_vjp_and_rf_paths_agree(step):
    from neural_me.physics.milne_eddington import MilneEddington

    psf = _gaussian()
    w = torch.tensor([-0.35, -0.21, -0.07, 0.07, 0.21, 0.35], dtype=torch.float64)
    me = quiet(MilneEddington, line=6173, backend="cpp", lambda_axis=w, precision="float64", spectralPSF=psf,
               spectral_step=step)
    weights = torch.randn(4, 32, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    out = {}
    for store_rf in (False, True):
        me.store_rf = store_rf
        p = _params().requires_grad_(True)
        S = me(p)
        (sum((s * wt).sum() for s, wt in zip(S, weights))).backward()
        out[store_rf] = (torch.stack(S).detach(), p.grad.clone())
    assert torch.allclose(out[False][0], out[True][0], rtol=0, atol=1e-12)
    assert torch.allclose(out[False][1], out[True][1], rtol=1e-8, atol=1e-8)
    # the response functions the LM solver reads are those of the degraded Stokes
    rf = me.get_last_response_functions()
    assert rf.shape == (32, 9, 4 * 6)
