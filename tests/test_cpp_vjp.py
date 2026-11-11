"""cpp backend: the vector-Jacobian product path (default) against the response-function path (store_rf=True).

The VJP path computes the Stokes only in the forward and the 9 parameter gradients in the backward inside pyMilne,
without the (n_pixels, 9, 4 n_lambda) response functions. Both paths must give the same Stokes and the same
gradients of a loss, for every line the cpp backend implements, in the Hui and Humlicek Voigt branches (damping above
and below 1e-3), for mu != 1, and in both precisions.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpp


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _params(n=64, damping=None, seed=0, dtype=torch.float64):
    g = np.random.default_rng(seed)
    p = np.stack([g.uniform(0, 2500, n), g.uniform(0.1, 3.0, n), g.uniform(0.05, 3.0, n), g.uniform(-4, 4, n),
                  g.uniform(2, 80, n), g.uniform(0.01, 1.0, n) if damping is None else np.full(n, damping),
                  g.uniform(0.02, 0.05, n), g.uniform(0.1, 0.4, n), g.uniform(0.5, 0.9, n)], 1)
    return torch.tensor(p, dtype=dtype)


def _stokes_and_grad(line, precision, store_rf, params, mu=1.0):
    from neural_me.physics.milne_eddington import MilneEddington

    wl = torch.linspace(-0.4, 0.4, 40, dtype=torch.float64)
    me = quiet(MilneEddington, line=line, backend="cpp", lambda_axis=wl, precision=precision, mu=mu)
    me.store_rf = store_rf
    p = params.clone().requires_grad_(True)
    S = quiet(me, p, wl)
    w = torch.tensor([1.0, 10.0, 10.0, 10.0], dtype=p.dtype)
    loss = sum(w[i] * (S[i] - 0.3 * i) ** 2 for i in range(4)).sum()   # non-trivial upstream gradient per Stokes
    loss.backward()
    return torch.stack([s.detach() for s in S]), p.grad.detach()


@pytest.mark.parametrize("line", [6301, 6302, 6173])
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("damping", [None, 5e-4])          # None: Hui branch (a > 1e-3); 5e-4: Humlicek branch
def test_vjp_matches_response_functions(line, precision, damping):
    dt = torch.float64 if precision == "float64" else torch.float32
    p = _params(damping=damping, dtype=dt)
    S0, g0 = _stokes_and_grad(line, precision, True, p)
    S1, g1 = _stokes_and_grad(line, precision, False, p)
    assert torch.equal(S0, S1)                                   # the same synthesis code
    # only the contraction order differs: float64 to 1e-10; float32 to its rounding (measured up to 2.4e-5 of the max
    # in the Humlicek branch, where the float64 paths agree to 1e-10)
    tol = 1e-10 if precision == "float64" else 1e-4
    for k in range(9):
        scale = float(g0[:, k].abs().max()) + 1e-30
        assert float((g0[:, k] - g1[:, k]).abs().max()) <= tol * scale, f"parameter {k}"


def test_vjp_scalar_mu():
    p = _params()
    S0, g0 = _stokes_and_grad(6302, "float64", True, p, mu=0.7)
    S1, g1 = _stokes_and_grad(6302, "float64", False, p, mu=0.7)
    S2, _ = _stokes_and_grad(6302, "float64", False, p, mu=1.0)
    assert torch.equal(S0, S1) and not torch.equal(S1, S2)       # mu is used, and the same way
    assert float((g0 - g1).abs().max()) <= 1e-10 * float(g0.abs().max())


def test_vjp_rejects_per_pixel_mu():
    """pyMilne takes one mu; a per-pixel mu must raise (the RF path raises too), not use only its first value."""
    n = 16
    with pytest.raises((ValueError, TypeError)):
        _stokes_and_grad(6302, "float64", False, _params(n=n), mu=torch.linspace(0.5, 1.0, n, dtype=torch.float64))


def test_levmar_still_reads_response_functions():
    """The LM solver switches the cpp model to store_rf and gets its Jacobian from the forward."""
    from neural_me.core import MEProblem
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.core.observation import Observation
    from neural_me.optimization import LevMarSolver

    wl = torch.linspace(-0.4, 0.4, 30)
    obs = Observation(torch.rand(2, 2, 4, 30) * 0.01 + torch.tensor([1.0, 0, 0, 0]).view(1, 1, 4, 1), wl)
    prob = quiet(MEProblem, obs, line=6302, backend="cpp", precision="float64")
    solver = quiet(LevMarSolver, prob, atmosphere=quiet(SphericalSoftplusAtmosphere, grid_shape=(2, 2)))
    assert prob.model.store_rf
    quiet(solver.solve)
