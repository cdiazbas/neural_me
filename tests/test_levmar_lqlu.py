"""LevMarSolver(field="lqlu"): the LM steps in [Blos, Lq, Lu] with (Lq, Lu) = B_perp^2 (cos 2 xi, sin 2 xi).

The coordinates must map to and from [B, gamma, xi] exactly (xi modulo pi), the Jacobian carried by the chain rule must
agree with finite differences of the forward model in the new coordinates, |L| must be floored at 1 G^2 so that a start
at B_perp = 0 has a finite Jacobian, and a fit started at B_perp ~ 0 must leave the cone where the spherical step stalls,
without fitting the strong-field pixels worse.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest
import torch

from neural_me.optimization.levmar_solver import L_FLOOR, L_SCALE, _floor_lqlu, _lqlu_to_sph, _sph_to_lqlu

WL = torch.linspace(-0.5, 0.5, 48, dtype=torch.float64)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _truth(n, seed=0):
    """Half weak (B_perp 30-150 G), half strong (B_perp 300-2000 G) pixels, [B, gamma, xi, vlos, eta0, a, ddop, S0, S1]."""
    g = np.random.default_rng(seed)
    weak = np.arange(n) < n // 2
    bt = np.where(weak, g.uniform(30, 150, n), g.uniform(300, 2000, n))
    bl = np.where(weak, g.uniform(-200, 200, n), g.uniform(-1500, 1500, n))
    p = np.stack([np.hypot(bl, bt), np.arctan2(bt, bl), g.uniform(0, np.pi, n), g.uniform(-2, 2, n), g.uniform(5, 30, n),
                  g.uniform(0.05, 0.4, n), g.uniform(0.025, 0.04, n), g.uniform(0.15, 0.35, n), g.uniform(0.6, 0.8, n)], 1)
    return torch.tensor(p), weak


def _problem(params, noise=1e-3, seed=1):
    from neural_me.core import MEProblem
    from neural_me.core.observation import Observation
    from neural_me.physics.milne_eddington import MilneEddington

    me = quiet(MilneEddington, line=6302, backend="cpp", lambda_axis=WL, precision="float64")
    with torch.no_grad():
        S = torch.stack(quiet(me, params, WL.unsqueeze(0).expand(len(params), -1)), 1)    # (n, 4, nwl)
    S = S + noise * torch.as_tensor(np.random.default_rng(seed).normal(size=S.shape))
    n = len(params)
    obs = Observation(S.reshape(4, n // 4, 4, len(WL)).float(), WL.float())      # float32, as the atmospheres
    return quiet(MEProblem, obs, line=6302, backend="cpp", precision="float64")


def _solver(prob, field, start):
    from neural_me.core.atmospheres import SphericalSoftplusAtmosphere
    from neural_me.optimization import LevMarSolver

    atm = quiet(SphericalSoftplusAtmosphere, grid_shape=prob.obs.grid_shape)
    s = quiet(LevMarSolver, prob, atmosphere=atm, field=field)
    s._params_to_atmosphere_sph(start.clone())
    s.set_weights(1e-3, 1e-3, 1e-3, 1e-3)
    return s


def test_coordinates_round_trip_and_floor():
    p, _ = _truth(64)
    back = _lqlu_to_sph(_sph_to_lqlu(p))
    assert torch.allclose(back[:, :2], p[:, :2], rtol=1e-10, atol=1e-8)
    dxi = (back[:, 2] - p[:, 2] + np.pi / 2) % np.pi - np.pi / 2
    assert dxi.abs().max() < 1e-10 and torch.equal(back[:, 3:], p[:, 3:])
    # below the floor: radially onto |L| = L_FLOOR (B_perp = 1 G), the origin onto (L_FLOOR, 0)
    q = torch.zeros(3, 9, dtype=torch.float64)
    q[1, 1], q[1, 2] = 0.3 / L_SCALE, -0.4 / L_SCALE           # |L| = 0.5 G^2, azimuth kept
    q[2, 1] = 5.0 / L_SCALE                                      # above the floor: untouched
    f = _floor_lqlu(q)
    L = torch.hypot(f[:, 1], f[:, 2]) * L_SCALE
    assert torch.allclose(L, torch.tensor([L_FLOOR, L_FLOOR, 5.0], dtype=torch.float64))
    assert torch.allclose(f[1, 1:3] / f[1, 1:3].norm(), q[1, 1:3] / q[1, 1:3].norm())
    assert f[0, 1] * L_SCALE == L_FLOOR and f[0, 2] == 0
    assert torch.isclose(_lqlu_to_sph(f)[0, 0] * torch.sin(_lqlu_to_sph(f)[0, 1]), torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.cpp
def test_jacobian_matches_finite_differences():
    p, _ = _truth(8, seed=3)
    p[0, 0], p[0, 1] = 40.0, 1.2                                 # B_perp ~ 37 G: near the cone, still away from the floor
    prob = _problem(p)
    s = _solver(prob, "lqlu", p)
    q = _sph_to_lqlu(p)
    syn, J = quiet(s._forward_and_jacobian, q)
    for k, h in ((0, 0.05), (1, 2e-3), (2, 2e-3)):               # Blos in G; Lq, Lu in L_SCALE G^2 (200 G^2)
        dq = torch.zeros_like(q); dq[:, k] = h
        fd = (quiet(s._forward_only, q + dq) - quiet(s._forward_only, q - dq)) / (2 * h)
        scale = J[:, k].abs().amax(-1, keepdim=True)
        assert ((fd - J[:, k]).abs() / scale).max() < 5e-3, k
    # at the floor (B_perp = 1 G) the Jacobian is finite and the Q, U columns of (Lq, Lu) do not vanish
    z = q.clone(); z[:, 1:3] = 0.0
    _, Jz = quiet(s._forward_and_jacobian, z)
    assert torch.isfinite(Jz).all()
    nw = len(WL)
    assert (Jz[:, 1, nw:3 * nw].abs().amax(-1) > 1e-3 * J[:, 1, nw:3 * nw].abs().amax(-1)).all()


@pytest.mark.cpp
def test_fit_from_the_cone():
    """Start at B_perp = 1 G: the lqlu fit leaves the cone; the strong-field fit is not worse than the spherical one."""
    p, weak = _truth(16, seed=5)
    prob = _problem(p)
    start = p.clone()
    start[:, 0], start[:, 1] = 1.0, 0.5 * np.pi                  # B = B_perp = 1 G
    start[:, 2] = 0.3; start[:, 3:] = torch.tensor([0.0, 10.0, 0.2, 0.03, 0.25, 0.7], dtype=torch.float64)
    out = {}
    for field in ("spherical", "lqlu"):
        s = _solver(prob, field, start)
        quiet(s.solve, max_iter=60, chi2_threshold=0.0)
        fit = s.atmosphere.to_stack().reshape(-1, 9).detach()
        chi2 = ((prob_stokes(prob, fit) - prob_stokes(prob, None)) ** 2).mean(-1) / 1e-6
        out[field] = (fit[:, 0] * torch.sin(fit[:, 1]), chi2)
    bt, chi2 = out["lqlu"]
    true_bt = p[:, 0] * torch.sin(p[:, 1])
    assert (bt > 5.0).all()                                      # nobody left at the floor
    assert chi2.max() < 2.0                                      # every pixel at the noise
    assert (chi2[~torch.as_tensor(weak)] <= out["spherical"][1][~torch.as_tensor(weak)] * 1.05 + 0.05).all()
    assert ((bt - true_bt).abs() / true_bt)[~torch.as_tensor(weak)].max() < 0.05


def prob_stokes(prob, params):
    """Observed (params None) or synthetic Stokes of every pixel, (n, 4 nwl)."""
    if params is None:
        return torch.cat([prob.obs.stokes_I, prob.obs.stokes_Q, prob.obs.stokes_U, prob.obs.stokes_V], -1)
    with torch.no_grad():
        S = quiet(prob.model, params, WL.unsqueeze(0).expand(len(params), -1))
    return torch.cat(list(S), -1)


def test_field_argument():
    from neural_me.optimization import LevMarSolver
    with pytest.raises(ValueError):
        LevMarSolver(None, atmosphere=None, field="cartesian")
