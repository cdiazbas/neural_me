"""core.velocity: the velocity vector from the LOS velocities of two views.

The line of sight of view 2 must be the one of StereoscopicAtmosphere; the field-aligned solution must recover a
synthetic speed and its zero points exactly; the umbral-core zero must be the median of the darkest umbra; and the
neural field must fit both velocities, recover a smooth speed along the field, leave the unseen component alone
without a prior, and move towards the field with the alignment penalty.
"""

import numpy as np
import pytest
import torch

from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere
from neural_me.core.velocity import (field_aligned_velocity, fit_velocity_field, solve_velocity, to_local,
                                     unseen_direction, view_los_vectors, zero_at_umbral_core)

GAMMA, BETA = 101.2, 0.8


def _unit_field(shape, seed=0):
    g = np.random.default_rng(seed)
    b = g.normal(size=shape + (3,))
    return b / np.linalg.norm(b, axis=-1, keepdims=True)


def _smooth(shape, seed=1):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]] / max(shape)
    g = np.random.default_rng(seed)
    a = g.uniform(0.5, 1.5, 3)
    return 2.0 * np.sin(2 * np.pi * a[0] * xx) * np.cos(2 * np.pi * a[1] * yy) + a[2]


def test_view_los_matches_the_stereoscopic_atmosphere():
    atm = StereoscopicAtmosphere(grid_shape=(4, 4), gamma=GAMMA, beta=BETA, delta=90.0)
    B = 1000.0 * _unit_field((4, 4), seed=3)
    atm.set_parameters({"global_Blos": B[..., 0], "global_Bn": B[..., 1], "global_Bw": B[..., 2]})
    with torch.no_grad():
        v2 = atm.get_atmosphere(2)
        blos2 = (v2.B * torch.cos(v2.gamma)).numpy().reshape(4, 4)
    e1, e2 = view_los_vectors(GAMMA, BETA)
    assert np.allclose(blos2, B @ e2, rtol=1e-4, atol=1e-2)
    assert np.allclose(e1, [1, 0, 0])


def test_unseen_direction_is_perpendicular_to_both_views():
    e1, e2 = view_los_vectors(GAMMA, BETA)
    n = unseen_direction(e1, e2)
    assert abs(n @ e1) < 1e-12 and abs(n @ e2) < 1e-12 and np.isclose(np.linalg.norm(n), 1)
    R = np.broadcast_to(np.eye(3)[[2, 1, 0]], (2, 2, 3, 3))               # any rotation (here a permutation)
    assert np.allclose(unseen_direction(e1, e2, R)[0, 0], n[[2, 1, 0]])


@pytest.mark.parametrize("zero", ["fit", "given"])
def test_field_aligned_velocity_recovers_speed_and_zero_points(zero):
    shape = (40, 50)
    b = _unit_field(shape)
    s = _smooth(shape)
    e1, e2 = view_los_vectors(GAMMA, BETA)
    o = (0.3, -0.7)
    v1, v2 = o[0] - s * (b @ e1), o[1] - s * (b @ e2)
    out = field_aligned_velocity(v1, v2, b, e1, e2, zero="fit" if zero == "fit" else o)
    u = out["use"]
    assert u.sum() > 0.5 * u.size
    assert np.allclose(out["offsets"], o, atol=1e-8)
    assert np.allclose(out["s"][u], s[u], atol=1e-8)
    assert np.abs(out["residual"][u]).max() < 1e-8
    # the other azimuth solution (transverse field reversed) does not explain the data
    flipped = field_aligned_velocity(v1, v2, b * np.array([1.0, -1.0, -1.0]), e1, e2, zero=o)
    assert flipped["robust_sigma"] > 0.1


def test_zero_at_umbral_core():
    g = np.random.default_rng(0)
    cont = np.ones((100, 100)); cont[30:70, 30:70] = 0.3; cont[45:55, 45:55] = 0.1
    v = g.normal(0, 0.1, cont.shape) + 2.0; v[45:55, 45:55] = -0.5
    assert np.isclose(zero_at_umbral_core(v, cont), -0.5)


@pytest.fixture(scope="module")
def synthetic():
    shape = (32, 32)
    g = np.random.default_rng(2)
    b0 = np.array([0.3, 0.5, 0.8]); b0 /= np.linalg.norm(b0)
    yy, xx = np.mgrid[0:32, 0:32] / 32.0
    b = b0 + 0.3 * np.stack([np.sin(3 * xx), np.cos(2 * yy), np.sin(2 * (xx + yy))], -1)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    s = _smooth(shape)
    e1, e2 = view_los_vectors(GAMMA, BETA)
    R = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()              # local frame = frame of view 1
    v = s[..., None] * b
    v1, v2 = -(v @ e1), -(v @ e2)
    return dict(shape=shape, b=b, s=s, e1=e1, e2=e2, R=R, v1=v1, v2=v2, mask=np.ones(shape, bool), v=v, g=g)


def test_fit_velocity_field_aligned_recovers_the_speed(synthetic):
    d = synthetic
    out = fit_velocity_field(d["v1"], d["v2"], d["e1"], d["e2"], d["R"], d["mask"], b=d["b"], mode="aligned", n_iter=400)
    s_fit = (out["v_local"] * d["b"]).sum(-1)
    assert out["data_residual"] < 0.05
    assert np.sqrt(np.mean((s_fit - d["s"]) ** 2)) < 0.1 * np.std(d["s"])


def test_fit_velocity_field_free_and_alignment_penalty(synthetic):
    d = synthetic
    n = unseen_direction(d["e1"], d["e2"])
    free = fit_velocity_field(d["v1"], d["v2"], d["e1"], d["e2"], d["R"], d["mask"], mode="free", n_iter=400)
    pen = fit_velocity_field(d["v1"], d["v2"], d["e1"], d["e2"], d["R"], d["mask"], b=d["b"], mode="free",
                             lam_align=1.0, n_iter=400)
    assert free["data_residual"] < 0.05 and pen["data_residual"] < 0.1
    # without a prior the unseen component stays exactly at zero (minimum norm); the true one is not zero here
    assert np.abs(free["v_local"] @ n).max() < 1e-6 < np.median(np.abs(d["v"] @ n))
    assert np.abs(pen["v_local"] @ n).max() > 0.1
    # the penalty pulls the solution towards the true (field-aligned) velocity
    err = lambda v: np.sqrt(np.mean(np.sum((v - d["v"]) ** 2, -1)))
    assert err(pen["v_local"]) < 0.5 * err(free["v_local"])


def test_to_local_and_argument_checks(synthetic):
    d = synthetic
    assert np.allclose(to_local(d["v"], d["R"]), d["v"])
    with pytest.raises(ValueError):
        fit_velocity_field(d["v1"], d["v2"], d["e1"], d["e2"], d["R"], d["mask"], mode="aligned", n_iter=1)
    with pytest.raises(ValueError):
        field_aligned_velocity(d["v1"], d["v2"], d["b"], d["e1"], d["e2"], zero="umbra")


def test_solve_velocity_closed_form(synthetic):
    d = synthetic
    n = unseen_direction(d["e1"], d["e2"])
    v0 = solve_velocity(d["v1"], d["v2"], d["e1"], d["e2"], d["R"])
    assert np.allclose(-(v0 @ d["e1"]), d["v1"], atol=1e-10) and np.allclose(-(v0 @ d["e2"]), d["v2"], atol=1e-10)
    assert np.abs(v0 @ n).max() < 1e-10                                    # minimum norm: nothing along the unseen axis
    # with a strong alignment penalty the solution tends to the true field-aligned velocity
    v1_ = solve_velocity(d["v1"], d["v2"], d["e1"], d["e2"], d["R"], b=d["b"], lam_align=1e4)
    assert np.sqrt(np.mean(np.sum((v1_ - d["v"]) ** 2, -1))) < 1e-3
    with pytest.raises(ValueError):
        solve_velocity(d["v1"], d["v2"], d["e1"], d["e2"], d["R"], lam_align=1.0)


def test_solve_velocity_l1_and_l2(synthetic):
    """With the alignment penalty the two losses differ; the l1 solution is the minimum of its own objective."""
    d = synthetic
    g = np.random.default_rng(5)
    v1 = d["v1"] + g.normal(0, 0.3, d["v1"].shape); v2 = d["v2"] + g.normal(0, 0.3, d["v2"].shape)
    l2 = solve_velocity(v1, v2, d["e1"], d["e2"], d["R"], b=d["b"], lam_align=1.0, loss="l2")
    l1 = solve_velocity(v1, v2, d["e1"], d["e2"], d["R"], b=d["b"], lam_align=1.0, loss="l1")
    E = -np.stack([d["e1"], d["e2"]])
    def J(v, p):                     # per-pixel loss of fit_velocity_field: mean over the two views + lam |v_perp|^2
        r = v @ E.T - np.stack([v1, v2], -1)
        perp = v - (v * d["b"]).sum(-1, keepdims=True) * d["b"]
        return 0.5 * (np.abs(r) ** p).sum(-1) + 1.0 * (perp ** 2).sum(-1)
    assert np.abs(l1 - l2).max() > 1e-3
    assert np.all(J(l1, 1) <= J(l2, 1) + 1e-6) and np.all(J(l2, 2) <= J(l1, 2) + 1e-6)
    # each is the minimum of that loss, not only better than the other: no small step lowers it
    for v, p in ((l1, 1), (l2, 2)):
        for step in g.normal(0, 0.05, (20,) + v.shape):
            assert np.all(J(v, p) <= J(v + step, p) + 1e-6)
    # without the penalty both fit the data exactly and coincide
    assert np.allclose(solve_velocity(v1, v2, d["e1"], d["e2"], d["R"], loss="l1"), solve_velocity(v1, v2, d["e1"], d["e2"], d["R"]))


def test_fit_velocity_field_smoothness(synthetic):
    """With noisy velocities the smoothness penalty gives a smoother field, closer to the true one."""
    d = synthetic
    g = np.random.default_rng(7)
    v1 = d["v1"] + g.normal(0, 0.5, d["v1"].shape); v2 = d["v2"] + g.normal(0, 0.5, d["v2"].shape)
    kw = dict(b=d["b"], mode="free", lam_align=0.1, n_iter=400)
    rough = lambda v: np.mean(np.sum((v[1:, :] - v[:-1, :]) ** 2, -1)) + np.mean(np.sum((v[:, 1:] - v[:, :-1]) ** 2, -1))
    v0 = fit_velocity_field(v1, v2, d["e1"], d["e2"], d["R"], d["mask"], **kw)["v_local"]
    vs = fit_velocity_field(v1, v2, d["e1"], d["e2"], d["R"], d["mask"], smooth=10.0, **kw)["v_local"]
    err = lambda v: np.sqrt(np.mean(np.sum((v - d["v"]) ** 2, -1)))
    assert rough(vs) < 0.5 * rough(v0)
    assert err(vs) < err(v0)
