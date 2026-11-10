"""regularization.priors.log_prior, and StereoSolver's penalty term."""

import math

import numpy as np
import pytest
import torch

from neural_me.core import MEProblem, Observation
from neural_me.core.atmospheres.nf_stereoscopic import StereoscopicNFAtmosphere
from neural_me.optimization.stereo_solver import StereoSolver
from neural_me.regularization.priors import log_prior


def test_log_prior_values():
    x = torch.tensor([50.0, 100.0, 25.0])
    assert float(log_prior(x, 50.0)) == pytest.approx((0 + 2 * math.log(2) ** 2) / 3, rel=1e-6)
    # scale-free: a factor 2 weighs the same at any centre
    assert float(log_prior(torch.tensor([400.0]), 200.0)) == pytest.approx(float(log_prior(torch.tensor([10.0]), 5.0)))
    mask = torch.tensor([True, False, False])
    assert float(log_prior(x, 50.0, mask)) == 0.0
    with pytest.raises(ValueError):
        log_prior(x, 0.0)


def _solver(penalty=None):
    torch.manual_seed(0)
    wl1 = np.linspace(-0.4, 0.4, 20); wl2 = np.linspace(-0.3, 0.15, 6)
    rng = np.random.default_rng(0)
    d1 = (1 + 0.01 * rng.standard_normal((8, 8, 4, 20))).astype(np.float32); d1[:, :, 1:] *= 0.01
    d2 = (1 + 0.01 * rng.standard_normal((8, 8, 4, 6))).astype(np.float32); d2[:, :, 1:] *= 0.01
    p1 = MEProblem(observation=Observation(data=d1, wavelengths=wl1), line=6302, backend="torch")
    p2 = MEProblem(observation=Observation(data=d2, wavelengths=wl2), line=6173, backend="torch")
    atm = StereoscopicNFAtmosphere(grid_shape=(8, 8), encoding="grid")
    return StereoSolver(atm, p1, p2, lr=1e-2, penalty=penalty), atm


def test_penalty_pulls_eta0_and_shares_the_network_evaluation():
    pen = lambda atm: 1.0 * log_prior(atm.get_atmosphere(2).eta0, 300.0)
    s0, a0 = _solver()
    s1, a1 = _solver(pen)
    n = [0]
    fwd = a1.NF.forward

    def counted(*a, **k):
        n[0] += 1
        return fwd(*a, **k)
    a1.NF.forward = counted
    s0.solve(n_iterations=20, verbose=False)
    s1.solve(n_iterations=20, verbose=False)
    with torch.no_grad():
        e0 = float(a0.get_atmosphere(2).eta0.mean()); e1 = float(a1.get_atmosphere(2).eta0.mean())
    assert abs(math.log(e1 / 300.0)) < abs(math.log(e0 / 300.0))     # pulled towards 300
    assert len(s1.penalties) == 20 and s1.penalties[-1] < s1.penalties[0]
    assert n[0] == 20 + 1                                              # one per step (+ the read above)
