"""Thermodynamic output head of the neural-field atmospheres (nf_heads.ThermoHead)."""
import contextlib
import io

import pytest
import torch

from neural_me.core.atmospheres import NFAtmosphere, StereoscopicNFAtmosphere
from neural_me.core.atmospheres.base import Atmosphere
from neural_me.core.atmospheres.nf_heads import NAMES, ThermoHead


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


NORM = Atmosphere.DEFAULT_NORMALIZATION
BOUNDS = Atmosphere.DEFAULT_CLAMP_BOUNDS
INIT = {"vlos": 0.21, "eta0": 12.9, "a": 0.23, "ddop": 0.037, "S0": 0.31}


@pytest.mark.parametrize("bounds", [None, "clamp"])
def test_zero_output_gives_the_initial_values(bounds):
    head = ThermoHead(NORM, BOUNDS, init=INIT, bounds=bounds)
    zero = torch.zeros(3, dtype=torch.float64)
    out = head(*(zero for _ in NAMES))
    for k, v in INIT.items():
        assert torch.allclose(out[k], torch.full_like(zero, v), rtol=1e-6)
    assert torch.allclose(out["S1"], 1 - out["S0"])


@pytest.mark.parametrize("bounds", [None, "clamp"])
def test_outputs_stay_inside_their_ranges(bounds):
    head = ThermoHead(NORM, BOUNDS, init=INIT, bounds=bounds)
    big = torch.tensor([-1e3, -10.0, 0.0, 10.0, 1e3], dtype=torch.float64)
    out = head(*(big for _ in NAMES))
    assert out["vlos"].abs().max() <= head.amp
    for k in ("eta0", "a", "ddop", "S0"):
        lo, hi = head.range[k]
        assert out[k].min() >= lo and out[k].max() <= hi


def test_clamp_bounds_reach_the_pixel_range():
    head = ThermoHead(NORM, BOUNDS, init={"eta0": 185.0}, bounds="clamp")
    assert head.range["eta0"] == BOUNDS["eta0"]
    assert float(head(*(torch.zeros(1) for _ in NAMES))["eta0"]) == pytest.approx(185.0, rel=1e-5)
    legacy = ThermoHead(NORM, BOUNDS)
    assert legacy.range["eta0"][1] == NORM["eta0"] < 185.0


def test_unknown_init_key_is_rejected():
    with pytest.raises(ValueError, match="unknown init keys"):
        ThermoHead(NORM, BOUNDS, init={"Ddop": 0.03})
    with pytest.raises(ValueError, match="unknown init keys"):
        quiet(StereoscopicNFAtmosphere, grid_shape=(2, 2), m_freqs=8, dim_hidden=4,
              num_resnet_blocks=1, init={"eta0": 10.0})


def test_nf_atmosphere_starts_near_init_with_a_fresh_network():
    torch.manual_seed(0)
    atm = quiet(NFAtmosphere, grid_shape=(16, 16), m_freqs=16, dim_hidden=16,
                num_resnet_blocks=1, init=INIT, bounds="clamp")
    st = atm.to_stack().detach()
    # a fresh network outputs values of order 0.1-1, so the start sits around p0
    assert float(st[..., 6].median()) == pytest.approx(INIT["ddop"], rel=0.5)
    assert float(st[..., 4].median()) == pytest.approx(INIT["eta0"], rel=0.5)


def test_stereo_nf_views_get_their_own_init():
    torch.manual_seed(0)
    atm = quiet(StereoscopicNFAtmosphere, grid_shape=(8, 8), m_freqs=16, dim_hidden=8,
                num_resnet_blocks=1, bounds="clamp",
                init={"thermo1_eta0": 24.0, "thermo2_eta0": 185.0})
    assert atm._thermo_heads[1].range["eta0"] == Atmosphere.DEFAULT_STEREO_CLAMP_BOUNDS["thermo1_eta0"]
    e1 = float(atm.get_atmosphere(1).to_stack()[..., 4].median())
    e2 = float(atm.get_atmosphere(2).to_stack()[..., 4].median())
    assert e2 > 3 * e1
