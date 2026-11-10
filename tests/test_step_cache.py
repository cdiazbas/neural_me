"""One network evaluation per optimisation step (atmospheres.step_cache.StepCacheMixin)."""

import pytest
import torch

from neural_me.core.atmospheres import NFAtmosphere
from neural_me.core.atmospheres.nf_stereoscopic import StereoscopicNFAtmosphere


def _stereo():
    torch.manual_seed(0)
    return StereoscopicNFAtmosphere(grid_shape=(16, 16), encoding="grid")


def _single():
    torch.manual_seed(0)
    return NFAtmosphere(encoding="grid", source="free", grid_shape=(16, 16))


def _count(atm):
    """Counts the evaluations of the network."""
    n = [0]
    forward = atm.NF.forward

    def counted(*a, **k):
        n[0] += 1
        return forward(*a, **k)
    atm.NF.forward = counted
    return n


def _reads_stereo(atm):
    """What a stereo step reads: both views' synthesis inputs, a regulariser on the stack, a prior on eta0."""
    v1, v2 = atm.get_atmosphere(1), atm.get_atmosphere(2)
    return v1.to_stack().sum() + v2.to_stack().sum() + atm.to_stack().pow(2).mean() + v2.eta0.log().mean()


def _reads_single(atm):
    return atm.to_stack().sum() + atm.to_stack().pow(2).mean() + atm.eta0.log().mean()


@pytest.mark.parametrize("make, reads", [(_stereo, _reads_stereo), (_single, _reads_single)])
def test_one_evaluation_per_step(make, reads):
    atm = make()
    n = _count(atm)
    reads(atm)
    without = n[0]
    n[0] = 0
    with atm.step_cache():
        reads(atm)
    assert without > 1 and n[0] == 1


@pytest.mark.parametrize("make, reads", [(_stereo, _reads_stereo), (_single, _reads_single)])
def test_same_value_and_gradient(make, reads):
    """In float64, so that only the order in which autograd sums the uses differs (float32: ~4e-7 relative)."""
    atm = make().double()
    params = [p for p in atm.parameters() if p.requires_grad]
    v0 = reads(atm)
    g0 = torch.autograd.grad(v0, params, allow_unused=True)
    with atm.step_cache():
        v1 = reads(atm)
        g1 = torch.autograd.grad(v1, params, allow_unused=True)
    assert torch.allclose(v0, v1, rtol=1e-12, atol=0)
    for a, b in zip(g0, g1):
        if a is None:
            assert b is None
        else:
            assert float((a - b).abs().max()) <= 1e-12 * float(a.abs().max())


def test_first_read_under_no_grad_keeps_the_graph():
    """A logged quantity read under no_grad first must not leave the cache without a graph."""
    atm = _stereo()
    with atm.step_cache():
        with torch.no_grad():
            atm.to_stack().sum()
        loss = atm.to_stack().pow(2).mean()
        assert loss.requires_grad
        loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in atm.parameters())


def test_closed_after_the_block():
    """Outside the block every read evaluates the network again (the parameters may have changed)."""
    atm = _stereo()
    n = _count(atm)
    with atm.step_cache():
        atm.to_stack()
    atm.to_stack(); atm.to_stack()
    assert n[0] == 3
