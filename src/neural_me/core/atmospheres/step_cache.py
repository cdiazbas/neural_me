"""
One network evaluation per optimisation step.

The physical maps of a neural-field atmosphere come from evaluating the whole
network. A step reads them several times: the synthesis of each view
(``to_stack``), and every regularisation term that reads the maps (spatial
smoothness, priors, penalties). Each read evaluated the network again, forward
and backward. Inside ``step_cache()`` the first read evaluates it and the others
reuse that output; the gradient is the same, because autograd sums every use of
one graph.

The cache must be closed before the parameters change (``optimizer.step()``):
the solvers open it after ``zero_grad`` and close it after ``backward``.
"""

import contextlib

import torch


class StepCacheMixin:
    """Adds ``step_cache()`` and ``_cached(fn)`` to an atmosphere."""

    _step_cache_on = False
    _step_cache_out = None

    @contextlib.contextmanager
    def step_cache(self):
        """Within the block, the network is evaluated once and its output reused."""
        self._step_cache_on, self._step_cache_out = True, None
        try:
            yield self
        finally:
            self._step_cache_on, self._step_cache_out = False, None

    def _cached(self, evaluate):
        """``evaluate()`` once per open cache. The first call may come from code
        under ``torch.no_grad()`` (a logged quantity): the cached output is built
        with the graph anyway, so later reads still carry the gradient."""
        if not self._step_cache_on:
            return evaluate()
        if self._step_cache_out is None:
            with torch.enable_grad():
                self._step_cache_out = evaluate()
        return self._step_cache_out
