"""Output head for the thermodynamic parameters of the neural-field atmospheres.

Each parameter is a bounded map of one network output ``o``:

    vlos = amp * tanh(o + b0)                          (|vlos| < amp)
    p    = lo + (hi - lo) * sigmoid(o + b0)            (eta0, a, ddop, S0, S1)

``b0`` places the start: with ``init={name: p0}`` it is chosen so that o = 0
gives exactly p0 (a randomly initialised network outputs o ~ 0); without it
b0 = 0, the mid-point of the range. ``bounds=None`` keeps the original ranges
(eta0 in (0, norm), a in (0.001, 5.001) norm, ddop in (0.01, 0.21) norm,
vlos in +-5 norm, S0 and S1 in (0, norm)); ``bounds="clamp"`` takes them from
the clamp bounds used by the pixel atmospheres.

Source function, ``source``:
  "free"      (default) S0 and S1 are independent network outputs.
  "continuum" S0 = Ic * s, S1 = Ic * (1 - s), s in (0, 1): the continuum is the
              observed map ``ic`` (given at call time) and the network fits
              only the fraction s; the same as normalising each pixel to its
              own continuum, one parameter less. init "S0" is the fraction s.
The original S1 = 1 - S0 (continuum 1 in every pixel) is gone: it cannot fit
an umbra (audit v4 C20).
"""
import math
from typing import Dict, Mapping, Optional

import torch

NAMES = ("vlos", "eta0", "a", "ddop", "S0", "S1")
SOURCES = ("free", "continuum")


def n_outputs(source: str) -> int:
    """Network outputs per view used by the head: S1 has its own output only for "free"."""
    return 6 if source == "free" else 5


class ThermoHead:
    def __init__(self, norm: Mapping[str, float], clamp_bounds: Mapping[str, tuple],
                 init: Optional[Mapping[str, float]] = None, bounds: Optional[str] = None,
                 prefix: str = "", source: str = "free"):
        if bounds not in (None, "clamp"):
            raise ValueError("bounds must be None or 'clamp'")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        self.source = source
        names = NAMES[:n_outputs(source)]
        init = dict(init or {})
        unknown = set(init) - {prefix + n for n in names}
        if unknown:
            raise ValueError(f"unknown init keys {sorted(unknown)}; expected {[prefix + n for n in names]}")
        self.prefix = prefix
        n = {k: float(norm[prefix + k]) for k in NAMES}
        self.norm = n
        rng = {"eta0": (0.0, n["eta0"]), "a": (0.001 * n["a"], 5.001 * n["a"]),
               "ddop": (0.01 * n["ddop"], 0.21 * n["ddop"]), "S0": (0.0, n["S0"]), "S1": (0.0, n["S1"])}
        self.amp = 5.0 * n["vlos"]
        if bounds == "clamp":
            for k in ("eta0", "a", "ddop", "S0", "S1"):
                rng[k] = tuple(float(v) for v in clamp_bounds[prefix + k])
            lo, hi = clamp_bounds[prefix + "vlos"]
            self.amp = float(max(abs(lo), abs(hi)))
        if source == "continuum":
            rng["S0"] = (0.0, 1.0)       # the fraction s = S0 / (S0 + S1)
        self.range = rng
        self.b0 = {k: 0.0 for k in names}
        for k in names:
            if prefix + k not in init:
                continue
            p0 = float(init[prefix + k])
            if k == "vlos":
                u = min(max(p0 / self.amp, -0.999), 0.999)
                self.b0[k] = math.atanh(u)
            else:
                lo, hi = rng[k]
                u = min(max((p0 - lo) / (hi - lo), 1e-4), 1 - 1e-4)
                self.b0[k] = math.log(u / (1 - u))

    def __call__(self, o_vlos, o_eta0, o_a, o_ddop, o_S0, o_S1=None, ic=None) -> Dict[str, torch.Tensor]:
        """``o_S1``: network output for S1 (source="free"); ``ic``: continuum map (source="continuum")."""
        def sig(o, k):
            lo, hi = self.range[k]
            return lo + (hi - lo) * torch.sigmoid(o + self.b0[k])

        vlos = self.amp * torch.tanh(o_vlos + self.b0["vlos"])
        eta0, a, ddop, S0 = sig(o_eta0, "eta0"), sig(o_a, "a"), sig(o_ddop, "ddop"), sig(o_S0, "S0")
        if self.source == "continuum":
            if ic is None:
                raise ValueError('source="continuum" needs the continuum map ic')
            S0, S1 = ic * S0, ic * (1.0 - S0)
        else:
            if o_S1 is None:
                raise ValueError('source="free" needs the S1 network output')
            S1 = sig(o_S1, "S1")
        return {"vlos": vlos, "eta0": eta0, "a": a, "ddop": ddop, "S0": S0, "S1": S1}
