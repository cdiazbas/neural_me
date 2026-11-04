"""Output head for the thermodynamic parameters of the neural-field atmospheres.

Each parameter is a bounded map of one network output ``o``:

    vlos = amp * tanh(o + b0)                          (|vlos| < amp)
    p    = lo + (hi - lo) * sigmoid(o + b0)            (eta0, a, ddop, S0)

``b0`` places the start: with ``init={name: p0}`` it is chosen so that o = 0
gives exactly p0 (a randomly initialised network outputs o ~ 0); without it
b0 = 0, the mid-point of the range. ``bounds=None`` keeps the original ranges
(eta0 in (0, norm), a in (0.001, 5.001) norm, ddop in (0.01, 0.21) norm,
vlos in +-5 norm); ``bounds="clamp"`` takes vlos, eta0, a and ddop from the
clamp bounds used by the pixel atmospheres. S0 always lies in (0, norm["S0"])
and S1 = 1 - S0. With neither init nor bounds the original expressions are
evaluated unchanged.
"""
import math
from typing import Dict, Mapping, Optional

import torch

NAMES = ("vlos", "eta0", "a", "ddop", "S0")


class ThermoHead:
    def __init__(self, norm: Mapping[str, float], clamp_bounds: Mapping[str, tuple],
                 init: Optional[Mapping[str, float]] = None, bounds: Optional[str] = None,
                 prefix: str = ""):
        if bounds not in (None, "clamp"):
            raise ValueError("bounds must be None or 'clamp'")
        init = dict(init or {})
        unknown = set(init) - {prefix + n for n in NAMES}
        if unknown:
            raise ValueError(f"unknown init keys {sorted(unknown)}; expected {[prefix + n for n in NAMES]}")
        self.prefix = prefix
        self.norm = {n: float(norm[prefix + n]) for n in NAMES}
        self.legacy = bounds is None and not init
        n = self.norm
        rng = {"eta0": (0.0, n["eta0"]), "a": (0.001 * n["a"], 5.001 * n["a"]),
               "ddop": (0.01 * n["ddop"], 0.21 * n["ddop"]), "S0": (0.0, n["S0"])}
        self.amp = 5.0 * n["vlos"]
        if bounds == "clamp":
            for k in ("eta0", "a", "ddop"):
                rng[k] = tuple(float(v) for v in clamp_bounds[prefix + k])
            lo, hi = clamp_bounds[prefix + "vlos"]
            self.amp = float(max(abs(lo), abs(hi)))
        self.range = rng
        self.b0 = {k: 0.0 for k in NAMES}
        for k in NAMES:
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

    def __call__(self, o_vlos, o_eta0, o_a, o_ddop, o_S0) -> Dict[str, torch.Tensor]:
        n = self.norm
        if self.legacy:  # the original expressions, unchanged
            vlos = 10.0 * torch.tanh(o_vlos) * (n["vlos"] / 2.0)
            eta0 = torch.sigmoid(o_eta0) * n["eta0"]
            a = (0.001 + 5.0 * torch.sigmoid(o_a)) * n["a"]
            ddop = (0.01 + 0.2 * torch.sigmoid(o_ddop)) * n["ddop"]
            S0 = torch.sigmoid(o_S0) * n["S0"]
        else:
            def sig(o, k):
                lo, hi = self.range[k]
                return lo + (hi - lo) * torch.sigmoid(o + self.b0[k])
            vlos = self.amp * torch.tanh(o_vlos + self.b0["vlos"])
            eta0, a, ddop, S0 = sig(o_eta0, "eta0"), sig(o_a, "a"), sig(o_ddop, "ddop"), sig(o_S0, "S0")
        return {"vlos": vlos, "eta0": eta0, "a": a, "ddop": ddop, "S0": S0, "S1": 1.0 - S0}
