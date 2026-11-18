"""Outcome distributions: one location parameter mu -> probability of any line.

Every book's (line, price) is inverted to mu and FanDuel is priced from mu with
the same curve, so a conversion can never manufacture an edge.
"""
import json
from functools import lru_cache

import numpy as np

from . import config


def dec(american):
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def devig(price_a, price_b):
    """(fair p of side a, overround). Proportional: it calibrated best on our data."""
    a, b = 1 / dec(price_a), 1 / dec(price_b)
    return a / (a + b), a + b


def american(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


class Dist:
    """Discretised skewed Student-t with learned key-number multipliers."""

    def __init__(self, spec):
        self.grid = np.arange(spec["lo"], spec["hi"] + 1)
        self.mus = np.arange(spec["mu_lo"], spec["mu_hi"], spec["step"])
        mult = np.array([spec["mult"].get(str(abs(int(g)) if spec["abs"] else int(g)), 1.0) for g in self.grid])
        scale = spec["scale"] + spec.get("slope", 0.0) * (self.mus - spec.get("ref", 0.0))
        d = self.grid[None] - self.mus[:, None]
        z = d / (scale[:, None] * np.where(d > 0, spec.get("skew", 1.0), 1 / spec.get("skew", 1.0)))
        p = (1 + z * z / spec["df"]) ** (-(spec["df"] + 1) / 2)
        # label each curve by its median, so mu reads like a betting line even when skewed
        cdf = p.cumsum(1) / p.sum(1, keepdims=True)
        self.mus = np.array([np.interp(0.5, row, self.grid + 0.5) for row in cdf])
        p = p * mult[None]
        self.pmf = p / p.sum(1, keepdims=True)
        self._curves = {}

    def _curve(self, x):
        if x not in self._curves:
            over = self.pmf[:, self.grid > x].sum(1)
            under = self.pmf[:, self.grid < x].sum(1)
            self._curves[x] = (over, under, np.maximum.accumulate(over / (over + under)))
        return self._curves[x]

    def invert(self, x, p):
        """mu at which P(outcome > x | no push) equals p."""
        return float(np.interp(p, self._curve(x)[2], self.mus))

    def probs(self, x, mu):
        """(P(outcome > x), P(outcome < x)); the remainder is a push."""
        over, under, _ = self._curve(x)
        return float(np.interp(mu, self.mus, over)), float(np.interp(mu, self.mus, under))

    def surprise(self, outcome, mu, lim):
        """Clipped miss vs mu, centred so that a perfectly priced market averages zero."""
        i = np.searchsorted(self.mus, mu).clip(0, len(self.mus) - 1)
        expected = (self.pmf[i] * np.clip(self.grid - np.asarray(mu)[..., None], -lim, lim)).sum(-1)
        return np.clip(outcome - mu, -lim, lim) - expected

    def fair_line(self, mu):
        """The half-point-interpolated line that is a coin flip at mu."""
        xs = np.arange(np.floor(mu) - 4.5, np.floor(mu) + 5.6)
        ps = np.array([self.probs(x, mu)[0] for x in xs])
        return float(np.interp(0.5, ps[::-1], xs[::-1]))


@lru_cache(None)
def params():
    return json.loads((config.PARAMS / "model.json").read_text())


@lru_cache(None)
def get(sport, kind):
    return Dist(params()["dist"][f"{sport}_{kind}"])
