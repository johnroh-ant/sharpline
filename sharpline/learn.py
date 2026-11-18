"""The learning loop: Bayesian regressions that start from the backtest.

1. Signal weights: (result - market) on the signals, for every analysed game.
2. Market trust: when FanDuel and the sharp books disagree, which one moves to close the gap.
"""
import numpy as np

from . import dist, signals

DECAY = 0.9997  # per game: old evidence fades so rule changes get absorbed


def fresh():
    p = dist.params()
    return {"signals": {k: {"A": v["A"], "b": v["b"], "n": 0} for k, v in p["signals"].items()},
            "trust": {k: dict(v, n=0) for k, v in p["trust"].items()}}


def weights(state, sport, kind):
    """Points of adjustment per unit of each signal, already shrunk."""
    s = state["signals"][f"{sport}_{kind}"]
    return np.linalg.solve(np.array(s["A"]), np.array(s["b"])) * dist.params()["signals"][f"{sport}_{kind}"]["kappa"]


def trust(state, sport, market):
    t = state["trust"][f"{sport}_{market}"]
    return min(max(t["xy"] / max(t["xx"], 1e-9), 0.3), 0.95)


def learn_result(state, sport, kind, x, market_mu, outcome):
    s, x = state["signals"][f"{sport}_{kind}"], np.array(x)
    y = float(dist.get(sport, kind).surprise(outcome, market_mu, signals.CLIP[sport, kind]))
    s["A"] = (DECAY * np.array(s["A"]) + np.outer(x, x)
              + (1 - DECAY) * np.diag(dist.params()["signals"][f"{sport}_{kind}"]["ridge"])).tolist()
    s["b"] = (DECAY * np.array(s["b"]) + x * y).tolist()
    s["n"] += 1


def learn_close(state, sport, market, fd_mu, sharp_mu, fd_later, sharp_later):
    t = state["trust"][f"{sport}_{market}"]
    gap, lim = sharp_mu - fd_mu, signals.CLIP[sport, "total" if market == "totals" else "margin"] / 5
    fd_move, sharp_move = (min(max(v, -lim), lim) for v in (fd_later - fd_mu, sharp_later - sharp_mu))
    t["xy"] += gap * fd_move
    t["xx"] += gap * (fd_move - sharp_move)  # a gap that simply persists teaches nothing
    t["n"] += 1
