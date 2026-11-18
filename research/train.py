"""Fit params/model.json and run the walk-forward backtest.

usage: python research/train.py HIST_DIR   (see research/README.md for the inputs)
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize as op
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sharpline import config, dist, feeds, ratings, signals
H = sys.argv[1]
ip = lambda a: np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def fit_dist(spec, keys, line, p, y, shrink, fit=("scale", "df")):
    """Alternate: locate mu from each (line, fair p), fit shape by ML, rescale key numbers."""
    spec = dict(spec, mult={str(k): 1.0 for k in keys})
    if 0 in keys: spec["mult"]["0"] = 0.05
    u, inv = np.unique(np.c_[line, np.round(p, 2)], axis=0, return_inverse=True); inv = inv.ravel()
    col = (y - spec["lo"]).astype(int).clip(0, spec["hi"] - spec["lo"])
    def rows(s):
        d = dist.Dist(s); mu = np.array([d.invert(a, b) for a, b in u])
        return d, d.pmf[np.searchsorted(d.mus, mu).clip(0, len(d.mus) - 1)]
    for _ in range(5):
        def nll(t): return -np.log(rows(dict(spec, **dict(zip(fit, t * x0))))[1][inv, col] + 1e-12).mean()
        x0 = np.array([spec[k] for k in fit]); r = op.minimize(nll, np.ones(len(fit)), method="Nelder-Mead", options=dict(maxiter=30 * len(fit), xatol=.01, fatol=1e-5))
        spec.update(zip(fit, map(float, r.x * x0)))
        d, P = rows(spec); E = (P * np.bincount(inv, minlength=len(u))[:, None]).sum(0); O = np.bincount(col, minlength=len(d.grid))
        for k in keys:
            m = (np.abs(d.grid) if spec["abs"] else d.grid) == k; s = 0.01 if k == 0 else shrink
            spec["mult"][str(k)] = float(round(spec["mult"][str(k)] * (O[m].sum() + s) / (E[m].sum() + s), 4))
    print(" dist", {k: round(spec[k], 3) for k in fit}, "nll", round(r.fun, 4))
    return spec


# ---- history
G = {}
for sp, lg in (("NBA", "nba"), ("NCAAB", "mbb")):
    rd = lambda k: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{H}/dl/hoop/{lg}_{k}_*.parquet"))], ignore_index=True)
    g = feeds.hoops_frame(sp, rd("sch"), rd("tb"))
    L = pd.read_parquet(f"{H}/work/{sp.lower()}.parquet")
    L = L.assign(h_id=L.h_id.astype(str), line_m=-L.c_spread, line_t=L.c_total)
    G[sp] = g.merge(L[["game_id", "h_id", "line_m", "line_t"]], on=["game_id", "h_id"], how="left")
n = pd.read_csv(f"{H}/dl/nfl/games.csv"); n = n[n.season >= 2002]; G["NFL"] = feeds.nfl_frame(n)
m = pd.read_parquet(f"{H}/work/mlb.parquet").rename(columns=dict(h="h_id", a="a_id")); m = m[m.season <= 2025]
o = pd.read_parquet(f"{H}/work/mlb_odds.parquet")
def mlbq(mk, A, B, ln):
    d = o[(o.market == mk) & o[A].abs().between(100, 450) & o[B].abs().between(100, 450)].copy()
    d["ov"] = ip(d[A]) + ip(d[B]); d = d[d.ov.between(1.0, 1.09)]; d["p"] = ip(d[A]) / d.ov
    d["line"], d["pa"], d["pb"] = (0.0 if ln is None else d[ln]), d[A], d[B]; return d
Q = {"h2h": mlbq("moneyline", "opening_home_odds", "opening_away_odds", None),
     "spreads": mlbq("pointspread", "opening_home_odds", "opening_away_odds", "opening_home_spread"),
     "totals": mlbq("totals", "opening_over_odds", "opening_under_odds", "opening_total")}
Q["spreads"] = Q["spreads"][Q["spreads"].line.abs() == 1.5].assign(line=lambda d: -d.line)
Q["totals"] = Q["totals"][Q["totals"].line.between(5.5, 14)]

# ---- distributions
D, half = {}, lambda x: np.full(len(x), .5)
x = G["NFL"][(G["NFL"].season >= 2015)].dropna(subset=["h_pts"])
D["NFL_margin"] = fit_dist(dict(lo=-65, hi=65, mu_lo=-30, mu_hi=30, step=.05, abs=True, scale=12., df=20.), list(range(25)), x.line_m.values, half(x), (x.h_pts - x.a_pts).values, 15)
D["NFL_total"] = fit_dist(dict(lo=0, hi=115, mu_lo=25, mu_hi=70, step=.05, abs=False, scale=12., df=15., skew=1.05), list(range(20, 67)), x.line_t.values, half(x), (x.h_pts + x.a_pts).values, 40, ("scale", "df", "skew"))
for sp, y0, sm, st, lo, hi in (("NBA", 2021, 12.7, 17.5, 170, 280), ("NCAAB", 2016, 10.5, 15.5, 95, 200)):
    x = G[sp][G[sp].season >= y0].dropna(subset=["h_pts", "line_m", "line_t"])
    D[sp + "_margin"] = fit_dist(dict(lo=-90, hi=90, mu_lo=-50, mu_hi=50, step=.1, abs=True, scale=sm, df=18.), list(range(13)), x.line_m.values, half(x), (x.h_pts - x.a_pts).values, 60)
    D[sp + "_total"] = fit_dist(dict(lo=lo - 60, hi=hi + 80, mu_lo=lo, mu_hi=hi, step=.1, abs=False, scale=st, df=20., skew=1.05, slope=.04, ref=(lo + hi) / 2), [], x.line_t.values, half(x), (x.h_pts + x.a_pts).values, 1, ("scale", "df", "skew", "slope"))
x = Q["h2h"][Q["h2h"].sportsbook == "fanduel"]
D["MLB_margin"] = fit_dist(dict(lo=-25, hi=25, mu_lo=-5, mu_hi=5, step=.01, abs=False, scale=4., df=10.), [0, 1, -1, 2, -2], x.line.values, x.p.values, (x.h_pts - x.a_pts).values, 20)
x = Q["totals"][Q["totals"].sportsbook == "fanduel"]
D["MLB_total"] = fit_dist(dict(lo=1, hi=38, mu_lo=3, mu_hi=15, step=.01, abs=False, scale=3.6, df=12., skew=1.25, slope=.2, ref=8.), list(range(1, 17)), x.line.values, x.p.values, (x.h_pts + x.a_pts).values, 30, ("scale", "df", "skew", "slope"))
P = dict(dist=D, ml_slope={"MLB": 0.93}, signals={}, trust={})
(config.PARAMS / "model.json").write_text(json.dumps(P)); dist.params.cache_clear(); dist.get.cache_clear()

# ---- market mu, ratings
for sp in ("NBA", "NCAAB", "NFL"):
    g = G[sp]
    for k, c in (("margin", "m"), ("total", "t")):
        d = dist.get(sp, k); g["mk_" + c] = g["line_" + c].map({v: d.invert(v, .5) for v in g["line_" + c].dropna().unique()})
    G[sp] = ratings.run(sp, g)
for mk, q in Q.items():
    d = dist.get("MLB", "total" if mk == "totals" else "margin"); p = q.p if mk != "h2h" else 1 / (1 + ((1 - q.p) / q.p) ** .93)
    q["mu"] = [d.invert(a, b) for a, b in zip(q.line, p)]
m["mk_m"] = m.game_id.map(Q["h2h"].groupby("game_id").mu.median()); m["mk_t"] = m.game_id.map(Q["totals"].groupby("game_id").mu.median())
G["MLB"] = ratings.run("MLB", m)

# ---- signal weights, walk-forward
HL = dict(NBA=float(sys.argv[2]), NCAAB=float(sys.argv[2]), NFL=8.0, MLB=7.0) if len(sys.argv) > 2 else dict(NBA=7.0, NCAAB=4.0, NFL=8.0, MLB=7.0)
FIRST = {"NBA": 2016, "NCAAB": 2016, "NFL": 2012, "MLB": 2022}; OOS = {}
for (sp, k), F in signals.FEATURES.items():
    g = G[sp].dropna(subset=["h_pts", "mk_m", "mk_t"]); g = g[g.season > g.season.min()] if sp != "MLB" else g
    if sp == "NCAAB": g = g[(g.h_key != "X") & (g.a_key != "X")]
    g = g[(g.m_raw - g.mk_m).abs() < signals.MAX_GAP[sp]]
    X = signals.matrix(sp, k, g, g.mk_m.values, g.mk_t.values)
    res = ((g.h_pts - g.a_pts - g.mk_m) if k == "margin" else (g.h_pts + g.a_pts - g.mk_t)).values
    y = dist.get(sp, k).surprise(res + g["mk_" + k[0]].values, g["mk_" + k[0]].values, signals.CLIP[sp, k])
    ridge = {"NBA": 600, "NCAAB": 600, "NFL": 600, "MLB": 3000}[sp] * (X ** 2).mean(0)
    def fit(upto):
        w = 0.5 ** ((upto - g.season.values) / HL[sp]) * (g.season.values < upto)
        return (X * w[:, None]).T @ X + np.diag(ridge), (X * w[:, None]).T @ y
    pred = np.full(len(g), np.nan)
    for s in range(FIRST[sp], g.season.max() + 1):
        A, b = fit(s); t = g.season.values == s; pred[t] = X[t] @ np.linalg.solve(A, b)
    ok = ~np.isnan(pred); d = dist.get(sp, k); line = (g["line_" + k[0]] if sp != "MLB" else (g.mk_t.round() + .5 if k == "total" else 0 * g.mk_m)).values
    mk = g["mk_" + k[0]].values; out = (res + mk)
    def ll(kap, t):
        pr = np.array([d.probs(a, b) for a, b in zip(line[t], mk[t] + kap * pred[t])]); p = pr[:, 0] / pr.sum(1); z = out[t] != line[t]
        return -np.where(out[t] > line[t], np.log(p), np.log(1 - p))[z].mean()
    grid = np.arange(0, 2.51, .25); curve = [ll(v, ok) for v in grid]; best = grid[int(np.argmin(curve))]
    # kappa for season s uses earlier out-of-sample seasons only, then a 25% haircut for selection effects
    kap = {s: (min(1.3, 0.75 * grid[int(np.argmin([ll(v, ok & (g.season.values < s)) for v in grid]))]) if s > FIRST[sp] + 1 else .6) for s in range(FIRST[sp], g.season.max() + 1)}
    A, b = fit(g.season.max() + 1); beta = np.linalg.solve(A, b)
    print(sp, k, "n", len(g), "kappa*", best, "gain", round((curve[0] - min(curve)) * 1e4, 1), dict(zip(signals.names(sp, k), beta.round(3).tolist())), "sd", np.nanstd(pred).round(2), [float(round(v, 2)) for v in kap.values()])
    P["signals"][f"{sp}_{k}"] = dict(A=A.tolist(), b=b.tolist(), ridge=ridge.tolist(), kappa=float(min(max(0.75 * best, 0.1), 1.3)), n=int(len(g)))
    OOS[sp, k] = pd.Series(pred * g.season.map(kap).values, index=g.game_id.values)
for sp, v in {"NFL": .22, "NBA": .35, "NCAAB": .45, "MLB": .15}.items():
    for mk in ("h2h", "spreads", "totals"):  # prior worth ~150 games
        t = 0.45 if (sp, mk) == ("NBA", "totals") else 0.85; P["trust"][f"{sp}_{mk}"] = dict(xx=150 * v * v, xy=150 * v * v * t)
(config.PARAMS / "model.json").write_text(json.dumps(P))
pd.to_pickle(dict(G=G, OOS=OOS, Q=Q), f"{H}/work/trained.pkl")
