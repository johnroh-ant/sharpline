"""Walk-forward team ratings. Each game's numbers use only earlier days."""
import numpy as np
import pandas as pd


class Ridge:
    """Exponentially decayed least squares shrunk toward a preseason prior."""

    def __init__(self, n_team, lam, mean, hca, lam_mean=30.0, lam_hca=200.0):
        k = n_team + 2
        self.A, self.b = np.zeros((k, k)), np.zeros(k)
        self.prior = np.zeros(k)
        self.prior[-2:] = mean, hca
        self.lam = np.full(k, float(lam))
        self.lam[-2:] = lam_mean, lam_hca
        self.x = self.prior.copy()

    def add(self, idx, val, y):
        self.A[np.ix_(idx, idx)] += np.outer(val, val)
        self.b[idx] += val * y

    def step(self, decay, active):
        self.A *= decay
        self.b *= decay
        a = np.asarray(active)
        self.x[a] = np.linalg.solve(self.A[np.ix_(a, a)] + np.diag(self.lam[a]),
                                    self.b[a] + self.lam[a] * self.prior[a])

    def new_season(self, carry):
        self.prior = self.x.copy()
        self.prior[:-2] *= carry
        self.A[:] = 0
        self.b[:] = 0
        self.x = self.prior.copy()


HOOPS = {
    "NBA": dict(half_life=40, lam=5, carry=0.60, minutes=48, periods=4, luck3=0.65, luck_ft=0.3, ft=0.78, pace=98, total=225),
    "NCAAB": dict(half_life=90, lam=3, carry=0.75, minutes=40, periods=2, luck3=0.60, luck_ft=0.3, ft=0.72, pace=68, total=145),
}


def _history(g, out):
    """Rest, schedule density and recent against-the-spread form."""
    played, ats, ou = {}, {}, {}
    cols = {s + k: np.zeros(len(g)) for s in "ha" for k in ("_rest", "_g4", "_ats", "_last", "_gp")}
    for i, r in enumerate(g.itertuples()):
        for s, t, sign in (("h", r.h_id, 1), ("a", r.a_id, -1)):
            k = (r.season, t)
            days, a = played.setdefault(k, []), ats.setdefault(k, [])
            cols[s + "_rest"][i] = min((r.date - days[-1]).days - 1, 14) if days else 7
            cols[s + "_g4"][i] = sum((r.date - d).days <= 4 for d in days[-4:])
            cols[s + "_ats"][i], cols[s + "_last"][i] = sum(a[-8:]) / 8, (a[-1] if a else 0)
            cols[s + "_gp"][i] = len(days)
            if r.h_pts == r.h_pts:
                days.append(r.date)
                if r.mk_m == r.mk_m:
                    a.append(sign * (r.h_pts - r.a_pts - r.mk_m))
    for k, v in cols.items():
        out[k] = v


def hoops(sport, g):
    """Tempo-free offence/defence ratings, raw and with shooting luck removed.

    g needs date, season, h_id, a_id, neutral, rated (both teams are league
    members), box-score columns (NaN when unplayed) and mk_m / mk_t (past market
    margin and total, NaN when unknown).
    """
    P = HOOPS[sport]
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    teams = {t: i for i, t in enumerate(sorted(set(g.h_key) | set(g.a_key)))}
    n = len(teams)
    hi, ai = g.h_key.map(teams).values, g.a_key.map(teams).values
    raw, adj = (Ridge(2 * n, P["lam"], 105, 1.5) for _ in range(2))
    pace = Ridge(n, P["lam"] * 0.6, P["pace"], 0)
    mk_m, mk_t = Ridge(n, 0.6, 0, 2.5, lam_mean=1e6), Ridge(n, 0.6, P["total"], 0)
    use_mkt = sport == "NBA"
    p3, pft, season, prev = 0.35, P["ft"], None, None
    out = np.full((len(g), 12), np.nan)
    for day, x in g.groupby("date", sort=True):
        s = x.season.iloc[0]
        if s != season:
            if season is not None:
                for m, c in ((raw, P["carry"]), (adj, P["carry"]), (pace, 0.6), (mk_m, 0.7), (mk_t, 0.7)):
                    m.new_season(c)
            season = s
            m = (g.season == s).values
            act = sorted(set(hi[m]) | set(ai[m]))
            act2, act1 = act + [n + i for i in act] + [2 * n, 2 * n + 1], act + [n, n + 1]
        else:
            gap = (day - prev).days
            for m in (raw, adj):
                m.step(0.5 ** (gap / P["half_life"]), act2)
            pace.step(0.5 ** (gap / P["half_life"]), act1)
            if use_mkt:
                for m in (mk_m, mk_t):
                    m.step(0.5 ** (gap / 10), act1)
        prev = day
        for r in x.itertuples():
            h, a, H = hi[r.Index], ai[r.Index], 0.0 if r.neutral else 1.0
            pc, row = pace.x[n] + pace.x[h] + pace.x[a], []
            for m in (raw, adj):
                eh = m.x[-2] + m.x[h] - m.x[n + a] + H * m.x[-1]
                ea = m.x[-2] + m.x[a] - m.x[n + h] - H * m.x[-1]
                row += [(eh - ea) * pc / 100, (eh + ea) * pc / 100]
            row += [mk_m.x[h] - mk_m.x[a] + H * mk_m.x[-1], mk_t.x[n] + mk_t.x[h] + mk_t.x[a], mk_t.x[n], pc]
            row += [raw.x[h] + raw.x[n + h], adj.x[h] + adj.x[n + h], raw.x[a] + raw.x[n + a], adj.x[a] + adj.x[n + a]]
            out[r.Index] = row
        for r in x.itertuples():
            h, a, H = hi[r.Index], ai[r.Index], 0.0 if r.neutral else 1.0
            poss = (r.h_poss + r.a_poss) / 2
            if poss > 40 and r.h_pts == r.h_pts:  # suspended games have a box score but no final
                for t, o, sign, pts, tpa, tpm, fta, ftm in (
                        (h, a, 1, r.h_pts, r.h_tpa, r.h_tpm, r.h_fta, r.h_ftm),
                        (a, h, -1, r.a_pts, r.a_tpa, r.a_tpm, r.a_fta, r.a_ftm)):
                    idx, v = np.array([t, n + o, 2 * n, 2 * n + 1]), np.array([1, -1, 1, sign * H])
                    luck3, luck_ft = tpm - p3 * tpa, ftm - pft * fta
                    raw.add(idx, v, 100 * pts / poss)
                    adj.add(idx, v, 100 * (pts - P["luck3"] * 3 * luck3 - P["luck_ft"] * luck_ft) / poss)
                    p3 += 0.0004 * luck3 / 25
                    pft += 0.0004 * luck_ft / 20
                if h != a:
                    ot = min(max(r.periods - P["periods"], 0), 6)
                    pace.add(np.array([h, a, n]), np.ones(3), poss * P["minutes"] / (P["minutes"] + 5 * ot))
            if use_mkt and r.h_pts == r.h_pts:
                if r.mk_m == r.mk_m:
                    mk_m.add(np.array([h, a, n + 1]), np.array([1, -1, H]), r.mk_m)
                if r.mk_t == r.mk_t:
                    mk_t.add(np.array([h, a, n]), np.ones(3), r.mk_t)
    g[["m_raw", "t_raw", "m_adj", "t_adj", "m_mkt", "t_mkt", "t_lg", "pace",
       "h_net_raw", "h_net_adj", "a_net_raw", "a_net_adj"]] = out
    _history(g, g)
    return g


def simple(g, specs):
    """One-number ratings for NFL / MLB. specs: name -> (target, symmetric, half-life days, lam, carry, mean, hca)."""
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    teams = {t: i for i, t in enumerate(sorted(set(g.h_id) | set(g.a_id)))}
    n = len(teams)
    hi, ai, H = g.h_id.map(teams).values, g.a_id.map(teams).values, 1.0 - g.neutral.values.astype(float)
    for name, (target, sym, hl, lam, carry, mean, hca) in specs.items():
        m, y = Ridge(n, lam, mean, hca, lam_mean=40 if sym else 1e6, lam_hca=150), g[target].values
        out, season, prev = np.full(len(g), np.nan), None, None
        for day, x in g.groupby("date", sort=True):
            if x.season.iloc[0] != season:
                if season is not None:
                    m.new_season(carry)
                season = x.season.iloc[0]
            else:
                m.step(0.5 ** ((day - prev).days / hl), range(n + 2))
            prev = day
            rows = [(i, np.array([hi[i], ai[i], n if sym else n + 1]),
                     np.array([1, 1, 1.0]) if sym else np.array([1, -1, H[i]])) for i in x.index]
            for i, idx, v in rows:
                out[i] = v @ m.x[idx]
            for i, idx, v in rows:
                if y[i] == y[i]:
                    m.add(idx, v, y[i])
        g[name] = out
    _history(g, g)
    return g


def nfl(g):
    g = g.assign(done_m=g.mk_m.where(g.h_pts.notna()), done_t=g.mk_t.where(g.h_pts.notna()),
                 y_m=(g.h_pts - g.a_pts).clip(-24, 24), y_t=g.h_pts + g.a_pts)
    return simple(g, {"m_mkt": ("done_m", False, 21, 0.5, 0.7, 0, 2), "t_mkt": ("done_t", True, 21, 0.5, 0.6, 44, 0),
                      "m_raw": ("y_m", False, 84, 5, 0.6, 0, 2), "t_raw": ("y_t", True, 84, 8, 0.5, 44, 0)})


def mlb(g):
    g = g.assign(y_m=(g.h_pts - g.a_pts).clip(-6, 6), y_t=(g.h_pts + g.a_pts).clip(0, 18))
    return simple(g, {"m_raw": ("y_m", False, 120, 40, 0.6, 0, 0.1), "t_raw": ("y_t", True, 120, 60, 0.6, 8.8, 0)})


def run(sport, g):
    return {"NFL": nfl, "MLB": mlb}[sport](g) if sport in ("NFL", "MLB") else hoops(sport, g)
