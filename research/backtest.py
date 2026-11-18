"""Daily top-10 card on history, using only out-of-sample signal weights."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sharpline import card, dist, market
H = sys.argv[1]; T = pd.read_pickle(f"{H}/work/trained.pkl"); G, OOS, Q = T["G"], T["OOS"], T["Q"]
nba_ml = pd.read_parquet(f"{H}/work/nba.parquet").set_index("game_id")[["c_hml", "c_aml"]]
ok = lambda *v: all(x == x and abs(x) >= 100 for x in v)
mlb = {}
for mk, q in Q.items():
    fd = q[q.sportsbook == "fanduel"].set_index("game_id"); ot = q[q.sportsbook != "fanduel"]
    if mk == "spreads": ot = ot[ot.line == ot.game_id.map(fd.line)]
    c = ot.groupby("game_id").mu.agg(["mean", "size"]); fd = fd.join(c[c["size"] >= 2], how="inner")
    mlb[mk] = {i: ((r.line, r.pa, r.pb), r.mu + (.85) * (r.mean - r.mu)) for i, r in zip(fd.index, fd.rename(columns={"mean": "mean_"}).assign(mean=fd["mean"]).itertuples())}
rows = []
for sp, g in G.items():
    dm, dt = OOS[sp, "margin"].dropna().to_dict(), OOS[sp, "total"].dropna().to_dict()
    for r in g[g.game_id.isin(dm) & (g.date >= "2016-10-01")].itertuples():
        q = {}
        if sp == "MLB": q = {mk: v[r.game_id] for mk, v in mlb.items() if r.game_id in v}
        else:
            pr = (r.home_spread_odds, r.away_spread_odds, r.over_odds, r.under_odds, r.home_moneyline, r.away_moneyline) if sp == "NFL" else (-110,) * 4 + (tuple(nba_ml.loc[r.game_id]) if sp == "NBA" else (np.nan,) * 2)
            for mk, ln, a, b in (("spreads", r.line_m, *pr[:2]), ("totals", r.line_t, *pr[2:4]), ("h2h", 0.0, *pr[4:])):
                if ok(a, b) and (mu := market.book_mu(sp, mk, (ln, a, b))[0]) is not None: q[mk] = ((ln, a, b), mu)
        side = [o for mk in ("spreads", "h2h") if mk in q for o in card.options(sp, mk, q[mk][0], q[mk][1] + dm[r.game_id])]
        tot = card.options(sp, "totals", q["totals"][0], q["totals"][1] + dt[r.game_id]) if "totals" in q else []
        for o in (card.best(side), card.best(tot)):
            if o:
                st = card.grade(o["market"], o["side"], o["line"], r.h_pts, r.a_pts)
                rows.append(dict(o, sport=sp, date=r.date, status=st, unit=card.profit(st, 1, o["price"])))
c = pd.DataFrame(rows); c["rank"] = c.groupby("date").kelly.rank(ascending=False, method="first"); b = c[c["rank"] <= 10].copy()
b["pnl"] = b.unit * b.conf; b["res"] = b.status.map(dict(won=1, lost=-1, push=0)); b.to_parquet(f"{H}/work/new_bt.parquet"); c.to_parquet(f"{H}/work/new_all.parquet")
s = lambda x: pd.Series(dict(n=len(x), win=(x.res > 0).sum() / max(1, (x.res != 0).sum()), wwin=(x.conf * (x.res > 0)).sum() / (x.conf * (x.res != 0)).sum(), p=x.p.mean(), ev=(x.ev * x.conf).sum() / x.conf.sum(), staked=x.conf.sum(), pnl=x.pnl.sum(), roi=x.pnl.sum() / x.conf.sum(), se=np.sqrt(((x.pnl - x.pnl.mean()) ** 2).sum()) / x.conf.sum(), flat=x.unit.mean()))
pd.set_option("display.width", 200)
print(s(b).round(3).to_dict())
for k in ("sport", ["sport", "market"], b.date.dt.year, pd.cut(b.conf, [0, 1, 10, 25, 50, 75, 100])): print(b.groupby(k).apply(s).round(3))
w = b[b.date >= "2020-12-01"]; print("window", s(w).round(3).to_dict()); print(w.groupby("sport").apply(s).round(3))
