"""Summarise both backtests into site/backtest.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
H = sys.argv[1]; A, B = "2022-01-01", "2026-11-16"
new = pd.read_parquet(f"{H}/work/new_bt.parquet"); old = pd.read_parquet(f"{H}/work/old_opening.parquet"); old = old[old.card].assign(ev=lambda d: d.edge)
old["pnl"] = old.unit * old.conf; old["market"] = old.market.map(dict(spread="spreads", moneyline="h2h", total="totals"))
def s(x):
    d = x[x.res != 0]
    return dict(n=len(x), w=int((x.res > 0).sum()), l=int((x.res < 0).sum()), win=round((d.res > 0).mean(), 4), said=round(x.p.mean(), 4), claimed=round(np.average(x.ev, weights=x.conf), 4),
                staked=int(x.conf.sum()), pnl=round(x.pnl.sum()), roi=round(x.pnl.sum() / x.conf.sum(), 4), se=round(np.sqrt(((x.pnl - x.conf * x.pnl.sum() / x.conf.sum()) ** 2).sum()) / x.conf.sum(), 4))
w = lambda d: d[d.date.between(A, B)]; n, o = w(new), w(old)
cum = lambda d: d.groupby(pd.Grouper(key="date", freq="W")).pnl.sum().cumsum().round()
c = pd.concat([cum(n), cum(o)], axis=1).ffill()
bins = lambda d: {str(k): s(v) for k, v in d.groupby(pd.cut(d.conf, [0, 1, 25, 50, 75, 100], labels=["1", "2-25", "26-50", "51-75", "76-100"]), observed=True)}
dd = lambda x: int((x - x.cummax()).min())
out = dict(window=[A, B], new=s(n), old=s(o), new_ex_mlb=s(n[n.sport != "MLB"]), full=s(new), full_ex_mlb=s(new[new.sport != "MLB"]), real=s(n[n.conf > 1]),
           curve=[[str(i.date()), *map(float, r)] for i, r in zip(c.index, c.values)],
           sport={k: dict(new=s(n[n.sport == k]), old=s(o[o.sport == k])) for k in ("NFL", "NBA", "NCAAB", "MLB")},
           conf=dict(new=bins(n), old=bins(o)), year={str(k): s(v) for k, v in new.groupby(new.date.dt.year)},
           drawdown=dd(new.groupby("date").pnl.sum().cumsum()),
           losing_months=round(float((new.groupby(new.date.dt.to_period("M")).pnl.sum() < 0).mean()), 2))
(Path(__file__).resolve().parent.parent / "site/backtest.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=0)[:6000])
