"""Replay the real pipeline over timestamped multi-book snapshots (9:00 and 15:20 PT each day)."""
import datetime as dt, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sharpline import config, feeds, run
H, a, b = sys.argv[1:4]
O = pd.read_parquet(f"{H}/work/abet.parquet"); O["t"] = pd.to_datetime(O.fetch_timestamp, format="ISO8601"); O["c"] = pd.to_datetime(O.commence_time)
full, clock = {}, {}
real = feeds.games
def games(sport, day):
    if sport not in full: full[sport] = real(sport, day)
    g = full[sport].copy(); late = g.start > clock["now"] - pd.Timedelta(hours=5)
    g.loc[late, [c for c in g if c[2:] in ("pts", "poss")]] = np.nan; g.loc[late, "status"] = "STATUS_SCHEDULED"; return g
def odds(sport, start, end):
    d = O[(O.sport == sport.lower()) & (O.c > start) & (O.c <= end)]; d = d[(d.t - clock["now"]).abs() < pd.Timedelta("80min")]; d = d[d.t < d.c]
    if d.empty: return []
    d = d[d.t == d.t.iloc[(d.t - clock["now"]).abs().values.argmin()]]
    return [dict(id=i, commence_time=e.commence_time.iloc[0], home_team=e.home_team.iloc[0], away_team=e.away_team.iloc[0], bookmakers=[
        dict(key=k, markets=[dict(key=m, outcomes=[dict(name=r.outcome_name, price=int(r.price_american), point=None if r.point != r.point else r.point) for r in x.itertuples()])
             for m, x in bk.groupby("market_key")]) for k, bk in e.groupby("bookmaker_key")]) for i, e in d.groupby("game_id")]
feeds.games, feeds.odds, feeds.scores, feeds.wind = games, odds, lambda s: {}, lambda *a: None
config.SPORTS = {k: v for k, v in config.SPORTS.items()}
for day in pd.date_range(a, b):
    for fn, hm in ((run.morning, dt.time(9, 0)), (run.afternoon, dt.time(15, 20))):
        clock["now"] = dt.datetime.combine(day.date(), hm, config.TZ).astimezone(dt.timezone.utc); print(day.date(), end=" "); fn(clock["now"], fn is run.afternoon)
