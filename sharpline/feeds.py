"""Data sources. Odds cost API credits; results and box scores are free."""
import datetime as dt
import http.client
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import config

UA = "sharpline/1.0 (personal paper-betting project)"
RAW = "https://raw.githubusercontent.com"
credits = {"remaining": None}


def get(url, cache=None, tries=3, timeout=30):
    """Bytes at url. `cache` names a file for data that no longer changes."""
    path = config.CACHE / cache if cache else None
    if path and path.exists():
        return path.read_bytes()
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                body = r.read()
                if r.headers.get("x-requests-remaining"):
                    credits["remaining"] = int(float(r.headers["x-requests-remaining"]))
            break
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or i == tries - 1:  # the URL may hold the API key: report host and code only
                raise RuntimeError(f"{urllib.parse.urlsplit(url).netloc} answered {e.code}") from None
        except (OSError, http.client.HTTPException) as e:
            if i == tries - 1:
                raise RuntimeError(f"{urllib.parse.urlsplit(url).netloc} unreachable ({type(e).__name__})") from None
        time.sleep(2 ** i * 2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return body


# --- The Odds API -----------------------------------------------------------

def _odds_api(path, **q):
    q["apiKey"] = os.environ["ODDS_API_KEY"]
    return json.loads(get(f"https://api.the-odds-api.com/v4{path}?{urllib.parse.urlencode(q)}"))


def _iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def odds(sport, start, end):
    """Events with quotes from ten books. The events check is free, so empty days cost nothing."""
    key, window = config.SPORTS[sport], dict(commenceTimeFrom=_iso(start), commenceTimeTo=_iso(end))
    try:
        if not _odds_api(f"/sports/{key}/events", **window):
            return []
    except RuntimeError as e:
        if "404" in str(e) or "422" in str(e):  # out of season
            return []
        raise
    q = dict(markets="h2h,spreads,totals", oddsFormat="american", **window)
    try:
        return _odds_api(f"/sports/{key}/odds", bookmakers=",".join([config.BOOK, *config.BOOK_WEIGHTS]), **q)
    except RuntimeError as e:
        if "422" not in str(e):
            raise
        return _odds_api(f"/sports/{key}/odds", regions="us", **q)  # a retired book key must not cost the day


def scores(sport):
    """Backup settlement: {event id: (home, away)} for finished games."""
    out = {}
    for ev in _odds_api(f"/sports/{config.SPORTS[sport]}/scores", daysFrom=3):
        s = {x["name"]: int(float(x["score"])) for x in ev.get("scores") or []}
        if ev.get("completed") and len(s) == 2:
            out[ev["id"]] = (s[ev["home_team"]], s[ev["away_team"]])
    return out


# --- schedules, results, box scores ------------------------------------------

BOX = dict(field_goals_attempted="fga", free_throws_attempted="fta", free_throws_made="ftm",
           three_point_field_goals_attempted="tpa", three_point_field_goals_made="tpm", offensive_rebounds="orb")
DEAD = ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FORFEIT")


def season_of(sport, day):
    if sport == "MLB":
        return day.year
    if sport == "NFL":
        return day.year - (day.month < 8)
    return day.year + (day.month >= 9)


def hoops_frame(sport, sch, box):
    g = pd.DataFrame(dict(
        game_id=sch.game_id.astype(int), season=sch.season.astype(int), playoff=sch.season_type.astype(int) > 2,
        date=pd.to_datetime(sch.game_date), start=pd.to_datetime(sch.date, utc=True),
        h_id=sch.home_id.astype(str), a_id=sch.away_id.astype(str),
        h_name=sch.home_display_name, a_name=sch.away_display_name,
        neutral=sch.neutral_site.fillna(False).astype(bool),
        periods=pd.to_numeric(sch.status_period, errors="coerce").fillna(0), status=sch.status_type_name))
    done = g.status == "STATUS_FINAL"
    g["h_pts"] = pd.to_numeric(sch.home_score, errors="coerce").where(done)
    g["a_pts"] = pd.to_numeric(sch.away_score, errors="coerce").where(done)
    g = g[(g.h_name != "TBD") & (g.a_name != "TBD")].drop_duplicates("game_id")
    b = box[["game_id", "team_id", *BOX, "total_turnovers", "turnovers"]].apply(pd.to_numeric, errors="coerce")
    b = b.rename(columns=BOX)
    b["poss"] = b.fga - b.orb + b.total_turnovers.where(b.total_turnovers > 0, b.turnovers) + 0.46 * b.fta
    b = b.assign(team_id=b.team_id.astype(int).astype(str))[["game_id", "team_id", "poss", "tpa", "tpm", "fta", "ftm"]]
    b = b.dropna().drop_duplicates(["game_id", "team_id"])
    for s in "ha":
        g = g.merge(b.rename(columns={c: f"{s}_{c}" for c in b if c != "game_id"} | {"team_id": s + "_id"}),
                    on=["game_id", s + "_id"], how="left")
    # league members play a full schedule; everyone else (D-II, All-Star squads) shares one rating
    n = pd.concat([g[["season", s + "_id"]].set_axis(["season", "t"], axis=1) for s in "ha"]).value_counts()
    member = set(n[n >= 12].index)
    for s in "ha":
        g[s + "_key"] = [t if (y, t) in member else "X" for y, t in zip(g.season, g[s + "_id"])]
    if sport == "NBA":
        g = g[(g.h_key != "X") & (g.a_key != "X")]
    return g


def hoops(sport, season):
    lg = "nba" if sport == "NBA" else "mbb"
    base, sch, box = f"{RAW}/sportsdataverse/hoopR-{lg}-data/main/{lg}", [], []
    for y in range(season - 2, season + 1):
        keep = lambda name: f"{lg}_{name}_{y}.parquet" if y < season else None
        sch.append(pd.read_parquet(io.BytesIO(get(f"{base}/schedules/parquet/{lg}_schedule_{y}.parquet", keep("s")))))
        try:
            box.append(pd.read_parquet(io.BytesIO(get(f"{base}/team_box/parquet/team_box_{y}.parquet", keep("b")))))
        except RuntimeError as e:  # only opening night may lack box scores
            if y < season or "404" not in str(e) or (sch[-1].status_type_name == "STATUS_FINAL").any():
                raise
    return hoops_frame(sport, pd.concat(sch, ignore_index=True), pd.concat(box, ignore_index=True))


NFL_NAMES = dict(
    ARI="Arizona Cardinals", ATL="Atlanta Falcons", BAL="Baltimore Ravens", BUF="Buffalo Bills", CAR="Carolina Panthers",
    CHI="Chicago Bears", CIN="Cincinnati Bengals", CLE="Cleveland Browns", DAL="Dallas Cowboys", DEN="Denver Broncos",
    DET="Detroit Lions", GB="Green Bay Packers", HOU="Houston Texans", IND="Indianapolis Colts", JAX="Jacksonville Jaguars",
    KC="Kansas City Chiefs", LA="Los Angeles Rams", LAC="Los Angeles Chargers", LV="Las Vegas Raiders", MIA="Miami Dolphins",
    MIN="Minnesota Vikings", NE="New England Patriots", NO="New Orleans Saints", NYG="New York Giants", NYJ="New York Jets",
    PHI="Philadelphia Eagles", PIT="Pittsburgh Steelers", SEA="Seattle Seahawks", SF="San Francisco 49ers",
    TB="Tampa Bay Buccaneers", TEN="Tennessee Titans", WAS="Washington Commanders")
STADIUMS = dict(
    BAL00=(39.278, -76.623), BOS00=(42.091, -71.264), BUF00=(42.774, -78.787), CAR00=(35.226, -80.853),
    CHI98=(41.862, -87.617), CIN00=(39.095, -84.516), CLE00=(41.506, -81.700), DEN00=(39.744, -105.020),
    GNB00=(44.501, -88.062), JAX00=(30.324, -81.637), KAN00=(39.049, -94.484), MIA00=(25.958, -80.239),
    NAS00=(36.166, -86.771), NYC01=(40.814, -74.074), PHI00=(39.901, -75.168), PIT00=(40.447, -80.016),
    SEA00=(47.595, -122.332), SFO01=(37.403, -121.970), TAM00=(27.976, -82.503), WAS00=(38.908, -76.865))


def nfl_frame(n):
    moved = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
    start = pd.to_datetime(n.gameday + " " + n.gametime.fillna("13:00")).dt.tz_localize("America/New_York")
    g = pd.DataFrame(dict(
        game_id=n.game_id, season=n.season, playoff=n.game_type != "REG", date=pd.to_datetime(n.gameday),
        start=start.dt.tz_convert("UTC"), h_id=n.home_team.replace(moved), a_id=n.away_team.replace(moved),
        neutral=n.location == "Neutral", h_pts=n.home_score, a_pts=n.away_score, periods=4,
        line_m=n.spread_line, line_t=n.total_line, div_game=n.div_game, stadium=n.stadium_id,
        outdoors=n.roof == "outdoors", wind=n.wind.fillna(0) * n.roof.isin(["outdoors", "open"]),
        h_qb=n.home_qb_name, a_qb=n.away_qb_name,
        **{c: n[c] for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
                             "home_moneyline", "away_moneyline")}))
    g["h_name"], g["a_name"] = g.h_id.map(NFL_NAMES), g.a_id.map(NFL_NAMES)
    g["status"] = np.where(g.h_pts.notna(), "STATUS_FINAL", "STATUS_SCHEDULED")
    return g


def nfl(season):
    n = pd.read_csv(io.BytesIO(get(f"{RAW}/nflverse/nfldata/master/data/games.csv")))
    return nfl_frame(n[n.season >= season - 3])


def wind(stadium, start):
    """Forecast mph over the first three hours of the game, or None."""
    if stadium not in STADIUMS:
        return None
    lat, lon = STADIUMS[stadium]
    h = json.loads(get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                       "&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=GMT&forecast_days=2", tries=1, timeout=10))["hourly"]
    i = h["time"].index(start.strftime("%Y-%m-%dT%H:00"))
    return float(np.mean(h["wind_speed_10m"][i:i + 3]))


def mlb_frame(payloads):
    rows = {}
    for p in payloads:
        for d in p.get("dates", []):
            for x in d["games"]:
                if x["gameType"] not in ("R", "F", "D", "L", "W"):
                    continue
                h, a, state = x["teams"]["home"], x["teams"]["away"], x["status"].get("detailedState", "")
                dead = state.startswith(("Postponed", "Cancelled"))
                done = x["status"].get("abstractGameState") == "Final" and not dead and "score" in h and not state.startswith("Suspended")
                ls = x.get("linescore", {})
                # a postponed game keeps its gamePk on the make-up date, so the date is part of the id
                rows[f'{x["gamePk"]}_{d["date"]}'] = dict(
                    season=int(x["season"]), playoff=x["gameType"] != "R", date=pd.Timestamp(d["date"]),
                    start=pd.Timestamp(x["gameDate"]), h_id=str(h["team"]["id"]), a_id=str(a["team"]["id"]),
                    h_name=h["team"]["name"], a_name=a["team"]["name"], neutral=False,
                    h_pts=h["score"] if done else np.nan, a_pts=a["score"] if done else np.nan,
                    number=x.get("gameNumber", 1), repeat="resumedFrom" in x, periods=ls.get("currentInning", 9), full=ls.get("scheduledInnings", 9),
                    status="STATUS_POSTPONED" if dead else "STATUS_FINAL" if done else "STATUS_SCHEDULED",
                    h_sp=h.get("probablePitcher", {}).get("fullName"), a_sp=a.get("probablePitcher", {}).get("fullName"))
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("game_id").reset_index()


def mlb(season):
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R,F,D,L,W&hydrate=linescore,probablePitcher&season={}"
    return mlb_frame([json.loads(get(url.format(y), f"mlb_{y}.json" if y < season else None))
                      for y in (season - 1, season)])


def games(sport, day):
    s = season_of(sport, day)
    return {"NFL": nfl, "MLB": mlb}[sport](s) if sport in ("NFL", "MLB") else hoops(sport, s)
