"""python -m sharpline.run [auto|morning|afternoon] [--force]

Logs stay free of picks: Actions logs on a public repo are public.
"""
import calendar
import datetime as dt
import sys

import numpy as np
import pandas as pd

from . import card, config, dist, feeds, learn, market, match, ratings, signals, store

MARKETS = ("spreads", "h2h", "totals")
r2 = lambda v, n=2: None if v is None or v != v else round(float(v), n)


def load():
    state = store.load("state") or dict(learn=learn.fresh(), lines={}, pending=[], last_run=None)
    index = store.load("index") or dict(bankroll0=config.BANKROLL, bets=[], days=[], runs=[])
    return state, index


def save(state, index, now):
    index["updated"] = now.isoformat(timespec="seconds")
    index["credits"] = feeds.credits["remaining"] if feeds.credits["remaining"] is not None else index.get("credits")
    index["runs"] = index["runs"][-60:]
    prior = learn.fresh()
    index["weights"] = [
        dict(sport=s, kind=k, n=state["learn"]["signals"][f"{s}_{k}"]["n"], rows=[
            dict(label=signals.LABELS[k][f], start=r2(a, 3), now=r2(b, 3)) for f, a, b in
            zip(signals.names(s, k), learn.weights(prior, s, k), learn.weights(state["learn"], s, k))])
        for s, k in signals.FEATURES]
    index["trust"] = [dict(sport=s, market=m, start=r2(learn.trust(prior, s, m)), now=r2(learn.trust(state["learn"], s, m)),
                           n=state["learn"]["trust"][f"{s}_{m}"]["n"]) for s in config.SPORTS for m in MARKETS]
    store.save("state", state)
    store.save("index", index)


class Frames(dict):
    """Schedule/result frames, fetched once per sport; None when the feed is down."""

    def __init__(self, day, notes):
        self.day, self.notes = day, notes

    def __missing__(self, sport):
        try:
            self[sport] = feeds.games(sport, self.day).reset_index(drop=True)
        except Exception as e:
            self.notes.append(f"{sport} stats feed failed: {e}")
            self[sport] = None
        return self[sport]


# --- settle and learn ---------------------------------------------------------

def result(g, frames, backup, now):
    """(status, home, away, shortened) with scores in the odds feed's home/away order."""
    f, start = frames[g["sport"]], pd.Timestamp(g["start"])
    if f is not None:
        i, flip = match.find(f, g["home"], g["away"], start)
        if g["sport"] == "MLB" and g.get("row"):  # doubleheaders: the row chosen when the bet was placed
            i = next(iter(f.index[f.game_id == g["row"]]), None)
        if i is not None:
            r = f.loc[i]
            if r.status == "STATUS_FINAL":
                h, a = (r.a_pts, r.h_pts) if flip else (r.h_pts, r.a_pts)
                return "final", int(h), int(a), g["sport"] == "MLB" and r.periods < r.full
            if r.status in feeds.DEAD:
                return "void", None, None, False
    age = now - start
    if g["has_bet"] and age > pd.Timedelta(hours=8):
        if g["sport"] not in backup:
            try:
                backup[g["sport"]] = feeds.scores(g["sport"])
            except Exception as e:
                frames.notes.append(f'{g["sport"]} backup scores failed: {e}')
                backup[g["sport"]] = {}
        if g["id"] in backup[g["sport"]]:
            return ("final", *backup[g["sport"]][g["id"]], False)
    return ("void" if age > pd.Timedelta(days=config.VOID_AFTER_DAYS) else "pending"), None, None, False


def record_close(state, g, bets, market_, mu, fd_later):
    m = g["markets"].get(market_)
    if not m or mu is None or m.get("close") is not None:
        return
    m["close"] = r2(mu, 3)
    if m["sharp"] is not None and fd_later is not None:
        learn.learn_close(state["learn"], g["sport"], market_, m["fd"], m["sharp"], fd_later, mu)
    for b in bets:
        if b["game"] == g["id"] and b["market"] == market_:
            # how far the sharp win probability for our side moved after we bet
            then, later = (card.options(g["sport"], market_, m["quote"], v)[b["side"] == "b"]["p"] for v in (m["fd"] if m["sharp"] is None else m["sharp"], mu))
            b["clv"] = r2(later - then, 4)


def nfl_close(state, g, bets, frames):
    f = frames["NFL"]
    i, flip = match.find(f, g["home"], g["away"], pd.Timestamp(g["start"])) if f is not None else (None, 0)
    if i is None or flip:
        return
    r = f.loc[i]
    for mk, q in (("spreads", (r.line_m, r.home_spread_odds, r.away_spread_odds)),
                  ("totals", (r.line_t, r.over_odds, r.under_odds)), ("h2h", (0.0, r.home_moneyline, r.away_moneyline))):
        if not np.isnan(q).any():
            mu = market.book_mu("NFL", mk, q)[0]  # one independent closing line stands in for both
            record_close(state, g, bets, mk, mu, mu)


def settle(state, index, frames, now):
    backup, still, graded, learned = {}, [], 0, 0
    by_id = {b["id"]: b for b in index["bets"]}
    for date in sorted({d for d, _ in state["pending"]}):
        day = store.load(f"days/{date}")
        waiting = {gid for d, gid in state["pending"] if d == date}
        for g in day["games"]:
            if g["id"] not in waiting:
                continue
            status, h, a, short = result(g, frames, backup, now) if pd.Timestamp(g["start"]) < now else ("pending",) * 4
            if status == "pending":
                still.append([date, g["id"]])
                continue
            g["status"], g["score"] = status, None if h is None else [h, a]
            if status == "final":
                if g["sport"] == "NFL":
                    nfl_close(state, g, day["bets"], frames)
                for k, outcome in (("margin", h - a), ("total", h + a)):
                    if g["x"].get(k) and not short:
                        learn.learn_result(state["learn"], g["sport"], k, g["x"][k], g["base"][k], outcome)
                learned += 1
            for b in day["bets"]:
                if b["game"] != g["id"]:
                    continue
                st = "void" if status == "void" else card.grade(b["market"], b["side"], b["line"], h, a)
                # FanDuel rule for rain-shortened games: moneylines stand, run lines void, totals stand only if already over
                if short and (b["market"] == "spreads" or (b["market"] == "totals" and h + a <= b["line"])):
                    st = "void"
                b.update(status=st, profit=r2(card.profit(st, b["stake"], b["price"])), score=g["score"])
                by_id[b["id"]].update({k: b.get(k) for k in ("status", "profit", "score", "clv")})
                graded += 1
        store.save(f"days/{date}", day)
    state["pending"] = still
    return graded, learned


# --- analyse -------------------------------------------------------------------

def rate(sport, events, state, frames):
    """Ratings row per event id. Adds rows for games the schedule has not filled in yet."""
    f = frames[sport]
    if f is None:
        return {}
    f, where, taken = f.copy(), {}, []
    for ev in sorted(events, key=lambda e: e["commence_time"]):
        start = pd.Timestamp(ev["commence_time"])
        i, flip = match.find(f[f.status == "STATUS_SCHEDULED"] if sport == "MLB" else f,
                             ev["home_team"], ev["away_team"], start, taken=taken)
        if i is None and sport == "NCAAB":
            h, a = match.team(f, ev["home_team"]), match.team(f, ev["away_team"])
            if h and a:
                i, flip = len(f), False
                f.loc[i, ["game_id", "season", "playoff", "date", "start", "h_id", "a_id", "h_key", "a_key", "h_name", "a_name", "neutral"]] = [
                    -i, f.season.max(), False, pd.Timestamp(frames.day), start, h[0], a[0], h[0], a[0], h[1], a[1], True]
        if i is not None:
            taken.append(i)
            where[ev["id"]] = (f.at[i, "game_id"], flip)
    f = f[(f.h_pts.notna() | f.game_id.isin([w[0] for w in where.values()])) & ~f.status.isin(feeds.DEAD)]
    if sport == "MLB":  # a resumed game is listed on both dates
        f = f[~(f.repeat & f.h_pts.notna())]
    if sport == "NFL":
        for k, c in (("margin", "m"), ("total", "t")):
            f["mk_" + c] = [dist.get(sport, k).invert(v, 0.5) if v == v else np.nan for v in f["line_" + c]]
    else:
        seed = pd.read_csv(config.PARAMS / "nba_lines.csv") if sport == "NBA" else pd.DataFrame(columns=["game_id", "m", "t"])
        lines = {str(r.game_id): (r.m, r.t) for r in seed.itertuples()} | state["lines"].get(sport, {})
        ids = [str(int(i)) if sport != "MLB" else i for i in f.game_id]
        f["mk_m"], f["mk_t"] = ([np.nan if lines.get(i, (None, None))[j] is None else lines[i][j] for i in ids] for j in (0, 1))
    f = ratings.run(sport, f.astype({"neutral": bool, "playoff": bool, "season": int})).set_index("game_id", drop=False)
    return {e: (f.loc[gid].copy(), flip) for e, (gid, flip) in where.items() if gid in f.index}


def analyse(sport, ev, rated, state, notes):
    quotes, L = market.parse_event(ev), state["learn"]
    g = dict(id=ev["id"], sport=sport, start=ev["commence_time"], home=ev["home_team"], away=ev["away_team"],
             markets={}, x={}, base={}, why={}, info={}, status="pending", has_bet=False)
    for mk in MARKETS:
        q = quotes.get(config.BOOK, {}).get(mk)
        fd = market.book_mu(sport, mk, q)[0] if q else None
        if fd is None:
            continue
        sharp, n, rows = market.consensus(sport, mk, quotes)
        base = fd if sharp is None else fd + learn.trust(L, sport, mk) * (sharp - fd)
        g["markets"][mk] = dict(quote=list(q), fd=r2(fd, 3), sharp=r2(sharp, 3), books=n, base=r2(base, 3),
                                others=[[b, *qq] for b, _, _, qq in rows])
    M = g["markets"]
    if sport == "MLB" and ev["id"] in rated:
        g["row"] = rated[ev["id"]][0].game_id
    side = M.get("h2h") if sport == "MLB" else M.get("spreads") or M.get("h2h")
    shift = {"margin": 0.0, "total": 0.0}
    if ev["id"] in rated and side and "totals" in M:
        r, flip = rated[ev["id"]]
        sign = -1 if flip else 1  # neutral sites: the two feeds may disagree on who is "home"
        line, total = sign * side["base"], M["totals"]["base"]
        if sport == "NFL":
            try:
                up = r.outdoors and not any(n.startswith("wind") for n in notes)  # one failure is enough
                r["wind"] = (feeds.wind(r.stadium, r.start) if up else 0) or 0
            except Exception as e:
                notes.append(f"wind forecast failed: {e}")
                r["wind"] = 0
            g["info"].update(wind=r2(r.wind, 0) if r.outdoors else None, qbs=[x if x == x else None for x in (r.h_qb, r.a_qb)][::sign])
        if sport == "MLB":
            g["info"]["pitchers"] = [x if x == x else None for x in (r.h_sp, r.a_sp)][::sign]
        own = "adj" if sport in ratings.HOOPS else "raw"
        g["info"].update(own_margin=r2(sign * r["m_" + own], 1), own_total=r2(r["t_" + own], 1))
        if sport in ratings.HOOPS:
            g["info"].update(pace=r2(r.pace, 1), luck=[r2(r.h_net_raw - r.h_net_adj, 1), r2(r.a_net_raw - r.a_net_adj, 1)][::sign])
        known = r.get("h_key") != "X" and r.get("a_key") != "X"
        if known and abs(r.m_raw - line) < signals.MAX_GAP[sport]:  # NaN ratings fail this too
            for k in shift:
                x = signals.matrix(sport, k, r, line, total)
                parts = x * learn.weights(L, sport, k) * (sign if k == "margin" else 1)
                shift[k] = float(parts.sum())
                g["why"][k] = [[signals.LABELS[k][f], r2(p)] for f, p in zip(signals.names(sport, k), parts)]
                # stored in the odds feed's orientation so settlement can learn from it directly
                g["x"][k] = [r2(v, 4) for v in x * (sign if k == "margin" else 1)]
            g["base"] = {"margin": side["base"], "total": total}
        if sport == "NBA":
            state["lines"].setdefault(sport, {})[str(int(r.game_id))] = [r2(line, 2), r2(total, 2)]
            g["feed"] = [str(int(r.game_id)), sign]
    opts = {}
    for mk, m in M.items():
        k = market.kind(mk)
        m["fair"] = r2(m["base"] + shift[k], 3)
        m["fair_line"] = r2(dist.get(sport, k).fair_line(m["fair"]), 1)
        m["sharp_line"] = None if m["sharp"] is None else r2(dist.get(sport, k).fair_line(m["sharp"]), 1)
        opts[mk] = card.options(sport, mk, m["quote"], m["fair"])
        m["p"], m["ev"] = r2(opts[mk][0]["p"], 4), [r2(o["ev"], 4) for o in opts[mk]]
        if m["sharp"] is not None:
            m["sharp_p"] = r2(card.options(sport, mk, m["quote"], m["sharp"])[0]["p"], 4)
    picks = [card.best(opts.get("spreads", []) + opts.get("h2h", [])), card.best(opts.get("totals", []))]
    return g, [dict(p, g=g) for p in picks if p]


def describe(c):
    g, a = c["g"], c["side"] == "a"
    team = g["home"] if a else g["away"]
    if c["market"] == "totals":
        return f'{"Over" if a else "Under"} {c["line"]:g}'
    return f"{team} ML" if c["market"] == "h2h" else f'{team} {(-c["line"] if a else c["line"]) + 0:+g}'


def reasons(c):
    """Points of value toward the selection, split into market gap and each signal."""
    g, s = c["g"], 1 if c["side"] == "a" else -1
    m = g["markets"][c["market"]]
    rows = [["FanDuel vs sharp books" if m["sharp"] is not None else "No sharp price posted", r2(s * (m["base"] - m["fd"]))]]
    rows += [[k, r2(s * v)] for k, v in g["why"].get(market.kind(c["market"]), []) if abs(v) >= 0.02]
    return sorted(rows, key=lambda r: -abs(r[1]))


def morning(now, force=False):
    local = now.astimezone(config.TZ)
    date = local.date().isoformat()
    state, index = load()
    if not force and (state["last_run"] == date or local.strftime("%H:%M") < config.RUN_AFTER):
        return print("nothing to do")
    notes = []
    frames = Frames(local.date(), notes)
    graded, learned = settle(state, index, frames, now)
    save(state, index, now)  # results are banked even if the rest of the run fails
    day = store.load(f"days/{date}") or dict(date=date, games=[], bets=[])
    midnight = dt.datetime.combine(local.date() + dt.timedelta(days=1), dt.time(), config.TZ)
    seen, cands, slate, failed = {g["id"] for g in day["games"]}, [], {}, []
    for sport in config.SPORTS:
        try:
            slate[sport] = feeds.odds(sport, now + dt.timedelta(minutes=5), midnight - dt.timedelta(seconds=1))
        except Exception as e:
            notes.append(f"{sport} odds failed: {e}")
            failed.append(sport)
    # The top ten needs the whole board, so wait for a backup trigger. Two retries bound the credit cost.
    tries = state.get("tries", [None, 0])
    state["tries"] = [date, tries[1] + 1 if tries[0] == date else 1]
    if failed and (state["tries"][1] < 3 or not any(slate.values())) and not force:
        index["runs"].append(dict(at=now.isoformat(timespec="seconds"), kind="morning (will retry)", graded=graded, notes=notes))
        save(state, index, now)
        (config.ROOT / ".handled").touch()
        sys.exit("\n".join(notes))
    for sport, events in slate.items():
        events = [e for e in events if e["id"] not in seen]
        if not events:
            continue
        try:
            rated = rate(sport, events, state, frames)
        except Exception as e:
            notes.append(f"{sport} ratings failed ({type(e).__name__}); priced on the market alone")
            rated, frames[sport] = {}, None
        for ev in events:
            # pro exhibitions (All-Star games, Pro Bowl) never appear in the league schedule feeds
            if ev["id"] not in rated and sport != "NCAAB" and frames[sport] is not None:
                continue
            try:
                g, cs = analyse(sport, ev, rated, state, notes)
            except Exception as e:
                notes.append(f"{sport} game skipped ({type(e).__name__})")
                continue
            if g["markets"]:
                day["games"].append(g)
                state["pending"].append([date, g["id"]])
                cands += cs
    for c in card.pick(cands, config.BETS_PER_DAY - len(day["bets"])):
        g = c["g"]
        g["has_bet"] = True
        b = dict(id=f'{g["id"]}:{c["market"]}', date=date, game=g["id"], sport=g["sport"], start=g["start"],
                 home=g["home"], away=g["away"], market=c["market"], side=c["side"], line=c["line"], pick=describe(c),
                 price=c["price"], p=r2(c["p"], 4), ev=r2(c["ev"], 4), conf=c["conf"], stake=c["conf"],
                 status="pending", profit=None, score=None, clv=None)
        index["bets"].append(dict(b))
        day["bets"].append(dict(b, fair_price=dist.american(c["p"]), reasons=reasons(c)))
    store.save(f"days/{date}", day)
    if date not in index["days"]:
        index["days"].append(date)
    state["last_run"] = date
    index["runs"].append(dict(at=now.isoformat(timespec="seconds"), kind="morning", graded=graded, learned=learned,
                              games=len(day["games"]), bets=len(day["bets"]), notes=notes))
    save(state, index, now)
    print(f"graded {graded}, learned from {learned}, analysed {len(day['games'])}, bets {len(day['bets'])}", *notes, sep="\n")


def afternoon(now, force=False):
    """Re-price today's unstarted games: did the market move toward our bets?"""
    local = now.astimezone(config.TZ)
    date = local.date().isoformat()
    state, index = load()
    if not force and (state.get("last_afternoon") == date or local.strftime("%H:%M") < config.RECHECK_AFTER):
        return print("nothing to do")
    day = store.load(f"days/{date}")
    open_ = [g for g in (day or {}).get("games", []) if g["sport"] != "NFL" and pd.Timestamp(g["start"]) > now
             and not any("close" in m for m in g["markets"].values())]
    sports = sorted({g["sport"] for g in open_})
    days_left = calendar.monthrange(local.year, local.month)[1] - local.day
    reserve = days_left * (3 * len({g["sport"] for g in (day or {}).get("games", [])} - {"NFL"}) + 3) + 15
    if not sports or (not force and (index.get("credits") or 0) - 3 * len(sports) < reserve):
        return print("nothing to do")
    by_id, notes = {b["id"]: b for b in index["bets"]}, []
    for sport in sports:
        try:
            events = {e["id"]: e for e in feeds.odds(sport, now, now + dt.timedelta(hours=12))}
        except Exception as e:
            notes.append(f"{sport} odds failed: {e}")
            continue
        for g in open_:
            if g["id"] not in events:
                continue
            quotes = market.parse_event(events[g["id"]])
            for mk in g["markets"]:
                q = quotes.get(config.BOOK, {}).get(mk)
                if sport == "MLB" and mk == "spreads" and (not q or q[0] != g["markets"][mk]["quote"][0]):
                    continue  # the run line flipped sides, so it is a different bet now
                record_close(state, g, day["bets"], mk, market.consensus(sport, mk, quotes)[0],
                             market.book_mu(sport, mk, q)[0] if q else None)
            m, t = (g["markets"].get(k, {}).get("close") for k in ("spreads", "totals"))
            if "feed" in g and m is not None and t is not None:
                state["lines"][sport][g["feed"][0]] = [r2(g["feed"][1] * m), r2(t)]
    for b in day["bets"]:
        by_id[b["id"]]["clv"] = b["clv"]
    store.save(f"days/{date}", day)
    state["last_afternoon"] = date
    index["runs"].append(dict(at=now.isoformat(timespec="seconds"), kind="afternoon", games=len(open_), notes=notes))
    save(state, index, now)
    print(f"re-priced {len(open_)} games", *notes, sep="\n")


def auto(now, force=False):
    """Every trigger tries both; each step gates itself, so a missed morning is caught up later."""
    ran_before = load()[0]["last_run"] == now.astimezone(config.TZ).date().isoformat()
    morning(now, force)
    if ran_before:
        afternoon(now)


if __name__ == "__main__":
    step = next((a for a in sys.argv[1:] if not a.startswith("-")), "auto")
    {"auto": auto, "morning": morning, "afternoon": afternoon}[step](dt.datetime.now(dt.timezone.utc), "--force" in sys.argv)
