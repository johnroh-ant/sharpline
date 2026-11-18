"""Price FanDuel's markets, pick each game's best side and total, build the card."""
from . import config, dist


def confidence(ev, price):
    """Kelly-proportional: a thin edge on a likely winner outranks the same edge on a longshot."""
    b = dist.dec(price) - 1
    if ev <= 0:  # no edge: minimum stake, and rank likelier winners first when filling the card
        return 1, ev * b
    kelly = min(ev, config.MAX_EDGE) / b
    return max(1, min(100, round(100 * kelly / config.KELLY_AT_100))), kelly


def options(sport, market, quote, mu):
    """Both sides of one FanDuel market priced at mu."""
    line, pa, pb = quote
    over, under = dist.get(sport, "total" if market == "totals" else "margin").probs(line, mu)
    out = []
    for side, price, win, lose in (("a", pa, over, under), ("b", pb, under, over)):
        ev = win * (dist.dec(price) - 1) - lose
        conf, kelly = confidence(ev, price)
        out.append(dict(market=market, side=side, line=line, price=price, p=win / (win + lose),
                        push=max(0.0, 1 - win - lose), ev=ev, conf=conf, kelly=kelly))
    return out


def best(opts):
    lo, hi = config.PRICE_RANGE
    ok = [o for o in opts if lo <= o["price"] <= hi]
    return max(ok, key=lambda o: o["kelly"]) if ok else None


def pick(candidates, n):
    return sorted(candidates, key=lambda c: c["kelly"], reverse=True)[:n]


def grade(market, side, line, home, away):
    v = (home + away if market == "totals" else home - away) - line
    v = v if side == "a" else -v
    return "won" if v > 0 else "lost" if v < 0 else "push"


def profit(status, stake, price):
    return {"won": stake * (dist.dec(price) - 1), "lost": -stake}.get(status, 0.0)
