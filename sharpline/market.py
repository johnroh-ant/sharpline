"""Turn many books' quotes into one sharp consensus per market."""
import numpy as np

from . import config, dist

OUTLIER = {"NFL": 1.5, "NBA": 2.0, "NCAAB": 2.5, "MLB": 0.6}


def parse_event(ev):
    """Odds API event -> {book: {market: (line, price_a, price_b)}}.

    Side a is home / over. `line` is the outcome threshold side a must beat:
    home margin > line for spreads (so home -3.5 is 3.5), total > line for totals.
    """
    home, out = ev["home_team"], {}
    for bk in ev.get("bookmakers", []):
        q = {}
        for mk in bk.get("markets", []):
            key, pairs = mk["key"], {}
            if key not in ("h2h", "spreads", "totals"):
                continue
            for o in mk.get("outcomes", []):
                is_a = o["name"] == ("Over" if key == "totals" else home)
                pt = o.get("point") or 0.0
                line = pt if key == "totals" else (-pt if is_a else pt) + 0.0
                pairs.setdefault(line, {})["a" if is_a else "b"] = o["price"]
            full = [(ln, p["a"], p["b"]) for ln, p in pairs.items()
                    if len(p) == 2 and min(abs(p["a"]), abs(p["b"])) >= 100]
            if full:  # exchanges list ladders; the main line is the most balanced one
                q[key] = min(full, key=lambda t: abs(dist.devig(t[1], t[2])[0] - 0.5))
        if q:
            out[bk["key"]] = q
    return out


def kind(market):
    return "total" if market == "totals" else "margin"


def book_mu(sport, market, quote):
    line, a, b = quote
    p, over = dist.devig(a, b)
    if not 0.98 <= over <= 1.09:
        return None, over
    if market == "h2h":
        k = dist.params()["ml_slope"].get(sport, 1.0)
        p = 1 / (1 + ((1 - p) / p) ** k)
    return dist.get(sport, kind(market)).invert(line, p), over


def consensus(sport, market, quotes):
    """Weighted mu of every book except FanDuel -> (mu, n_books, rows for display)."""
    rows = []
    for book, w in config.BOOK_WEIGHTS.items():
        q = quotes.get(book, {}).get(market)
        if q is None:
            continue
        # a run line only carries information about the same side of +/-1.5
        fd = quotes.get(config.BOOK, {}).get(market)
        if sport == "MLB" and market == "spreads" and (fd is None or q[0] != fd[0]):
            continue
        mu, over = book_mu(sport, market, q)
        if mu is not None:  # a wide quote is a thin market: trust it less
            rows.append([book, mu, w * min(1.0, 0.05 / max(over - 1, 1e-3)), q])
    if len(rows) >= 3:
        med = float(np.median([r[1] for r in rows]))
        rows = [r for r in rows if abs(r[1] - med) <= OUTLIER[sport] * (1.5 if market == "totals" else 1)]
    if len(rows) < config.MIN_BOOKS:
        return None, len(rows), rows
    w = np.array([r[2] for r in rows])
    return float(w @ np.array([r[1] for r in rows]) / w.sum()), len(rows), rows
