"""Match an odds-feed event to its row in the schedule feed."""
import difflib
import re
import unicodedata

import pandas as pd

ALIASES = {
    "umkc kangaroos": "kansas city roos", "iupui jaguars": "iu indianapolis jaguars",
    "liu sharks": "long island university sharks", "tenn martin skyhawks": "ut martin skyhawks",
    "gw revolutionaries": "george washington revolutionaries", "long beach st 49ers": "long beach st beach",
    "grand canyon antelopes": "grand canyon lopes", "texas a m cc islanders": "texas a m corpus christi islanders",
    "texas a m commerce lions": "east texas a m lions", "appalachian st mountaineers": "app st mountaineers",
    "florida int l golden panthers": "florida international panthers", "n colorado bears": "northern colorado bears",
    "miss valley st delta devils": "mississippi valley st delta devils", "houston baptist huskies": "houston christian huskies",
    "los angeles clippers": "la clippers", "oakland athletics": "athletics",
}


def norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\b(state|saint)\b", "st", s).replace("csu ", "cal st ").replace("mt st", "mount st")
    return ALIASES.get(s, s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    return 0.5 * difflib.SequenceMatcher(None, a, b).ratio() + 0.5 * len(ta & tb) / min(len(ta), len(tb))


def find(games, home, away, start, hours=14, taken=()):
    """(row index, flipped) of the best matching scheduled game, or (None, False)."""
    near = games[((games.start - start).abs() <= pd.Timedelta(hours=hours)) & ~games.index.isin(taken)]
    best, score = (None, False), 0.0
    for r in near.itertuples():
        # doubleheaders: nearest start wins, and game 1 wins when game 2 is listed minutes later as a placeholder
        late = round(abs((r.start - start).total_seconds()) / 3600) * 0.002 + getattr(r, "number", 1) * 1e-4
        for flip, (h, a) in enumerate(((r.h_name, r.a_name), (r.a_name, r.h_name))):
            sh, sa = sim(home, h), sim(away, a)
            s = (sh + sa) / 2 - late
            if min(sh, sa) >= 0.55 and s > score:
                best, score = (r.Index, bool(flip)), s
    return best if score >= 0.72 else (None, False)


def team(games, name):
    """Team id by name alone, for games the schedule still lists as TBD."""
    last = games[games.season == games.season.max()]
    names = dict(zip(last.h_name, last.h_id)) | dict(zip(last.a_name, last.a_id))
    ranked = sorted(((sim(name, n), n) for n in names), reverse=True)
    if ranked and ranked[0][0] >= 0.85 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.05):
        return names[ranked[0][1]], ranked[0][1]
    return None
