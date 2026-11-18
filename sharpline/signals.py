"""Signals: reasons the sharp consensus itself may be off, in points.

Each feature is a function of the ratings row r and the market's margin L / total T.
Only features that held up out of sample against closing lines are here.
"""
import numpy as np

c = np.clip
b2b = lambda v: (v == 0) * 1.0
unseen = lambda r: (np.minimum(r.h_gp, r.a_gp) <= 2) * 1.0  # transfer-era rosters: last year says little
early = lambda r: ((r.h_gp + r.a_gp) < 16) * 1.0

FEATURES = {
    ("NBA", "margin"): {
        "luck": lambda r, L, T: c(r.m_adj - r.m_raw, -6, 6),
        "power": lambda r, L, T: c(r.m_adj - L, -10, 10),
        "mkt_hist": lambda r, L, T: c(r.m_mkt - L, -10, 10),
        "early_power": lambda r, L, T: c(r.m_adj - L, -10, 10) * early(r),
        "last_cover": lambda r, L, T: c(r.h_last - r.a_last, -30, 30),
        "fav": lambda r, L, T: L,
    },
    ("NBA", "total"): {
        "luck": lambda r, L, T: c(r.t_adj - r.t_raw, -8, 8),
        "power": lambda r, L, T: c(r.t_adj - T, -14, 14),
        "mkt_hist": lambda r, L, T: c(r.t_mkt - T, -14, 14),
        "high_total": lambda r, L, T: T - r.t_lg,
        "mismatch": lambda r, L, T: np.abs(L) - 6,
        "playoff": lambda r, L, T: r.playoff * 1.0,
    },
    ("NCAAB", "margin"): {
        "luck": lambda r, L, T: c(r.m_adj - r.m_raw, -6, 6),
        "power": lambda r, L, T: c(r.m_adj - L, -10, 10),
        "unseen": lambda r, L, T: c(r.m_adj - L, -10, 10) * unseen(r),
        "fav": lambda r, L, T: c(L, -25, 25),
        "b2b": lambda r, L, T: b2b(r.h_rest) - b2b(r.a_rest),
        "busy": lambda r, L, T: r.h_g4 - r.a_g4,
    },
    ("NCAAB", "total"): {
        "luck": lambda r, L, T: c(r.t_adj - r.t_raw, -8, 8),
        "power": lambda r, L, T: c(r.t_adj - T, -14, 14),
        "unseen": lambda r, L, T: c(r.t_adj - T, -14, 14) * unseen(r),
        "b2b": lambda r, L, T: b2b(r.h_rest) + b2b(r.a_rest),
        "early": lambda r, L, T: early(r),
    },
    ("NFL", "margin"): {
        "mkt_hist": lambda r, L, T: c(r.m_mkt - L, -8, 8),
        "rest": lambda r, L, T: c(r.h_rest - r.a_rest, -7, 7),
        "fav": lambda r, L, T: L,
    },
    ("NFL", "total"): {
        "wind": lambda r, L, T: c(r.wind - 7, 0, 8),
        "high_total": lambda r, L, T: T - 44,
        "division": lambda r, L, T: r.div_game * 1.0,
    },
    # Pitcher, bullpen and lineup stats added nothing beyond the MLB market, so
    # these start near zero weight and only matter if live results earn it.
    ("MLB", "margin"): {"power": lambda r, L, T: c(r.m_raw - L, -1.5, 1.5)},
    ("MLB", "total"): {"power": lambda r, L, T: c(r.t_raw - T, -2, 2), "high_total": lambda r, L, T: T - 8.7},
}

CLIP = {("NBA", "margin"): 13, ("NBA", "total"): 18, ("NCAAB", "margin"): 11, ("NCAAB", "total"): 16,
        ("NFL", "margin"): 13, ("NFL", "total"): 13, ("MLB", "margin"): 4, ("MLB", "total"): 4}
# beyond this gap the rating is missing news the market has (or the match is wrong)
MAX_GAP = {"NBA": 14, "NCAAB": 14, "NFL": 14, "MLB": 3}

LABELS = {
    "margin": {"luck": "Shooting luck due to even out", "power": "Efficiency rating vs the line",
               "mkt_hist": "Line has moved past how the market usually rates these teams",
               "early_power": "Early-season ratings gap", "unseen": "Discount: under 3 games of data", "last_cover": "Overreaction to last game",
               "fav": "Favorite/underdog pricing", "b2b": "Back-to-back", "busy": "Game rhythm (games in last 4 days)",
               "rest": "Rest edge", "bias": "Home/road pricing"},
    "total": {"luck": "Shooting luck due to even out", "power": "Pace x efficiency projection",
              "mkt_hist": "Total vs recent market totals", "high_total": "Extreme totals regress",
              "mismatch": "Blowouts run high", "playoff": "Playoff pace", "b2b": "Tired legs",
              "early": "Early-season scoring", "unseen": "Discount: under 3 games of data", "wind": "Wind", "division": "Division game", "bias": "Over/under pricing"},
}


def names(sport, kind):
    return ["bias"] + list(FEATURES[sport, kind])


def matrix(sport, kind, r, L, T):
    """Works on a DataFrame (training) or a single row (live)."""
    cols = [np.ones(np.shape(L))] + [np.asarray(f(r, L, T), float) * np.ones(np.shape(L))
                                     for f in FEATURES[sport, kind].values()]
    return np.nan_to_num(np.stack(cols, axis=-1))
