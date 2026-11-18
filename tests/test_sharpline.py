import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from sharpline import card, config, dist, feeds, learn, market, match, run, store


def event(eid, home, away, start, books):
    """books: {key: (home_spread, h_price, a_price, total, over, under, h_ml, a_ml)}"""
    return dict(id=eid, home_team=home, away_team=away, commence_time=start, bookmakers=[
        dict(key=k, markets=[
            dict(key="spreads", outcomes=[dict(name=home, price=hp, point=s), dict(name=away, price=ap, point=-s)]),
            dict(key="totals", outcomes=[dict(name="Over", price=o, point=t), dict(name="Under", price=u, point=t)]),
            dict(key="h2h", outcomes=[dict(name=home, price=hm), dict(name=away, price=am)])])
        for k, (s, hp, ap, t, o, u, hm, am) in books.items()])


class Pricing(unittest.TestCase):
    def test_agreeing_with_fanduel_never_shows_an_edge(self):
        # the old model's line-to-probability round trip invented edges; this one must not
        for sport, mk, q in (("NFL", "spreads", (3.0, -110, -110)), ("NFL", "h2h", (0.0, -180, 150)),
                             ("NBA", "totals", (228.5, -112, -108)), ("NCAAB", "spreads", (-7.5, -105, -115)),
                             ("MLB", "spreads", (1.5, 140, -170)), ("MLB", "totals", (8.0, -115, -105))):
            mu = market.book_mu(sport, mk, q)[0]
            self.assertTrue(all(o["ev"] < -0.005 for o in card.options(sport, mk, q, mu)), (sport, mk))

    def test_nfl_three_is_worth_more_than_five(self):
        d = dist.get("NFL", "margin")
        push = lambda x: 1 - sum(d.probs(x, d.invert(x, .5)))
        self.assertGreater(push(3.0), 2.5 * push(5.0))

    def test_off_market_fanduel_line_is_an_edge_on_the_right_side(self):
        sharp = (-6.5, -105, -105, 150.5, -105, -105, -250, 220)
        q = market.parse_event(event("e", "H", "A", "", dict(fanduel=(-4.5, -110, -110, 150.5, -110, -110, -250, 200),
                                                             pinnacle=sharp, novig=sharp)))
        mu = market.consensus("NCAAB", "spreads", q)[0]
        best = card.best(card.options("NCAAB", "spreads", q["fanduel"]["spreads"], mu))
        self.assertEqual(best["side"], "a")
        self.assertGreater(best["ev"], 0.05)

    def test_outlier_book_is_ignored(self):
        ok, bad = (-3, -105, -105, 44, -105, -105, -150, 135), (7, -105, -105, 44, -105, -105, -150, 135)
        q = market.parse_event(event("e", "H", "A", "", dict(pinnacle=ok, novig=ok, kalshi=bad)))
        self.assertEqual(market.consensus("NFL", "spreads", q)[1], 2)

    def test_grading(self):
        self.assertEqual(card.grade("spreads", "a", 3.0, 24, 21), "push")
        self.assertEqual(card.grade("spreads", "b", 3.5, 24, 21), "won")
        self.assertEqual(card.grade("spreads", "a", -2.5, 20, 21), "won")
        self.assertEqual(card.grade("totals", "b", 44.5, 24, 21), "lost")
        self.assertEqual(card.grade("h2h", "a", 0.0, 20, 20), "push")
        self.assertAlmostEqual(card.profit("won", 55, -110), 50)

    def test_learning_moves_weights_toward_results(self):
        s = learn.fresh()
        before = learn.weights(s, "NBA", "total")[0]
        for _ in range(300):
            learn.learn_result(s, "NBA", "total", [1, 0, 0, 0, 0, 0, 0], 225.0, 240)
        self.assertGreater(learn.weights(s, "NBA", "total")[0], before + 0.1)


class Matching(unittest.TestCase):
    def test_feed_spellings(self):
        for a, b in (("UMKC Kangaroos", "Kansas City Roos"), ("CSU Fullerton Titans", "Cal State Fullerton Titans"),
                     ("Arkansas-Little Rock Trojans", "Little Rock Trojans"), ("San José St Spartans", "San José State Spartans"),
                     ("Hawai'i Rainbow Warriors", "Hawai'i Rainbow Warriors"), ("Albany Great Danes", "UAlbany Great Danes"),
                     ("Los Angeles Clippers", "LA Clippers"), ("Mt. St. Mary's Mountaineers", "Mount St. Mary's Mountaineers"),
                     ("SE Missouri St Redhawks", "Southeast Missouri State Redhawks"), ("Army Knights", "Army Black Knights")):
            self.assertGreater(match.sim(a, b), 0.68, (a, b))

    def test_similar_names_pick_the_right_game(self):
        t = pd.Timestamp("2026-12-01T00:00Z")
        g = pd.DataFrame(dict(start=[t, t], h_name=["Michigan Wolverines", "Michigan State Spartans"],
                              a_name=["Iowa Hawkeyes", "Iowa State Cyclones"]))
        self.assertEqual(match.find(g, "Michigan St Spartans", "Iowa State Cyclones", t), (1, False))
        self.assertEqual(match.find(g, "Iowa Hawkeyes", "Michigan Wolverines", t), (0, True))
        self.assertEqual(match.find(g, "Duke Blue Devils", "Iowa Hawkeyes", t), (None, False))


def mlb_game(pk, date, hour, h, a, hs=None, as_=None, state="Scheduled", innings=9):
    side = lambda i, n, s: dict(team=dict(id=i, name=n), probablePitcher=dict(fullName="P " + n), **({} if s is None else dict(score=s)))
    return dict(date=date, games=[dict(
        gamePk=pk, gameType="R", season=date[:4], gameDate=f"{date}T{hour}:05:00Z",
        status=dict(detailedState=state, abstractGameState="Preview" if state == "Scheduled" else "Final"),
        linescore=dict(currentInning=innings, scheduledInnings=9), teams=dict(home=side(*h, hs), away=side(*a, as_)))])


class TwoDays(unittest.TestCase):
    def test_bet_settle_void_and_rerun(self):
        NY, BOS, SF, LA, CHC, STL = ((1, "New York Yankees"), (2, "Boston Red Sox"), (3, "San Francisco Giants"),
                                     (4, "Los Angeles Dodgers"), (5, "Chicago Cubs"), (6, "St. Louis Cardinals"))
        past = [mlb_game(i, f"2027-05-{i:02d}", 23, *p, 4, 3, "Final") for i, p in enumerate(((NY, BOS), (SF, LA), (CHC, STL)), 1)]
        before = past + [mlb_game(10 + i, "2027-06-01", 23, *p) for i, p in enumerate(((NY, BOS), (SF, LA), (CHC, STL)))]
        after = past + [mlb_game(10, "2027-06-01", 23, NY, BOS, 6, 2, "Final"), mlb_game(11, "2027-06-01", 23, SF, LA, state="Postponed"),
                        mlb_game(12, "2027-06-01", 23, CHC, STL, 2, 1, "Completed Early", innings=6)]
        fd, sharp = (-1.5, 150, -180, 8.5, -105, -115, -125, 105), (-1.5, 140, -155, 9.0, -105, -105, -140, 128)
        evs = [event(f"e{i}", h[1], a[1], "2027-06-01T23:05:00Z", dict(fanduel=fd, pinnacle=sharp, novig=sharp))
               for i, (h, a) in enumerate(((NY, BOS), (SF, LA), (CHC, STL)))]
        evs.append(event("asg", "American League", "National League", "2027-06-01T23:05:00Z", dict(fanduel=fd)))
        day1, day2 = (dt.datetime(2027, 6, d, 16, 0, tzinfo=dt.timezone.utc) for d in (1, 2))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "DATA", Path(tmp)), \
                mock.patch.dict(os.environ, LEDGER_PASSPHRASE="x" * 12), mock.patch.object(feeds, "scores", lambda s: {}):
            with mock.patch.object(feeds, "games", lambda s, d: feeds.mlb_frame([dict(dates=before)])), \
                    mock.patch.object(feeds, "odds", lambda s, a, b: evs if s == "MLB" else []):
                run.auto(day1 - dt.timedelta(hours=1))
                self.assertIsNone(store.load("index"))  # 8 AM: too early
                run.auto(day1)
                run.morning(day1, force=True)
            bets = store.load("index")["bets"]
            self.assertEqual(len(bets), 6)  # three games, side + total, no All-Star game, no double bets
            with mock.patch.object(feeds, "games", lambda s, d: feeds.mlb_frame([dict(dates=after)])), \
                    mock.patch.object(feeds, "odds", lambda s, a, b: []):
                run.auto(day2)
            got = {b["id"]: b["status"] for b in store.load("index")["bets"]}
            self.assertEqual(got["e0:totals"], "lost")
            self.assertEqual({v for k, v in got.items() if k.startswith("e1")}, {"void"})
            self.assertEqual(got["e2:totals"], "void")
            self.assertEqual(store.load("state")["pending"], [])
            self.assertNotIn(b"Yankees", (Path(tmp) / "index.enc").read_bytes())


class Outages(unittest.TestCase):
    def test_odds_outage_keeps_the_day_open_and_doubleheaders_map_in_order(self):
        NY, BOS = (1, "New York Yankees"), (2, "Boston Red Sox")
        rows = [mlb_game(1, "2027-05-01", 23, NY, BOS, 4, 3, "Final"), mlb_game(2, "2027-06-01", 20, NY, BOS), mlb_game(3, "2027-06-01", 20, NY, BOS)]
        rows[2]["games"][0].update(gameNumber=2, gameDate="2027-06-01T20:10:00Z")
        q = (-1.5, 150, -180, 8.5, -105, -115, -125, 105)
        evs = [event(e, NY[1], BOS[1], t, dict(fanduel=q)) for e, t in (("g2", "2027-06-01T23:30:00Z"), ("g1", "2027-06-01T20:09:00Z"))]
        now = dt.datetime(2027, 6, 1, 16, 0, tzinfo=dt.timezone.utc)
        def down(s, a, b):
            if s == "MLB":
                raise RuntimeError("429")
            return []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "DATA", Path(tmp)), mock.patch.object(config, "ROOT", Path(tmp)), \
                mock.patch.dict(os.environ, LEDGER_PASSPHRASE="x" * 12), \
                mock.patch.object(feeds, "games", lambda s, d: feeds.mlb_frame([dict(dates=rows)])):
            with mock.patch.object(feeds, "odds", down), self.assertRaises(SystemExit):
                run.auto(now)
            self.assertIsNone(store.load("state")["last_run"])
            with mock.patch.object(feeds, "odds", lambda s, a, b: evs if s == "MLB" else []):
                run.auto(now)
            got = {g["id"]: g["row"] for g in store.load("days/2027-06-01")["games"]}
            self.assertEqual(got, {"g1": "2_2027-06-01", "g2": "3_2027-06-01"})

    def test_unchanged_prices_teach_nothing(self):
        s = learn.fresh()
        for _ in range(200):
            learn.learn_close(s, "NBA", "spreads", 3.0, 3.5, 3.0, 3.5)
        self.assertEqual(learn.trust(s, "NBA", "spreads"), learn.trust(learn.fresh(), "NBA", "spreads"))


if __name__ == "__main__":
    unittest.main()
