import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SHARPLINE_DATA", ROOT / "data"))
CACHE = Path(os.environ.get("SHARPLINE_CACHE", ROOT / ".cache"))
PARAMS = ROOT / "params"
TZ = ZoneInfo("America/Los_Angeles")

BANKROLL = 10_000.0
BETS_PER_DAY = 10
RUN_AFTER = "08:45"  # schedulers run late, so the first trigger is armed just before 9:00
RECHECK_AFTER = "15:00"  # second look at the lines before the evening games

SPORTS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "MLB": "baseball_mlb",
}

ACTIVE = ("NFL", "NBA", "MLB")  # bet these; the rest of SPORTS can still settle

BOOK = "fanduel"
# Ten books cost the same API credits as one. Weights follow how well each
# book's 9 AM price predicted the closing line (MODEL.md section 3).
BOOK_WEIGHTS = {
    "pinnacle": 1.0, "novig": 1.0, "prophetx": 1.0, "kalshi": 1.0,
    "lowvig": 0.4, "betonlineag": 0.4, "draftkings": 0.5, "betmgm": 0.4,
    "betrivers": 0.25,
}
MIN_BOOKS = 2

# Longshots and heavy chalk are where de-vigging is least reliable.
PRICE_RANGE = (-260, 150)
MAX_EDGE = 0.08          # a bigger apparent edge is usually a stale or bad quote
KELLY_AT_100 = 0.06      # Kelly fraction that earns confidence 100
VOID_AFTER_DAYS = 3
