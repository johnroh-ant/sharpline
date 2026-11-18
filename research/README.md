# Research

These scripts built `params/model.json` and `site/backtest.json`.
Each takes a working directory that holds the downloads under `dl/` and intermediates under `work/`.

Order: `build_hoops.py`, `build_mlb.py`, `build_snapshots.py`, `train.py`, `backtest.py`, `replay_old_model.py`, `export_site.py`, `simulate.py`.

| Data | Source |
|---|---|
| NBA and NCAAB schedules and box scores, NBA closing lines 2020-26 | sportsdataverse/hoopR-nba-data, hoopR-mbb-data |
| NBA open and close 2011-22 | flancast90/sportsbookreview-scraper |
| NCAAB closing lines 2010-24 | zachmayer/point_spreads |
| NFL results and lines | nflverse/nfldata |
| MLB six-book odds 2021-25 | tanhungle/mlb-reference-data |
| MLB results, starters, pitcher lines | ethanbell528-cmd/fda-project-1, chadwickbureau/retrosplits (Retrosheet) |
| Timestamped multi-book snapshots, fall 2026 | huggingface.co/datasets/abetco/odds-history-archive, vladk-III/NBA |
