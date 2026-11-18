# Sharpline

A fake-money betting model for NFL, NBA, NCAA men's basketball and MLB.
It runs on GitHub Actions at about 9 AM Pacific, so no computer has to be on.

Each morning it settles yesterday's bets, learns from every game it analyzed, prices today's FanDuel lines, and bets the top ten at stake = confidence.
The dashboard shows the card, the reasoning, every game analyzed, progress and the backtest.

## Setup

```bash
sh setup.sh
```

The script creates the repo, stores your Odds API key and a new passphrase as secrets, and turns on the page.
The code is public and the ledger in `data/` is encrypted, so only the passphrase opens your bets.

## Day to day

| To do this | Run |
|---|---|
| Run now | `gh workflow run daily.yml -f force=true` |
| See the last runs | `gh run list -L 5` |
| Pause | `gh workflow disable daily.yml` |
| Test | `pip install -r requirements.txt && python -m unittest discover -s tests` |

GitHub emails you if a run fails.
A busy month uses about 400 of the 500 free Odds API credits, and the afternoon re-check skips itself when credits run short.

See [MODEL.md](MODEL.md) for the model and [research/](research/README.md) for training and backtests.
