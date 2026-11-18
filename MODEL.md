# The model

## 1. Why the previous model lost

Its code was replayed over 14,414 bets from January 2022 to November 2026.
It claimed a 15.7% edge, won 47.7% and returned -2.0%.

- Its weight against the market drifted to 48%, while regression on the same games supported 0-12%.
- It staked the most on its largest disagreements with FanDuel, which were mostly news it could not see.
- Converting lines with one curve and back with another created edges that did not exist.
- It read one book and was never tested against historical odds.

## 2. Outcome curves

Each sport has one curve for margin and one for total: a skewed Student-t on whole scores with fitted key-number multipliers.
A quote is de-vigged proportionally and inverted to the curve's median, `mu`.
FanDuel is priced from `mu` on the same curve, so agreeing with FanDuel always shows a negative edge.

## 3. Sharp line

`market = fanduel + trust * (sharp - fanduel)`.
`sharp` is a weighted mean of nine books: Pinnacle, Novig, ProphetX and Kalshi at 1.0, DraftKings at 0.5, LowVig, BetOnline and BetMGM at 0.4, BetRivers at 0.25.
Wide quotes are down-weighted and outliers are dropped.
Trust starts at 0.85, or 0.45 for NBA totals where FanDuel tends to lead.

## 4. Signals

`fair = market + kappa * weights . signals`.
Weights are a ridge regression of the clipped miss against closing lines.
Kappa is 75% of the value that minimized out-of-sample log loss, capped at 1.3.

| Sport | Ratings | Signals |
|---|---|---|
| NCAAB | Opponent-adjusted offense, defense and tempo, raw and luck-adjusted | luck, rating gap, under-3-games discount, favorite size, back-to-back, schedule density, early season |
| NBA | Same, plus ratings fitted to recent market lines | luck, rating gap, market history, last cover, mismatch, playoffs |
| NFL | Ratings fitted to past market lines | market history, rest, favorite size, wind, total size, division |
| MLB | Run-margin and run-total ratings | near-zero weight until live results earn more |

Luck-adjusted ratings remove 60-65% of three-point and 30% of free-throw deviation from league average.
Signals switch off when ratings and market differ by 14 or more points, because that gap is news.
Starter strikeout, walk and home-run rates, bullpen quality and fatigue, and wOBA were tested on 17,000 MLB games and added nothing.

## 5. Card

Edge is expected profit per dollar, capped at 8%.
Confidence is `100 * kelly / 0.06`, and stake equals confidence, which is about one-sixth Kelly on $10,000.
Bets without an edge get confidence 1 and rank by likelier winner.
Each game offers one side and one total, priced between -260 and +150.

## 6. Learning

Signal weights are Bayesian regressions that start from the backtest and update on every analyzed game.
Trust updates from which side closes the morning gap.
Ratings are rebuilt from box scores every morning.

## 7. Backtest

| | Bets | Won | Return |
|---|---|---|---|
| Old model, 2022-26 | 14,414 | 47.7% | -2.0% |
| Sharpline, 2022-26 | 14,421 | 52.8% | +6.1% (±1.7) |
| Sharpline without MLB | 8,075 | 53.4% | +4.0% (±2.3) |
| Sharpline, 2016-26 | 24,925 | 53.3% | +6.4% (±1.2) |

Weights for each season come from earlier seasons only, but the signal list was chosen after seeing this history.
MLB uses openers posted at different times, NFL wind uses actual readings, and college lines end in April 2024.
A live return of 1-3% would be a good result.
A replay on recorded 9 AM prices from October 21 to November 16, 2026 went 104-97 and lost $266.
