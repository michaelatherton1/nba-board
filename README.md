# NBA Daily Board

A dashboard that rebuilds itself twice a day on GitHub Actions and publishes to GitHub Pages:

- **Yesterday:** final scores, totals, pace and team box-score rates (eFG%, TOV%, 3PA rate, FT rate), graded against the model's pre-game projection and the market line captured before tip.
- **Today and tomorrow:** projected game totals from an opponent-adjusted pace × efficiency model, next to the market's main line, with the model's lean.
- **Best available lines:** for every moneyline, spread and total, the cheapest price on each side across **Kalshi, Polymarket US and Novig**, with each venue's taker fee included.
- **Arbitrage:** any pair of opposite sides on the same proposition, on different venues, that costs less than $1 after fees, with order-book size where available.
- **Track record:** the model's average miss vs the market's, and its lean record, accumulating from launch.

No API keys or paid services are needed: every source is public.

## Schedule

`.github/workflows/daily.yml` runs at **9:23 AM CT** (results, projections, morning lines) and **4:47 PM CT** (fresh lines before tip-off), both an hour later during daylight time. Run it any time from **Actions → Daily NBA board → Run workflow**. To change the times, edit the two `cron` lines (they're in UTC).

Each run commits a small snapshot to `data/history/` (the pre-game projection and market line for each game). That's what lets the next day's run grade the model honestly, so don't delete that folder.

## How it works

| Step | Source | Code |
|---|---|---|
| Box scores and schedules, 2014-15 to today | ESPN via [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) releases | `nbadash/model.py` |
| Last night's games if the release hasn't caught up | ESPN public site API | `nbadash/espn.py` |
| Team ratings and projections | ratings updated after every game; ridge correction for rest, back-to-backs, altitude and recent form | `nbadash/model.py` |
| Injury report | official NBA injury report PDF (latest before each run; ESPN injuries as fallback), names matched to ESPN ids; P(plays) by status and by games missed in `status_probs.json` | `nbadash/injuries.py`, `nbadash/injuries_live.py` |
| Kalshi prices | `KXNBAGAME`, `KXNBASPREAD`, `KXNBATOTAL` series, public order books | `nbadash/venues.py` |
| Polymarket US prices | `gateway.polymarket.us` NBA league events | `nbadash/venues.py` |
| Novig prices | `api.novig.com` public catalog and books | `nbadash/venues.py` |
| Best lines and arbitrage | every price mapped to one of three propositions per game: home wins, home margin > X, total > L | `nbadash/lines.py` |
| Page | static HTML reading `site/data/latest.json` | `site/index.html` |

Fees modelled (taker, per $1 contract at price p): Kalshi `0.07 × p × (1−p) × series multiplier`; Polymarket US `feeCoefficient × p × (1−p)` (about 0.0695); Novig `0.03 × p × (1−p)` but only once a game is live, so pre-game fills are free.

## Read before trading

- **No edge against the closing line.** Graded against true closing totals (2016-17 to 2025-26), the model's
  gap to the line carries almost no information (calibration slope about 0 with morning-of availability,
  +0.06 to +0.15 with the official injury report). The page's lean probabilities shrink toward the line accordingly.
- **Opening lines are the open question.** Against the first available total, 6+ point gaps hit 58% (2016-22) and
  66% (2022-25) but 50% in 2025-26. The board records the first line it sees and the last line before tip for every
  game, and the track record grades the 6+ gap flags at the first line plus how far the line moved toward the lean
  (closing-line value). Treat it as a small-stakes forward test.
- **Arbitrage windows are short.** Snapshots twice a day mostly miss the fast ones. Anything over 5¢ per pair is usually a rules mismatch or a stale quote. Read both rulebooks before trading, and note that whole-number lines are left out because venues settle pushes differently.
- Not financial advice.

## Run locally

```sh
pip install -r requirements.txt
python -m nbadash.run            # writes site/data/latest.json
python -m http.server -d site    # open http://localhost:8000
```

`python -m nbadash.run --no-venues` skips the market calls. Raw API samples from each run are saved to `debug/` (and kept for 7 days as a workflow artifact) so parsing can be checked against what the venues actually return.
