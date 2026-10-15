# stock-screener-ml

A tested Python stock screener that ranks stocks using technical indicators, cross-sectional factor scoring, a simple backtest engine, and an initial Bayesian factor-weighting layer.

The goal of this project is not to claim a profitable trading strategy. The goal is to build a clean, testable research framework while avoiding common mistakes such as lookahead bias, same-close execution assumptions, and in-sample backtest claims.

## Features

* Pure pandas/numpy technical indicators
* Cross-sectional winsorized z-score factors
* Weighted composite scoring
* Long-only top-N monthly rebalance backtest with transaction costs and turnover reporting
* Bayesian/ridge-style factor-weight estimation, refit walk-forward in backtests
* Offline synthetic tests

## Project Structure

```text
screener/
  indicators.py   # technical indicators
  score.py        # winsorization, z-scores, composite scoring
  screen.py       # feature table construction and stock ranking
  backtest.py     # long-only backtest engine
  bayes.py        # Bayesian/ridge-style factor weighting
  data.py         # yfinance download module
  synthetic.py    # synthetic data for offline tests

tests/
  test_screener.py

main.py           # command-line interface
```

## Current Status

Implemented:

* Indicator calculation
* Composite factor scoring
* Bayesian factor-weight estimation
* Basic long-only backtest
* Next-close entry timing
* Tests for lookahead-sensitive behavior
* Walk-forward Bayesian backtest (weights refit each rebalance on past data only)
* Transaction costs (`--cost-bps`, default 10 bps per side) and one-way turnover reporting
* S&P 500 universe (`--universe sp500`, bundled `data/sp500.csv` snapshot) and local parquet price cache
* Optional `--engine backtester` route to the sibling [backtester](../backtester) package (long-short, weighting schemes, benchmark stats, HTML tearsheet)

* Point-in-time S&P 500 membership (`--universe sp500-pit`): only names that were index members on each signal date are eligible
* Factor research (`main.py analyze`): rank IC, IC decay, quintile spreads, factor correlation

Not yet implemented:

* Real-data run of the fundamental factors (code and tests done, see Step 3)
* Prices for delisted former members (see below)

## Results: Survivorship Bias Was Most of the Edge

S&P 500, 2015-01-02 to 2026-10-13, top 25 equal weight, monthly rebalance, 10 bps per side, builtin engine (monthly stats):

| Universe | Weights | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| today's members (`sp500`) | fixed default | 18.7% | 19.2% | 1.00 | -31% |
| today's members (`sp500`) | walk-forward Bayes | 26.6% | 34.9% | 0.85 | -45% |
| **point-in-time (`sp500-pit`)** | **fixed default** | **12.6%** | **18.2%** | **0.75** | **-29%** |
| **point-in-time (`sp500-pit`)** | **walk-forward Bayes** | **9.6%** | **28.9%** | **0.47** | **-44%** |
| SPY buy and hold | | 13.9% | 14.9% | 0.97 | -34% |

Backtesting on *today's* members silently selects the companies that grew enough to stay in the index. Using who was actually a member on each date cuts CAGR by 6 to 17 points, and **neither strategy beats SPY on a risk-adjusted basis**. The Bayes book was the most inflated: it had learned to favour high-volatility names, and volatile survivors are exactly the big winners.

The factor research shows why (`main.py analyze`, 21-day forward return entered at the next close, non-overlapping monthly dates):

| Factor | Rank IC (today's members) | t | Rank IC (point-in-time) | t | Q5-Q1 / month (PIT) |
|---|---|---|---|---|---|
| momentum | -0.006 | -0.4 | -0.005 | -0.3 | -0.01% |
| trend | -0.028 | -1.8 | -0.025 | -1.5 | -0.57% |
| low_vol | -0.028 | -1.5 | -0.007 | -0.3 | -0.56% |
| reversal | 0.021 | 1.5 | 0.016 | 1.1 | +0.42% |

No price factor has a significant IC in large caps over this period. The low-vol quintile spread (high-vol names winning) was -1.35%/month with t = -2.5 on today's members and shrinks to -0.56% (t = -1.0) point-in-time. Trend and reversal are -0.5 correlated, so they partly double-count the same bet.

**Remaining bias.** Yahoo has no prices for 130 delisted former members (acquired, e.g. CELG, ATVI, XLNX; failed, e.g. SIVB, FRC), so 77% of 2015 members are priced, rising to 99% by 2026. Acquisitions usually exit at a premium and failures at a loss, so the direction of this residual bias is unclear. Membership comes from [fja05680/sp500](https://github.com/fja05680/sp500) (MIT) for former members and Wikipedia's per-company "Date added" for current ones, with ticker renames (FB to META, ANTM to ELV, 26 in all) detected and merged.

The `backtester` engine (daily marking, NAV costs) on today's members gave 18.5% CAGR, Sharpe 0.90, beta 0.95, alpha 5.1%; that alpha is the same survivorship effect.

## Why Bayesian Scoring?

The baseline screener uses hard-coded factor weights. The Bayesian layer estimates factor weights from historical factor values and next-period returns using a simple ridge/Bayesian linear regression approach.

This lets weak or noisy factors shrink toward zero instead of being assigned arbitrary fixed weights. The current Bayesian module is for research and scoring only. In backtests it is only used walk-forward: weights are refit at every rebalance on data up to that date.

## Avoiding Lookahead Bias

This project includes tests and design choices to reduce lookahead risk:

* Indicators are built from trailing price data.
* Ranking at an as-of date ignores future rows.
* Signals formed at date `t` enter on the next available close.
* `backtest --bayes` refits weights walk-forward; tests poison future rows and check the signal does not move.

## Usage

```powershell
py -3.14 main.py demo                                   # offline synthetic demo
py -3.14 main.py backtest --synthetic --bayes --cost-bps 10
py -3.14 main.py backtest --tickers AAPL MSFT NVDA --start 2018-01-01

# S&P 500 with the local cache (data/cache, git-ignored)
py -3.14 main.py fetch --universe sp500 --start 2015-01-01          # ~40s first time, ~3s after
py -3.14 main.py backtest --universe sp500 --start 2015-01-01 --offline --top-n 25
py -3.14 main.py backtest --universe sp500 --sector Energy --offline  # GICS sector filter

# Point-in-time membership (the honest version) and factor research
py -3.14 main.py fetch --universe sp500-pit --start 2015-01-01
py -3.14 main.py backtest --universe sp500-pit --start 2015-01-01 --offline --top-n 25
py -3.14 main.py analyze --universe sp500-pit --start 2015-01-01 --offline --out reports\research

# Through the backtester package, with benchmark stats and a tearsheet
py -3.14 -m pip install -e ..\backtester
py -3.14 main.py backtest --universe sp500 --start 2015-01-01 --top-n 25 `
    --engine backtester --cost-model nav --benchmark SPY --tearsheet reports\sp500.html

py -3.14 main.py refresh-universe                                     # rebuild data/sp500.csv
```

Cache rules: a ticker is re-downloaded (full history, never appended, because adjusted closes are restated after splits and dividends) when it is missing, was fetched from a later start, or is older than `--max-age-days` for open-ended requests. Today's bar is dropped until 16:30 New York time so an intraday price is never treated as a close. `--offline` never touches the network.

## Running Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
py -3.14 tests\test_screener.py
```

Expected result:

```text
60/60 passed
```

## Roadmap

### Step 1: Bayesian scoring layer

Status: Complete.

* Bayesian/ridge-style factor weighting
* Synthetic tests for sign recovery and shrinkage

### Step 2: Backtest realism

Status: Complete.

* Transaction costs charged on traded notional (drifted weights to new targets)
* One-way turnover and annualized cost drag in backtest stats
* Walk-forward factor fitting, out-of-sample evaluation only

### Step 3: Fundamentals

Status: Built and tested offline; first real run pending (needs `SEC_USER_AGENT`).

* Point-in-time data from SEC EDGAR XBRL `companyfacts`. Each value is used only from the first trading day after its `filed` date, so restatements never rewrite the past. `yfinance.Ticker.info` was rejected because it is today's snapshot (lookahead).
* Factors: `roe` (TTM net income / equity, NaN if equity <= 0) and `rev_growth` (TTM revenue vs one year earlier). Both are ratios of reported figures, so stock splits cannot distort them.
* Periods identified by start/end dates, Q4 derived as FY minus 9M, tag fallbacks merged per period (revenue tags changed with ASC 606), values older than 15 months dropped.
* Opt-in: `--fundamentals`. Default runs and the README numbers are unchanged.
* Validation gate: `main.py check-fundamentals --universe sp500` checks at every 10-K that the quarters add up to the full year.
* Deferred: P/E. It mixes as-reported EPS with split-adjusted prices and would jump at every split; it needs split data and a continuity test first.

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@example.com"   # SEC requires a contact
py -3.14 main.py check-fundamentals --universe sp500         # fetch + validate (~1 min at SEC's rate limit)
py -3.14 main.py backtest --universe sp500 --start 2015-01-01 --offline --fundamentals --top-n 25
```

### Step 4: Universe and caching

Status: Complete.

* S&P 500 tickers from a local CSV snapshot, with sector filter and Wikipedia refresh
* Per-ticker parquet cache with manifest, staleness rules and one retry for transient failures
* Reruns fully offline with `--offline`

Next: point-in-time membership to remove survivorship bias.

## Performance

A 12-year, 503-name, 135-rebalance backtest (builtin engine):

| Signal | Before | After |
|---|---|---|
| fixed weights | 226 s | 15 s |
| walk-forward Bayes | 389 s | 31 s |
| fixed weights via `--engine backtester` | 194 s | 13 s |

Three changes, all exact (results unchanged; max difference vs the old path 6e-14 on real data):

* The cache returns prices as one contiguous block. Concatenating 503 per-ticker series had produced 503 memory blocks, so every pandas op looped column by column.
* Fixed-weight signals compute features on the trailing `WARMUP_ROWS` (127) rows only. Every factor is a fixed-length trailing window and scoring is per row, so the last row is identical.
* Walk-forward Bayes keeps running sufficient statistics (`X'X, X'y, y'y, n`) and only absorbs dates whose forward return newly realized. It resets if a window is not a strict extension of the previous one, so a reused signal can never train on rows beyond the current window.

Tests compare both fast paths against the old full-window implementation on data with late listings, delistings and missing days.

## Disclaimer

This project is for education and research. It is not financial advice and should not be used as a live trading system without further validation.
