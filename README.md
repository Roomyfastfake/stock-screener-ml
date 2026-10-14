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

Not yet implemented:

* Fundamentals
* Point-in-time index membership (see survivorship bias below)

## First Real-Data Run (2026-10-14)

S&P 500, 2015-01-02 to 2026-10-13, fixed default weights, top 25 equal weight, monthly, 10 bps per side:

| Engine | CAGR | Sharpe | Max DD | Beta vs SPY | Alpha (ann.) | Info ratio |
|---|---|---|---|---|---|---|
| builtin (monthly stats) | 20.9% | 1.00 | -31% | n/a | n/a | n/a |
| backtester (daily stats, NAV costs) | 18.5% | 0.90 | -42% | 0.95 | 5.1% | 0.33 |

**Treat these as optimistic.** The universe is *today's* S&P 500 members, so every stock that was dropped from the index after doing badly is missing from the history (survivorship bias). The daily drawdown is deeper than the monthly one because monthly stats only see month-end marks.

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
33/33 passed
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

Planned:

* Add trailing P/E, ROE, revenue growth, and other basic fields
* Handle missing values explicitly
* Convert fundamentals into cross-sectional z-score factors

### Step 4: Universe and caching

Status: Complete.

* S&P 500 tickers from a local CSV snapshot, with sector filter and Wikipedia refresh
* Per-ticker parquet cache with manifest, staleness rules and one retry for transient failures
* Reruns fully offline with `--offline`

Next: point-in-time membership to remove survivorship bias; faster incremental feature computation (each rebalance currently rebuilds features on the full window, so a 12-year S&P 500 run takes several minutes).

## Disclaimer

This project is for education and research. It is not financial advice and should not be used as a live trading system without further validation.
