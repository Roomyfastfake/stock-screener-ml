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

Not yet implemented:

* Fundamentals
* S&P 500 universe loading
* Local parquet caching

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
```

## Running Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
py -3.14 tests\test_screener.py
```

Expected result:

```text
22/22 passed
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

Planned:

* Load S&P 500 tickers from a local CSV
* Cache downloaded prices locally
* Keep reruns offline where possible

## Disclaimer

This project is for education and research. It is not financial advice and should not be used as a live trading system without further validation.
