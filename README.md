# stock-screener-ml

A tested Python stock screener that ranks stocks using technical indicators, cross-sectional factor scoring, a simple backtest engine, and an initial Bayesian factor-weighting layer.

The goal of this project is not to claim a profitable trading strategy. The goal is to build a clean, testable research framework while avoiding common mistakes such as lookahead bias, same-close execution assumptions, and in-sample backtest claims.

## Features

* Pure pandas/numpy technical indicators
* Cross-sectional winsorized z-score factors
* Weighted composite scoring
* Long-only top-N monthly rebalance backtest
* Bayesian/ridge-style factor-weight estimation
* Offline synthetic tests
* Explicit protections against misleading Bayesian backtests before walk-forward training exists

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
* Bayesian backtest disabled until walk-forward training is implemented

Not yet implemented:

* Walk-forward Bayesian training
* Transaction costs and turnover
* Fundamentals
* S&P 500 universe loading
* Local parquet caching

## Why Bayesian Scoring?

The baseline screener uses hard-coded factor weights. The Bayesian layer estimates factor weights from historical factor values and next-period returns using a simple ridge/Bayesian linear regression approach.

This lets weak or noisy factors shrink toward zero instead of being assigned arbitrary fixed weights. The current Bayesian module is for research and scoring only. It is not used for performance claims in backtests until walk-forward training is added.

## Avoiding Lookahead Bias

This project includes tests and design choices to reduce lookahead risk:

* Indicators are built from trailing price data.
* Ranking at an as-of date ignores future rows.
* Signals formed at date `t` enter on the next available close.
* `backtest --bayes` is disabled until walk-forward fitting is implemented.

## Running Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
py -3.14 tests\test_screener.py
```

Expected result:

```text
18/18 passed
```

## Roadmap

### Step 1: Bayesian scoring layer

Status: Mostly complete.

* Add Bayesian/ridge-style factor weighting
* Add synthetic tests for sign recovery
* Disable in-sample Bayesian backtesting until walk-forward training exists

### Step 2: Backtest realism

Planned:

* Add transaction costs
* Report turnover
* Add walk-forward factor fitting
* Evaluate out of sample only

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
