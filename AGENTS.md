# stock-screener-ml

Python stock screener: technical factors, cross-sectional z-scores, composite
and walk-forward Bayesian scoring, long-only top-N monthly backtest.
GitHub: Roomyfastfake/stock-screener-ml. See README.md for usage and roadmap.

## Conventions
- Core deps numpy/pandas only; yfinance only in `screener/data.py` (lazy import).
- Tests: `py -3 tests/test_screener.py` with PYTHONPATH set to repo root. Tiny runner, no pytest.
- Every timing rule gets a hand-checked test; lookahead tests poison future rows.
- No performance claims from in-sample fits. Backtests use walk-forward weights only.
- Work on feature branches, merge via PR.

## Status (2026-10-14)
- Roadmap steps 1 (Bayesian scoring) and 2 (backtest realism: costs, turnover,
  walk-forward) done. Branch `feat/costs-turnover` adds costs; CLI default
  `--cost-bps 10`, library default 0.
- Next: step 4 (S&P 500 CSV universe + local parquet cache, first real-data run),
  then step 3 (fundamentals).
- Sibling repo `backtester` (C:\Users\TANUSH\.claude\projects\backtester) is for a
  standalone backtesting engine that should accept this repo's `signal_fn` interface.
