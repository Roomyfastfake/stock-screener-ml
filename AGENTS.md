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
- Sibling repo `backtester` (moving to C:\Users\TANUSH\projects\backtester) is a
  standalone backtesting engine. See Integration plan.

## Integration plan with `backtester` (agreed 2026-10-14)
Split of responsibility:
- **stock-screener-ml = alpha layer**: data + cache, universe, factors, scoring,
  walk-forward Bayesian weights. Produces `signal_fn(prices_upto_t) -> pd.Series`.
- **backtester = evaluation layer**: rebalance schedule, portfolio construction
  (equal / score / inverse-vol, long-only and long-short), costs + slippage,
  metrics (benchmark-relative, bootstrap CIs, deflated Sharpe), tearsheet report.
  Depends on nothing from the screener.

Contract (the only coupling):
- `signal_fn(prices: DataFrame[dates x tickers], up to and including t) -> Series[ticker -> score]`,
  higher = better, NaN = ineligible. Timing: signal after close t, fill at next close.
- Backtester is a pip-installable package (`pyproject.toml`); the screener installs
  it with `pip install -e ../backtester` as an optional dependency.

Migration steps:
1. Backtester ships v0.1 with the contract and a parity test: on the screener's
   hand-checked cases it reproduces `screener/backtest.py` results exactly.
2. Screener adds `--engine backtester` that routes `backtest` through the package;
   keep `screener/backtest.py` as the no-dependency fallback until parity holds on real data.
3. Shared extras: factor attribution, parameter sweeps, one HTML tearsheet, then a
   Streamlit app (rank -> backtest -> tearsheet) and GitHub Actions CI in both repos.
