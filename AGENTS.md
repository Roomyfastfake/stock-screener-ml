# backtester

Standalone backtesting engine, spun off from the sibling project
`stock-screener-ml` (GitHub: Roomyfastfake/stock-screener-ml, local clone at
`C:\Users\TANUSH\.claude\projects\stock-screener-ml`).

## Goal
A reusable, tested backtesting library that the screener (and other strategies)
can plug into, going beyond the screener's built-in monthly top-N backtest.

## Starting point (from stock-screener-ml, branch feat/costs-turnover)
- `screener/backtest.py`: long-only top-N, monthly rebalance, signal at close of
  `t`, entry at next close, equal weight. Costs charged per side on traded
  notional (drifted weights to new targets); reports one-way turnover and cost drag.
- Interface: `signal_fn(prices_window_up_to_t) -> pd.Series of scores`.
- Tests poison future rows to prove no lookahead. Keep that discipline here.

## Conventions
- Python, pandas/numpy only for core; no pytest dependency (tiny runner) unless
  decided otherwise.
- Every timing rule gets a hand-checked test.
- No performance claims from in-sample fits.

## Status
Empty. Scope and design not yet decided.
