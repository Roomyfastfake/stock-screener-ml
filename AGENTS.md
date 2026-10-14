# backtester

Standalone backtesting engine, spun off from the sibling project
`stock-screener-ml` (GitHub: Roomyfastfake/stock-screener-ml, local clone at
`C:\Users\TANUSH\.claude\projects\stock-screener-ml`).

## Role (integration plan agreed 2026-10-14, see screener AGENTS.md)
- **stock-screener-ml = alpha layer**: data, universe, factors, scoring, walk-forward
  Bayesian weights. Produces `signal_fn`.
- **backtester = evaluation layer**: rebalance schedule, portfolio construction,
  costs + slippage, metrics (benchmark-relative, bootstrap CIs, deflated Sharpe),
  HTML tearsheet. Imports nothing from the screener.
- Contract (the only coupling): `signal_fn(prices[dates x tickers] up to and
  including t) -> Series[ticker -> score]`, higher = better, NaN = ineligible.
  Signal after close t, fill at next close.
- Packaged via `pyproject.toml`; screener installs with `pip install -e ../backtester`.
- Next on the screener side: `--engine backtester` routes `backtest` through this
  package; keep `screener/backtest.py` as fallback until parity holds on real data.
  Later shared extras: factor attribution, parameter sweeps, Streamlit app, CI.

## Layout
- `backtester/engine.py`: `backtest(...) -> BacktestResult`.
- `backtester/schedule.py`: rebalance calendars D/W/M/Q.
- `backtester/weights.py`: equal, score, inverse_vol, or any `f(scores, window)`.
- `backtester/stats.py`: `perf_stats`, `benchmark_stats` (252 periods/yr).
- `backtester/research.py`: walk-forward splits/runner, stationary-bootstrap
  Sharpe CI, probabilistic and deflated Sharpe.
- `backtester/report.py`: `tearsheet(result, path)` self-contained HTML.
- `backtester/synthetic.py`: deterministic prices + a momentum `signal_fn`.
- Tests: `py -3 tests/test_backtester.py` (tiny runner, no pytest).
- Demo: `py -3 examples/demo.py out.html`.

## Decisions
- **Timing** (same as screener): signal dates = last trading day of each period;
  `signal_fn(prices.loc[:r])`; trade at next close `e`; hold until next entry.
  Final book is held to the last row (screener drops it). A rebalance whose entry
  would be the last row is skipped. If a signal is all-NaN, the old book keeps drifting.
- **Weekly** grouping uses `to_period("W")` (Mon-Sun), safe across year end.
- **Daily marking**: weights drift every day, `w' = w(1+r)/(1 + w.r - fee)`, so
  stats use daily returns at 252/yr for every rebalance frequency (screener
  annualized monthly periods with 12, which breaks for D/W and hides intra-month DD).
- **Gross convention**: long leg sums to +1; short leg (`short_n > 0`) to -1, picked
  from the bottom scores not already long. Long-short = gross 2, net 0, cash at 0%.
- **Score weighting**: positive part of the leg-oriented score (short leg uses
  `-score`); equal weight if none positive. **Inverse-vol**: 1/stdev of last 63
  daily returns from the signal window only.
- **Costs**: `(cost_bps + slippage_bps)` per unit traded on `sum|target - drifted|`
  at entry; one-way turnover `0.5 * traded`; `borrow_bps` annual on short gross,
  accrued daily /252.
  - `cost_model="nav"` (default): cost reduces NAV at entry; period net
    `(1 - c)(1 + g) - 1`.
  - `cost_model="additive"`: screener accounting, net `g - c`, book drifts cost-free.
    Used for parity; matches screener's code to 1e-12 on synthetic data.
- **Missing data**: prices forward-filled, so a NaN close = 0% return for a held
  name. A name is eligible at `r` only if it has a price at `r`.
- **Walk-forward**: `fit_fn(train_prices) -> signal_fn`; each test fold starts flat
  (full rebuild cost at fold start). Splits are non-overlapping, train strictly before test.
- **Bootstrap**: Politis-Romano stationary bootstrap (mean block 20), seeded.
- **Deflated Sharpe**: Bailey & Lopez de Prado 2014, per-period Sharpes, raw
  kurtosis in the PSR denominator, `statistics.NormalDist` (no scipy).

## Conventions
- Python, pandas/numpy only for core; no pytest dependency (tiny runner).
- Every timing rule gets a hand-checked test; lookahead tests poison future rows.
- Parity tests: screener's two hand-checked fixtures, plus a live comparison
  against `stock-screener-ml/screener/backtest.py` when it is on disk
  (`SCREENER_PATH` env var or a sibling folder; skipped otherwise).
- No performance claims from in-sample fits.

## Status
v0.1: engine, research metrics, tearsheet, 28 tests passing (Python 3.14,
pandas 3.0, numpy 2.4). Not pushed; no GitHub repo yet.
