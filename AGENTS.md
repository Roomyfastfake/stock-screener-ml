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
- Step 4 done on branch `feat/universe-cache` (2026-10-15): `screener/universe.py`
  (data/sp500.csv snapshot, 503 names), `screener/cache.py` (parquet per ticker +
  manifest.json in git-ignored data/cache; full re-download on miss, never splice;
  drops today's bar before 16:30 ET). First real run documented in README.
- Speed fix on branch `perf/fast-signals` (2026-10-15): cache returns a single-block
  frame; fixed signal uses the trailing `screen.WARMUP_ROWS` rows; walk-forward Bayes
  is `screen._WalkForwardBayes` with running `bayes.DesignStats`, reset unless the
  window strictly extends the last one. S&P run: 226s -> 15s fixed, 389s -> 31s Bayes.
  If an indicator lookback changes, `WARMUP_ROWS` is derived from the constants in
  screen.py; the equivalence tests fail if it is too short.
- Fundamentals on branch `feat/fundamentals` (2026-10-15): `screener/fundamentals.py`,
  SEC companyfacts, point-in-time by `filed` date (usable next trading day). Factors
  `roe`, `rev_growth`; opt-in `--fundamentals`. `events()` is the fast one-pass path;
  `snapshot()` is the pandas reference, pinned by a test. CIKs live in data/sp500.csv.
- SEC needs `SEC_USER_AGENT` (name + email). Never hardcode or commit it, and never
  use the user's address without asking. A 403 means it is missing.
- P/E deliberately deferred: as-reported EPS vs split-adjusted prices breaks at splits.
- Before reporting any fundamentals backtest, run `check-fundamentals` and report the
  pass rate (TTM from quarters must equal 10-K FY).
- Branch `feat/ci-research` (2026-10-15): GitHub Actions CI; `screener/research.py`
  (rank IC, decay, quintiles, factor correlation; next-close timing); point-in-time
  membership `--universe sp500-pit` (data/sp500_history.csv from fja05680/sp500, MIT,
  plus Wikipedia date_added; renames detected by `universe._renamed_predecessors`);
  `screen.mask_to_members`; cache remembers dead symbols for 30 days; builtin stats
  gain true `cagr`.
- Headline (README "Results"): on point-in-time membership, fixed weights 12.6% CAGR
  Sharpe 0.75, Bayes 9.6% / 0.47, SPY 13.9% / 0.97. No price factor has significant IC.
  Report PIT numbers as the primary result; `sp500` (today's members) is biased.
- `main.py report` (HTML research page, screener/report.py), `main.py sweep` (24 variants,
  common start, deflated Sharpe on active return vs SPY: best IR 0.19, DSR 0.05 = luck),
  `screen.make_panel_signal_fn` (precomputed composite, exact, used by sweeps), factor
  registry `screen.FACTORS` with `--factors` (adds mom_12_1, high_52w), ruff lint in CI.
- Any new variant added to a sweep raises the deflation bar; report DSR with the trial
  count, never the best raw Sharpe alone.
- Known limits: 130 delisted former members have no Yahoo prices (77% of 2015 members
  priced, 99% by 2026).
- Branch stack: feat/costs-turnover -> feat/universe-cache -> perf/fast-signals
  -> feat/fundamentals -> feat/ci-research.
- Next: real fundamentals run once SEC_USER_AGENT is set; P/E with split handling.
- Sibling repo `backtester` (C:\Users\TANUSH\projects\backtester, a sibling folder) is a
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
