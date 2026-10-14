"""Offline test-suite on hand-built and synthetic data, no network.

    py -3 tests/test_backtester.py

Plain asserts + a tiny runner so there is no pytest dependency.
"""
from __future__ import annotations

import math
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import backtester as bt
from backtester import research as rs
from backtester import synthetic, weights


def _const(best: str, cols=("A", "B")):
    return lambda window: pd.Series({c: float(c == best) for c in window.columns})


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def test_monthly_dates_are_last_trading_day():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"])
    assert bt.rebalance_dates(idx, "M") == list(pd.to_datetime(
        ["2020-01-31", "2020-02-28", "2020-03-02"]))


def test_weekly_dates_across_year_end():
    # Mon 2018-12-31 and Wed 2019-01-02 are the same Mon-Sun week.
    idx = pd.to_datetime(["2018-12-27", "2018-12-28", "2018-12-31", "2019-01-02", "2019-01-07"])
    assert bt.rebalance_dates(idx, "W") == list(pd.to_datetime(
        ["2018-12-28", "2019-01-02", "2019-01-07"]))


def test_daily_and_quarterly_dates():
    idx = pd.bdate_range("2020-03-30", "2020-04-02")
    assert bt.rebalance_dates(idx, "D") == list(idx)
    assert bt.rebalance_dates(idx, "Q") == list(pd.to_datetime(["2020-03-31", "2020-04-02"]))


# --------------------------------------------------------------------------- #
# timing: parity with stock-screener-ml fixtures
# --------------------------------------------------------------------------- #
def test_enters_next_close_after_signal():
    """Screener fixture: signal at Jan 31 close, fill Feb 3, A goes 10 -> 20."""
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"])
    prices = pd.DataFrame({"A": [100.0, 100.0, 10.0, 999.0, 20.0],
                           "B": [100.0, 100.0, 10.0, 999.0, 5.0]}, index=idx)
    signal_ends = []

    def signal_fn(window):
        signal_ends.append(window.index[-1])
        return pd.Series({"A": 1.0, "B": 0.0})

    res = bt.backtest(prices, signal_fn, top_n=1, min_history=1)
    # Feb 28 signal would enter Mar 2 = last row, so it is skipped (no holding period).
    assert signal_ends == [pd.Timestamp("2020-01-31")]
    assert list(res.holdings) == [pd.Timestamp("2020-02-03")]
    assert list(res.period_returns.index) == [pd.Timestamp("2020-03-02")]
    assert np.isclose(res.period_returns.iloc[0], 1.0)            # 20 / 10 - 1
    # Daily marks: entry day flat, then 999/10 - 1, then 20/999 - 1.
    assert np.allclose(res.returns, [0.0, 98.9, 20 / 999 - 1])


def test_costs_and_turnover_hand_checked():
    """Screener fixture: hold A (+10%), then rotate fully to B (+20%)."""
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03",
                          "2020-02-28", "2020-03-02", "2020-03-31", "2020-04-01"])
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0],
                           "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 120.0]}, index=idx)

    def run(**kw):
        picks = iter(["A", "B"])
        return bt.backtest(prices, lambda w: _const(next(picks))(w),
                           top_n=1, min_history=1, **kw)

    free, paid = run(), run(cost_bps=30, slippage_bps=20)        # 50 bps all-in
    assert np.allclose(free.period_returns, [0.10, 0.20])         # same as screener
    assert np.allclose(paid.period_gross, [0.10, 0.20])
    assert np.allclose(paid.turnover, [0.5, 1.0])                 # buy book, full rotation
    assert np.allclose(paid.costs, [0.005, 0.010])                # traded 1.0 then 2.0
    # Costs come off NAV at entry, then the book earns on what is left.
    assert np.allclose(paid.period_returns, [0.995 * 1.10 - 1, 0.99 * 1.20 - 1])
    assert np.isclose(paid.equity.iloc[-1], 0.995 * 1.10 * 0.99 * 1.20)


def test_additive_cost_model_matches_screener_exactly():
    """Same fixture, screener accounting: net = gross - traded * rate, per period."""
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03",
                          "2020-02-28", "2020-03-02", "2020-03-31", "2020-04-01"])
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0],
                           "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 120.0]}, index=idx)
    picks = iter(["A", "B"])
    paid = bt.backtest(prices, lambda w: _const(next(picks))(w), top_n=1, min_history=1,
                       cost_bps=50, cost_model="additive")
    assert np.allclose(paid.turnover, [0.5, 1.0])
    assert np.allclose(paid.period_returns, [0.10 - 0.005, 0.20 - 0.010], rtol=0, atol=1e-15)
    assert np.allclose(paid.period_gross, [0.10, 0.20])


def _load_screener_backtest():
    """Import stock-screener-ml's backtest module from disk, or None if absent."""
    import importlib.util
    import os
    here = pathlib.Path(__file__).resolve()
    candidates = [os.environ.get("SCREENER_PATH", "")]
    candidates += [str(p / "stock-screener-ml") for p in here.parents]
    for c in candidates:
        f = pathlib.Path(c) / "screener" / "backtest.py" if c else None
        if f and f.exists():
            spec = importlib.util.spec_from_file_location("_screener_bt", f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod                    # dataclasses need it registered
            spec.loader.exec_module(mod)
            if "cost_bps" in mod.backtest.__code__.co_varnames:
                return mod
    return None


def test_parity_with_live_screener_code():
    """If stock-screener-ml (with costs) is on disk, match it on synthetic data."""
    sbt = _load_screener_backtest()
    if sbt is None:
        print("      (skip: stock-screener-ml backtest with cost_bps not found)")
        return
    prices = synthetic.make_prices(n_tickers=25, n_days=900, seed=5)
    sfn = synthetic.momentum_signal()
    ref = sbt.backtest(prices, sfn, top_n=5, cost_bps=25)
    new = bt.backtest(prices, sfn, top_n=5, cost_bps=25, cost_model="additive")
    # The screener drops the tail after its last full period; we hold to the end.
    common = ref.period_returns.index
    assert len(common) > 10
    assert np.allclose(new.period_returns.loc[common], ref.period_returns, rtol=0, atol=1e-12)
    assert np.allclose(new.period_gross.loc[common], ref.gross_returns, rtol=0, atol=1e-12)
    assert np.allclose(new.turnover.iloc[:len(common)], ref.turnover, rtol=0, atol=1e-12)
    assert list(new.holdings.values())[:len(common)] == list(ref.holdings.values())


def test_weekly_rebalance_enters_next_day():
    idx = pd.bdate_range("2021-01-04", periods=12)               # Mon .. Tue+1wk
    prices = pd.DataFrame({"A": np.arange(12.0) + 100, "B": 100.0}, index=idx)
    seen = []

    def signal_fn(window):
        seen.append(window.index[-1])
        return pd.Series({"A": 1.0, "B": 0.0})

    res = bt.backtest(prices, signal_fn, top_n=1, freq="W", min_history=1)
    fridays = [pd.Timestamp("2021-01-08"), pd.Timestamp("2021-01-15")]
    assert seen == fridays
    assert list(res.weights) == [pd.Timestamp("2021-01-11"), pd.Timestamp("2021-01-18")]
    assert res.returns.index[0] == pd.Timestamp("2021-01-11")


# --------------------------------------------------------------------------- #
# daily drift, long-short, borrow, missing data
# --------------------------------------------------------------------------- #
def test_long_only_drift_between_rebalances():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05"])
    prices = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.1, 1.1],
                           "B": [1.0, 1.0, 1.0, 1.0, 1.1]}, index=idx)
    res = bt.backtest(prices, lambda w: pd.Series({"A": 1.0, "B": 0.5}),
                      top_n=2, min_history=1)
    # Day 1: 0.5*10% = 5%. Weights drift to A=0.55/1.05, B=0.5/1.05.
    # Day 2: B +10% -> 0.05/1.05. Equity 0.55 + 0.55 = 1.10.
    assert np.allclose(res.returns, [0.0, 0.05, 0.05 / 1.05])
    assert np.isclose(res.equity.iloc[-1], 1.10)


def test_long_short_hand_checked():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05"])
    prices = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.1, 1.1],
                           "B": [1.0, 1.0, 1.0, 1.2, 1.08]}, index=idx)
    res = bt.backtest(prices, lambda w: pd.Series({"A": 1.0, "B": -1.0}),
                      top_n=1, short_n=1, min_history=1)
    w = res.weights[pd.Timestamp("2020-02-03")]
    assert w["A"] == 1.0 and w["B"] == -1.0                       # gross 2, net 0
    # NAV = 1 cash + long A - short liability B: 1 + 1.1 - 1.2 = 0.9, then 1 + 1.1 - 1.08.
    assert np.isclose(res.equity.iloc[-2], 0.9)
    assert np.isclose(res.equity.iloc[-1], 1.02)
    assert np.isclose(res.turnover.iloc[0], 1.0)                  # traded 2, one-way 1


def test_borrow_fee_on_short_gross():
    idx = pd.bdate_range("2020-01-30", periods=5)
    prices = pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx)
    res = bt.backtest(prices, lambda w: pd.Series({"A": 1.0, "B": -1.0}),
                      top_n=1, short_n=1, min_history=1, borrow_bps=252)
    # Flat prices; short gross 1 at 252 bps/yr = 1 bp per day for the 3 days held.
    # Day 1 exactly 1 bp; after that short gross is 1/(1 - 1e-4) of a smaller NAV.
    assert np.isclose(res.returns.iloc[1], -1e-4, rtol=0, atol=1e-15)
    assert np.isclose(res.returns.iloc[2], -1e-4 / (1 - 1e-4), rtol=0, atol=1e-15)


def test_missing_price_is_zero_return_and_ineligible():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05"])
    prices = pd.DataFrame({"A": [1.0, 1.0, 1.0, np.nan, 1.2],
                           "C": [np.nan, np.nan, 5.0, 5.0, 5.0]}, index=idx)
    res = bt.backtest(prices, lambda w: pd.Series({"A": 1.0, "C": 9.0}),
                      top_n=2, min_history=1)
    # C has no price at the Jan 31 signal date, so only A is bought.
    assert list(res.weights[pd.Timestamp("2020-02-03")].index) == ["A"]
    # NaN close on Feb 4 = carry last price (0%), full move lands on Feb 5.
    assert np.allclose(res.returns, [0.0, 0.0, 0.2])


# --------------------------------------------------------------------------- #
# weighting
# --------------------------------------------------------------------------- #
def test_score_weighting_positive_part():
    s = pd.Series({"A": 3.0, "B": 1.0, "C": -2.0})
    assert np.allclose(weights.score(s, None), [0.75, 0.25, 0.0])
    assert np.allclose(weights.score(-s.abs(), None), [1 / 3] * 3)  # none positive -> equal


def test_short_leg_uses_negated_scores():
    idx = pd.bdate_range("2020-01-30", periods=4)
    prices = pd.DataFrame(1.0, index=idx, columns=list("ABCD"))
    res = bt.backtest(prices, lambda w: pd.Series({"A": 4.0, "B": 2.0, "C": -1.0, "D": -3.0}),
                      top_n=2, short_n=2, weighting="score", min_history=1)
    w = res.weights[idx[2]]                                       # Jan 31 signal, Feb 3 entry
    assert np.allclose(w[["A", "B"]], [2 / 3, 1 / 3])
    assert np.allclose(w[["D", "C"]], [-0.75, -0.25])            # more negative = bigger short


def test_inverse_vol_hand_checked():
    idx = pd.bdate_range("2020-01-01", periods=5)
    # A alternates +/-2%, B +/-1%: identical shape, A has twice B's stdev.
    a = [100.0, 102.0, 99.96, 101.9592, 99.920016]
    b = [100.0, 101.0, 99.99, 100.9899, 99.979999]
    window = pd.DataFrame({"A": a, "B": b}, index=idx)
    w = weights.make_inverse_vol(lookback=4)(pd.Series({"A": 1.0, "B": 1.0}), window)
    assert np.allclose(w, [1 / 3, 2 / 3])


# --------------------------------------------------------------------------- #
# lookahead discipline
# --------------------------------------------------------------------------- #
def test_engine_no_lookahead_with_poisoned_future():
    prices = synthetic.make_prices(n_tickers=15, n_days=600, seed=4)
    t = prices.index[400]
    poisoned = prices.copy()
    poisoned.loc[poisoned.index > t] *= np.linspace(0.2, 9.0, prices.shape[1])
    kw = dict(top_n=4, short_n=3, weighting="inverse_vol", freq="W", cost_bps=10)
    sfn = synthetic.momentum_signal()
    a = bt.backtest(prices, sfn, **kw)
    b = bt.backtest(poisoned, sfn, **kw)
    pd.testing.assert_series_equal(a.returns.loc[:t], b.returns.loc[:t])
    for d in [d for d in a.weights if d <= t]:
        pd.testing.assert_series_equal(a.weights[d], b.weights[d])
    assert not np.allclose(a.returns.loc[t:].iloc[1:], b.returns.loc[t:].iloc[1:])


def test_signal_window_ends_at_signal_date():
    prices = synthetic.make_prices(n_tickers=6, n_days=300, seed=1)
    ends = []

    def sfn(window):
        ends.append(window.index[-1])
        return window.iloc[-1]

    res = bt.backtest(prices, sfn, top_n=2, freq="M", min_history=50)
    assert ends == res.signal_dates
    for sig, entry in zip(res.signal_dates, res.weights):
        assert prices.index.get_loc(entry) == prices.index.get_loc(sig) + 1


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def test_perf_stats_hand_checked():
    r = pd.Series([0.10, -0.20, 0.10])
    s = bt.perf_stats(r, periods_per_year=1)
    assert np.isclose(s["total_return"], 1.1 * 0.8 * 1.1 - 1)
    assert np.isclose(s["max_drawdown"], -0.2)                    # 1.10 -> 0.88
    sd = np.std([0.1, -0.2, 0.1], ddof=1)
    assert np.isclose(s["sharpe"], 0.0)
    assert np.isclose(s["ann_vol"], sd)
    assert np.isclose(s["hit_rate"], 2 / 3)


def test_first_day_loss_counts_as_drawdown():
    s = bt.perf_stats(pd.Series([-0.1, 0.05, 0.01]))
    assert np.isclose(s["max_drawdown"], -0.1)


def test_benchmark_stats_identities():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.0005, 0.01, 500))
    same = bt.benchmark_stats(b, b)
    assert np.isclose(same["beta"], 1.0) and np.isclose(same["alpha_ann"], 0.0)
    assert np.isclose(same["tracking_error"], 0.0)
    levered = bt.benchmark_stats(2 * b + 0.0001, b)
    assert np.isclose(levered["beta"], 2.0)
    assert np.isclose(levered["alpha_ann"], 0.0001 * 252)


def test_backtest_benchmark_wiring():
    prices = synthetic.make_prices(n_tickers=10, n_days=400, seed=2)
    res = bt.backtest(prices, lambda w: pd.Series(1.0, index=w.columns),
                      top_n=10, freq="D", min_history=1, benchmark=prices.mean(axis=1))
    assert np.isfinite(res.stats["beta"]) and 0.5 < res.stats["beta"] < 1.5


def test_costs_monotone_on_synthetic():
    prices = synthetic.make_prices(n_tickers=20, n_days=800, seed=0)
    sfn = synthetic.momentum_signal()
    cagr = [bt.backtest(prices, sfn, top_n=5, cost_bps=c).stats["cagr"] for c in (0, 10, 50)]
    assert cagr[0] > cagr[1] > cagr[2]


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #
def test_walk_forward_splits_disjoint_and_ordered():
    idx = pd.bdate_range("2020-01-01", periods=100)
    for expanding in (True, False):
        splits = rs.walk_forward_splits(idx, train_size=40, test_size=25, expanding=expanding)
        assert [s.test_start for s in splits] == [idx[40], idx[65], idx[90]]
        assert splits[-1].test_end == idx[99]
        for s, nxt in zip(splits, splits[1:]):
            assert s.test_end < nxt.test_start
        for s in splits:
            assert s.train_end < s.test_start
            assert idx.get_loc(s.test_start) == idx.get_loc(s.train_end) + 1
        assert splits[1].train_start == (idx[0] if expanding else idx[25])


def test_walk_forward_fit_sees_train_only():
    prices = synthetic.make_prices(n_tickers=8, n_days=500, seed=3)
    splits = rs.walk_forward_splits(prices.index, train_size=200, test_size=100)
    fit_ends = []

    def fit_fn(train):
        fit_ends.append(train.index[-1])
        return synthetic.momentum_signal()

    wf = rs.walk_forward(prices, fit_fn, splits, top_n=3, min_history=1)
    assert fit_ends == [s.train_end for s in splits]
    for sp, res in wf.folds:
        assert all(d >= sp.test_start for d in res.signal_dates)
        assert res.returns.index[-1] <= sp.test_end
    assert wf.returns.index.is_unique and wf.returns.index.is_monotonic_increasing


def test_bootstrap_ci_reproducible_and_brackets():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 750))
    a = rs.bootstrap_sharpe_ci(r, n_boot=400, seed=7)
    b = rs.bootstrap_sharpe_ci(r, n_boot=400, seed=7)
    assert a == b
    assert a["lo"] < a["sharpe"] < a["hi"]


def test_probabilistic_sharpe_hand_checked():
    # mean 0.01, sd(ddof=1) = 0.02*sqrt(4/3), skew 0, raw kurt 1 -> denom 1.
    r = pd.Series([0.03, -0.01, 0.03, -0.01])
    sr = 0.01 / (0.02 * math.sqrt(4 / 3))
    assert np.isclose(rs.probabilistic_sharpe(r), 0.5 * (1 + math.erf(sr * math.sqrt(3) / math.sqrt(2))))
    assert np.isclose(rs.probabilistic_sharpe(r), 0.77337, atol=1e-4)   # Phi(0.75)


def test_expected_max_sharpe_hand_checked():
    # N=10, V=1: 0.42278 * Phi^-1(0.9)=1.28155 + 0.57722 * Phi^-1(0.96321)~1.7896
    assert np.isclose(rs.expected_max_sharpe(10, 1.0), 1.5748, atol=2e-3)
    assert rs.expected_max_sharpe(1, 1.0) == 0.0
    r = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 500))
    trials = [0.05, 0.1, 0.15, 0.2]
    assert rs.deflated_sharpe(r, trials) < rs.probabilistic_sharpe(r)


def test_tearsheet_writes_html():
    import tempfile
    from backtester import report
    prices = synthetic.make_prices(n_tickers=10, n_days=400, seed=6)
    res = bt.backtest(prices, synthetic.momentum_signal(), top_n=3,
                      benchmark=prices.mean(axis=1))
    with tempfile.TemporaryDirectory() as d:
        out = report.tearsheet(res, pathlib.Path(d) / "t.html", notes={"dsr": 0.42})
        text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "<svg" in text
    assert "sharpe" in text and "info_ratio" in text and "dsr" in text


# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
