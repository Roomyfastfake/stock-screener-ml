"""Offline test-suite. Runs on deterministic synthetic data, no network.

    py -3.14 tests/test_screener.py          # (or: PYTHONPATH=. python ...)

Plain asserts + a tiny runner so there is no pytest dependency.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import main as cli
from screener import backtest as bt
from screener import bayes
from screener import indicators as ind
from screener import score as sc
from screener import screen
from screener import synthetic


# --------------------------------------------------------------------------- #
# indicators
# --------------------------------------------------------------------------- #
def test_momentum_sign_and_warmup():
    prices = synthetic.make_prices(n_tickers=6, n_days=300, seed=3)
    mom = ind.momentum(prices, lookback=126, skip=21)
    # Warm-up region is NaN (needs `lookback` history).
    assert mom.iloc[:126].isna().all().all()
    assert mom.iloc[150:].notna().all().all()
    # A monotonically rising series has positive momentum.
    rising = pd.Series(np.linspace(1, 2, 300), index=prices.index)
    assert ind.momentum(rising, 126, 21).iloc[-1] > 0


def test_rsi_bounds():
    prices = synthetic.make_prices(n_tickers=4, n_days=200, seed=5)
    r = ind.rsi(prices, 14)
    valid = r.dropna(how="all")
    assert ((valid >= 0) & (valid <= 100)).all().all()


def test_indicators_no_lookahead():
    """Truncation invariance: an indicator at t is unchanged by future rows."""
    prices = synthetic.make_prices(n_tickers=5, n_days=400, seed=7)
    t = prices.index[300]
    full = ind.momentum(prices).loc[t]
    truncated = ind.momentum(prices.loc[:t]).iloc[-1]
    pd.testing.assert_series_equal(full, truncated, check_names=False)


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def test_zscore_properties():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(50, 12)))
    z = sc.cross_sectional_zscore(frame)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1, ddof=0), 1.0, atol=1e-9)


def test_winsorize_clips_outliers():
    frame = pd.DataFrame([[0, 1, 2, 3, 1000.0]])
    w = sc.winsorize(frame, (0.2, 0.2))
    assert w.to_numpy().max() < 1000.0
    assert w.to_numpy().max() <= np.quantile([0, 1, 2, 3, 1000.0], 0.8) + 1e-9


def test_composite_is_weighted_sum():
    idx = pd.date_range("2020-01-01", periods=3)
    cols = ["A", "B"]
    f1 = pd.DataFrame([[1.0, -1.0]] * 3, index=idx, columns=cols)
    f2 = pd.DataFrame([[2.0, 0.0]] * 3, index=idx, columns=cols)
    comp = sc.composite_score({"f1": f1, "f2": f2}, {"f1": 1.0, "f2": 0.5})
    assert np.allclose(comp["A"], 1.0 * 1.0 + 0.5 * 2.0)
    assert np.allclose(comp["B"], 1.0 * -1.0 + 0.5 * 0.0)


# --------------------------------------------------------------------------- #
# screen
# --------------------------------------------------------------------------- #
def test_build_features_and_rank():
    prices = synthetic.make_prices(n_tickers=15, n_days=500, seed=2)
    factors = screen.build_features(prices)
    assert set(factors) == {"momentum", "trend", "low_vol", "reversal"}
    table = screen.rank_universe(prices)
    assert list(table["rank"]) == sorted(table["rank"])
    assert table["score"].is_monotonic_decreasing


def test_signal_fn_no_lookahead():
    prices = synthetic.make_prices(n_tickers=8, n_days=500, seed=9)
    t = prices.index[400]
    signal_fn = screen.make_signal_fn()
    base = signal_fn(prices.loc[:t])
    # Corrupt all future rows; the signal at t must not move.
    poisoned = prices.copy()
    poisoned.loc[poisoned.index > t] *= 5.0
    after = signal_fn(poisoned.loc[:t])
    pd.testing.assert_series_equal(base, after, check_names=False)


def test_rank_universe_asof_ignores_future_rows():
    prices = synthetic.make_prices(n_tickers=8, n_days=500, seed=10)
    t = prices.index[400]
    base = screen.rank_universe(prices, asof=t)

    poisoned = prices.copy()
    poisoned.loc[poisoned.index > t] *= np.linspace(2.0, 20.0, len(prices.columns))
    after = screen.rank_universe(poisoned, asof=t)

    pd.testing.assert_frame_equal(base, after)


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def test_backtest_runs_and_shapes():
    prices = synthetic.make_prices(n_tickers=20, n_days=900, seed=0)
    res = bt.backtest(prices, screen.make_signal_fn(), top_n=5)
    assert len(res.period_returns) > 5
    assert len(res.equity_curve) == len(res.period_returns)
    assert np.isfinite(res.stats["sharpe"])
    assert res.equity_curve.is_monotonic_increasing or res.equity_curve.min() > 0


def test_backtest_enters_next_close_after_signal():
    idx = pd.to_datetime([
        "2020-01-30",
        "2020-01-31",
        "2020-02-03",
        "2020-02-28",
        "2020-03-02",
    ])
    prices = pd.DataFrame({
        "A": [100.0, 100.0, 10.0, 999.0, 20.0],
        "B": [100.0, 100.0, 10.0, 999.0, 5.0],
    }, index=idx)
    signal_ends = []

    def signal_fn(window: pd.DataFrame) -> pd.Series:
        signal_ends.append(window.index[-1])
        return pd.Series({"A": 1.0, "B": 0.0})

    res = bt.backtest(prices, signal_fn, top_n=1, min_history=1)

    assert signal_ends == [pd.Timestamp("2020-01-31")]
    assert list(res.holdings) == [pd.Timestamp("2020-02-03")]
    assert list(res.period_returns.index) == [pd.Timestamp("2020-03-02")]
    assert np.isclose(res.period_returns.iloc[0], 1.0)  # 20 / 10 - 1


def test_backtest_bayes_runs_walk_forward():
    """`backtest --bayes` is re-enabled now that it uses the walk-forward path."""
    args = cli.build_parser().parse_args(["backtest", "--synthetic", "--bayes"])
    cli.cmd_backtest(args)                          # must not raise

    prices = synthetic.make_prices(n_tickers=20, n_days=900, seed=0)
    res = bt.backtest(prices, screen.make_bayes_signal_fn(horizon=21), top_n=5)
    assert len(res.period_returns) > 5             # warm-up skipped, then trades
    assert np.isfinite(res.stats["sharpe"])


def test_bayes_signal_fn_no_lookahead():
    """Walk-forward weights/signal at t are unchanged when post-t rows are poisoned."""
    prices = synthetic.make_prices(n_tickers=12, n_days=600, seed=11)
    t = prices.index[450]
    sfn = screen.make_bayes_signal_fn(prior_var=1.0, horizon=21, min_obs=100)
    base = sfn(prices.loc[:t])
    assert base.notna().any()                      # past warm-up: a real fit, not all-NaN

    poisoned = prices.copy()
    poisoned.loc[poisoned.index > t] *= 7.0        # wreck the future
    after = sfn(poisoned.loc[:t])
    pd.testing.assert_series_equal(base, after, check_names=False)


def test_walk_forward_not_full_sample():
    """Weights fit on the window at t differ from full-sample weights (no peeking)."""
    prices = synthetic.make_prices(n_tickers=15, n_days=600, seed=2)
    t = prices.index[300]
    win = prices.loc[:t]
    wf = bayes.fit_weights(screen.build_features(win), win, horizon=21).weights
    full = bayes.fit_weights(screen.build_features(prices), prices, horizon=21).weights
    assert set(wf) == set(full)
    assert any(not np.isclose(wf[k], full[k], atol=1e-6) for k in wf)


def test_momentum_has_edge_vs_reversed():
    """On persistent-drift data the momentum book beats its sign-flipped twin."""
    prices = synthetic.make_prices(n_tickers=30, n_days=900, seed=0, drift_spread=0.002)
    long = bt.backtest(prices, screen.make_signal_fn(screen.DEFAULT_WEIGHTS), top_n=6)
    flipped = {k: -v for k, v in screen.DEFAULT_WEIGHTS.items()}
    short = bt.backtest(prices, screen.make_signal_fn(flipped), top_n=6)
    assert long.stats["total_return"] > short.stats["total_return"]


# --------------------------------------------------------------------------- #
# bayes (step 1)
# --------------------------------------------------------------------------- #
def test_bayes_recovers_positive_sign():
    prices, exposure = synthetic.make_factor_panel(true_beta=0.03, seed=1)
    factors = {"exp": sc.zscore_factor(exposure)}
    post = bayes.fit_weights(factors, prices, horizon=1, prior_var=1.0)
    assert post.weights["exp"] > 0
    assert post.signal_to_noise["exp"] > 2.0          # clearly significant


def test_bayes_recovers_negative_sign():
    prices, exposure = synthetic.make_factor_panel(true_beta=-0.03, seed=4)
    factors = {"exp": sc.zscore_factor(exposure)}
    post = bayes.fit_weights(factors, prices, horizon=1, prior_var=1.0)
    assert post.weights["exp"] < 0


def test_bayes_noise_factor_is_shrunk():
    """The real factor must earn a larger weight than a pure-noise factor."""
    prices, exposure = synthetic.make_factor_panel(true_beta=0.03, seed=1)
    rng = np.random.default_rng(99)
    noise = pd.DataFrame(rng.normal(size=exposure.shape),
                         index=exposure.index, columns=exposure.columns)
    factors = {"exp": sc.zscore_factor(exposure), "noise": sc.zscore_factor(noise)}
    post = bayes.fit_weights(factors, prices, horizon=1, prior_var=1.0)
    assert abs(post.weights["exp"]) > abs(post.weights["noise"])
    assert abs(post.signal_to_noise["noise"]) < 2.0


def test_bayes_posterior_tightens_with_data():
    factors_small, prices_small = _factor_subset(n_days=150, seed=1)
    factors_big, prices_big = _factor_subset(n_days=600, seed=1)
    small = bayes.fit_weights(factors_small, prices_small, horizon=1)
    big = bayes.fit_weights(factors_big, prices_big, horizon=1)
    assert big.std[0] < small.std[0]                  # more data -> tighter posterior


def test_bayes_scorer_is_dropin():
    prices, exposure = synthetic.make_factor_panel(true_beta=0.03, seed=1)
    factors = screen.build_features(prices)
    scorer = bayes.BayesianScorer(prior_var=1.0, horizon=21).fit(factors, prices)
    comp = scorer.composite(factors)
    hard = sc.composite_score(factors, screen.DEFAULT_WEIGHTS)
    assert comp.shape == hard.shape
    assert list(comp.columns) == list(hard.columns)
    assert set(scorer.weights) == set(factors)


def _factor_subset(n_days: int, seed: int):
    prices, exposure = synthetic.make_factor_panel(n_days=n_days, true_beta=0.03, seed=seed)
    return {"exp": sc.zscore_factor(exposure)}, prices


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:  # noqa: BLE001 - test runner surfaces everything
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
