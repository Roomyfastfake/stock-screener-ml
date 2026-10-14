"""Offline test-suite. Runs on deterministic synthetic data, no network.

    py -3.14 tests/test_screener.py          # (or: PYTHONPATH=. python ...)

Plain asserts + a tiny runner so there is no pytest dependency.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

import main as cli
from screener import backtest as bt
from screener import bayes
from screener import cache
from screener import indicators as ind
from screener import score as sc
from screener import screen
from screener import synthetic
from screener import universe


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


def test_backtest_costs_and_turnover():
    """Hand-checked: A held, then rotate fully A -> B; costs hit traded notional."""
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03",
                          "2020-02-28", "2020-03-02", "2020-03-31", "2020-04-01"])
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0],
                           "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 120.0]},
                          index=idx)
    picks = iter(["A", "B"])

    def signal_fn(window: pd.DataFrame) -> pd.Series:
        best = next(picks)
        return pd.Series({c: float(c == best) for c in window.columns})

    free = bt.backtest(prices, signal_fn, top_n=1, min_history=1)
    picks = iter(["A", "B"])
    paid = bt.backtest(prices, signal_fn, top_n=1, min_history=1, cost_bps=50)

    assert np.allclose(free.period_returns, [0.10, 0.20])
    assert np.allclose(paid.turnover, [0.5, 1.0])          # buy book, then full rotation
    # traded notional 1.0 then 2.0, at 50 bps per unit
    assert np.allclose(paid.period_returns, [0.10 - 0.005, 0.20 - 0.010])
    assert np.allclose(paid.gross_returns, free.period_returns)
    assert np.isclose(paid.stats["avg_turnover"], 0.75)


def test_backtest_costs_reduce_return_monotonically():
    prices = synthetic.make_prices(n_tickers=20, n_days=900, seed=0)
    sfn = screen.make_signal_fn()
    totals = [bt.backtest(prices, sfn, top_n=5, cost_bps=c).stats["total_return"]
              for c in (0, 10, 50)]
    assert totals[0] > totals[1] > totals[2]


def test_external_engine_parity():
    """`backtester` (additive costs) reproduces the builtin engine period by period."""
    try:
        import backtester
    except ImportError:
        print("  (skipped: backtester package not installed)")
        return
    prices = synthetic.make_prices(n_tickers=20, n_days=900, seed=0)
    sfn = screen.make_signal_fn()
    ours = bt.backtest(prices, sfn, top_n=5, cost_bps=25)
    theirs = backtester.backtest(prices, sfn, top_n=5, cost_bps=25, cost_model="additive")
    common = ours.period_returns.index            # external engine adds a tail period
    pd.testing.assert_series_equal(ours.period_returns,
                                   theirs.period_returns.loc[common],
                                   check_names=False, rtol=0, atol=1e-12)
    assert ours.holdings == {d: theirs.holdings[d] for d in ours.holdings}


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
# fast signals: must equal the full-window reference exactly (up to float order)
# --------------------------------------------------------------------------- #
def _gappy_prices(seed=21):
    """Synthetic panel with late listings, holes and an early delisting."""
    prices = synthetic.make_prices(n_tickers=14, n_days=700, seed=seed)
    rng = np.random.default_rng(seed)
    cols = prices.columns
    prices.loc[prices.index[:300], cols[0]] = np.nan          # lists late
    prices.loc[prices.index[:450], cols[1]] = np.nan          # lists later
    prices.loc[prices.index[500:], cols[2]] = np.nan          # delists
    holes = rng.random(prices.shape) < 0.01                   # scattered missing days
    return prices.mask(holes)


def _reference_fixed_signal(window, weights=None):
    weights = screen.DEFAULT_WEIGHTS if weights is None else weights
    return sc.composite_score(screen.build_features(window), weights).iloc[-1]


def _reference_bayes_signal(window, horizon=21, prior_var=1.0, min_obs=100):
    """The pre-optimisation implementation: refit on the whole window."""
    factors = screen.build_features(window)
    try:
        post = bayes.fit_weights(factors, window, horizon=horizon, prior_var=prior_var)
    except ValueError:
        return pd.Series(np.nan, index=window.columns), None
    if post.n_obs < min_obs:
        return pd.Series(np.nan, index=window.columns), post
    return sc.composite_score(factors, post.weights).iloc[-1], post


def test_warmup_rows_cover_every_lookback():
    assert screen.WARMUP_ROWS >= screen.MOM_LOOKBACK + 1
    assert screen.WARMUP_ROWS >= screen.VOL_WINDOW + 1


def test_fast_fixed_signal_matches_full_window():
    prices = _gappy_prices()
    sfn = screen.make_signal_fn()
    for i in range(100, len(prices), 37):
        w = prices.iloc[:i + 1]
        pd.testing.assert_series_equal(sfn(w), _reference_fixed_signal(w),
                                       check_names=False, rtol=0, atol=1e-10)


def test_fast_bayes_signal_matches_full_refit():
    prices = _gappy_prices()
    sfn = screen.make_bayes_signal_fn(horizon=21, min_obs=100)
    checked = 0
    for i in range(150, len(prices), 23):                     # sequential: incremental path
        w = prices.iloc[:i + 1]
        ref, post = _reference_bayes_signal(w)
        fast = sfn(w)
        pd.testing.assert_series_equal(fast, ref, check_names=False, rtol=0, atol=1e-9)
        if post is not None and post.n_obs >= 100:
            assert sfn._stats.n == post.n_obs
            checked += 1
    assert checked > 10                                        # real fits, not just warm-up


def test_fast_bayes_resets_on_shorter_window():
    """Reusing an instance on an earlier window must not keep later training rows."""
    prices = _gappy_prices()
    sfn = screen.make_bayes_signal_fn(horizon=21, min_obs=100)
    sfn(prices)                                                # absorb everything
    short = prices.iloc[:400]
    pd.testing.assert_series_equal(sfn(short), _reference_bayes_signal(short)[0],
                                   check_names=False, rtol=0, atol=1e-9)
    assert sfn._stats.n == _reference_bayes_signal(short)[1].n_obs


def test_fast_bayes_resets_on_changed_history():
    prices = _gappy_prices()
    sfn = screen.make_bayes_signal_fn(horizon=21, min_obs=100)
    sfn(prices.iloc[:400])
    edited = prices.iloc[:450].copy()
    edited.iloc[100:110] *= 1.5                                # restated past (e.g. new adjustment)
    pd.testing.assert_series_equal(sfn(edited), _reference_bayes_signal(edited)[0],
                                   check_names=False, rtol=0, atol=1e-9)


def test_design_stats_match_direct_regression():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([0.3, -0.1, 0.0]) + rng.normal(scale=0.5, size=500)
    mean, cov, nv = bayes.bayesian_linear_regression(X, y, prior_var=1.0)
    # Direct residual formula (pre-refactor implementation).
    beta0 = np.linalg.solve(X.T @ X + np.eye(3), X.T @ y)
    resid = y - X @ beta0
    nv_ref = resid @ resid / (500 - 3)
    cov_ref = np.linalg.inv(X.T @ X / nv_ref + np.eye(3))
    assert np.isclose(nv, nv_ref, rtol=1e-10)
    assert np.allclose(mean, cov_ref @ X.T @ y / nv_ref, rtol=1e-10)
    halves = (bayes.DesignStats.from_arrays(X[:200], y[:200], list("abc"))
              + bayes.DesignStats.from_arrays(X[200:], y[200:], list("abc")))
    whole = bayes.DesignStats.from_arrays(X, y, list("abc"))
    assert np.allclose(halves.xtx, whole.xtx) and halves.n == whole.n


# --------------------------------------------------------------------------- #
# universe + cache (step 4)
# --------------------------------------------------------------------------- #
def test_universe_csv_normalises_and_filters():
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "u.csv"
        path.write_text("# snapshot\nticker,name,sector\nbrk.b,Berkshire,Financials\n"
                        "AAPL,Apple,Information Technology\nAAPL,dup,Information Technology\n"
                        "JPM,JPMorgan,Financials\n", "utf-8")
        assert universe.load_universe(path) == ["BRK-B", "AAPL", "JPM"]
        assert universe.load_universe(path, sector="financials") == ["BRK-B", "JPM"]


def test_bundled_sp500_snapshot():
    tickers = universe.load_sp500()
    assert 490 <= len(tickers) <= 510
    assert {"AAPL", "MSFT", "BRK-B"} <= set(tickers)
    assert all("." not in t for t in tickers)


class _FakeDownloader:
    """Deterministic stand-in for yfinance; records every call."""

    def __init__(self, known, scale=1.0):
        self.known, self.scale, self.calls = set(known), scale, []
        self.index = pd.bdate_range("2019-01-01", "2020-12-31")

    def __call__(self, tickers, start, end):
        self.calls.append(list(tickers))
        idx = self.index[self.index >= pd.Timestamp(start)]
        cols = {t: pd.Series(self.scale * (100 + np.arange(len(idx)) + i), index=idx)
                for i, t in enumerate(sorted(self.known)) if t in tickers}
        return pd.DataFrame(cols)


def test_cache_hit_avoids_download_and_matches():
    with tempfile.TemporaryDirectory() as d:
        fake = _FakeDownloader(["A", "B"])
        first = cache.PriceCache(d, downloader=fake).get(["A", "B"], "2019-06-01", "2020-06-30")
        again = cache.PriceCache(d, downloader=fake).get(["A", "B"], "2019-06-01", "2020-06-30")
        assert len(fake.calls) == 1
        pd.testing.assert_frame_equal(first, again, check_freq=False)
        assert first.index[0] >= pd.Timestamp("2019-06-01")
        assert first.index[-1] <= pd.Timestamp("2020-06-30")


def test_cache_offline_miss_raises_and_hit_works():
    with tempfile.TemporaryDirectory() as d:
        try:
            cache.PriceCache(d).get(["A"], "2019-06-01", offline=True)
            raise AssertionError("expected CacheMiss")
        except cache.CacheMiss:
            pass
        cache.PriceCache(d, downloader=_FakeDownloader(["A"])).get(["A"], "2019-06-01")
        out = cache.PriceCache(d).get(["A"], "2019-06-01", offline=True)
        assert list(out.columns) == ["A"] and len(out) > 100
        try:                                           # cached from June: earlier start is a miss
            cache.PriceCache(d).get(["A"], "2019-01-01", offline=True)
            raise AssertionError("expected CacheMiss for uncovered start")
        except cache.CacheMiss:
            pass


def test_cache_staleness_rules():
    with tempfile.TemporaryDirectory() as d:
        fake = _FakeDownloader(["A"])
        pc = cache.PriceCache(d, downloader=fake)
        pc.get(["A"], "2019-06-01")
        pc.manifest["A"]["fetched_at"] = "2020-07-01T00:00:00"    # pretend: old fetch
        pc.manifest["A"]["complete_through"] = "2020-06-30"
        pc.get(["A"], "2019-06-01", end="2020-06-30")              # closes final through end
        assert len(fake.calls) == 1
        # Fetched during market hours on the end date: that close was not final yet.
        pc.manifest["A"]["complete_through"] = "2020-06-29"
        pc.get(["A"], "2019-06-01", end="2020-06-30")
        assert len(fake.calls) == 2
        pc.manifest["A"]["fetched_at"] = "2020-07-01T00:00:00"
        pc.manifest["A"]["complete_through"] = "2020-06-30"
        pc.get(["A"], "2019-06-01")                                # open-ended + old: refetch
        assert len(fake.calls) == 3
        pc.get(["A"], "2019-01-01")                                # earlier start: refetch
        assert len(fake.calls) == 4
        pc.get(["A"], "2019-06-01", refresh=True)
        assert len(fake.calls) == 5


def test_cache_refetch_replaces_restated_history():
    """Adjusted closes get restated; a refetch must overwrite, never splice."""
    with tempfile.TemporaryDirectory() as d:
        cache.PriceCache(d, downloader=_FakeDownloader(["A"], scale=1.0)).get(["A"], "2019-06-01")
        restated = cache.PriceCache(d, downloader=_FakeDownloader(["A"], scale=0.5))
        out = restated.get(["A"], "2019-06-01", refresh=True)
        expected = _FakeDownloader(["A"], scale=0.5)(["A"], "2019-06-01", None)
        assert np.allclose(out["A"].to_numpy(), expected["A"].to_numpy())


def test_cache_returns_single_block_frame():
    """Per-column blocks made every frame op ~15x slower on 500 names."""
    with tempfile.TemporaryDirectory() as d:
        out = cache.PriceCache(d, downloader=_FakeDownloader(list("ABCDE"))).get(
            list("ABCDE"), "2019-06-01")
        assert out._mgr.nblocks == 1                  # private, but the thing that matters


def test_cache_drops_unknown_tickers():
    with tempfile.TemporaryDirectory() as d:
        pc = cache.PriceCache(d, downloader=_FakeDownloader(["A"]))
        out = pc.get(["A", "ZZZZ"], "2019-06-01")
        assert list(out.columns) == ["A"]
        assert pc.last_missing == ["ZZZZ"]


def test_last_complete_session_excludes_intraday_bar():
    et = "America/New_York"
    assert cache.last_complete_session(pd.Timestamp("2026-10-14 14:00", tz=et)) == pd.Timestamp("2026-10-13")
    assert cache.last_complete_session(pd.Timestamp("2026-10-14 17:00", tz=et)) == pd.Timestamp("2026-10-14")


def test_cache_retries_transient_failures_once():
    with tempfile.TemporaryDirectory() as d:
        inner = _FakeDownloader(["A", "B"])

        def flaky(tickers, start, end):               # B fails on the first attempt only
            out = inner(tickers, start, end)
            return out.drop(columns="B") if len(inner.calls) == 1 else out

        pc = cache.PriceCache(d, downloader=flaky)
        out = pc.get(["A", "B"], "2019-06-01")
        assert list(out.columns) == ["A", "B"]
        assert inner.calls == [["A", "B"], ["B"]]


def test_cli_uses_cache_offline():
    with tempfile.TemporaryDirectory() as d:
        prices = synthetic.make_prices(n_tickers=12, n_days=500, seed=0)

        def fake(tickers, start, end):
            return prices[[t for t in tickers if t in prices.columns]]

        cache.PriceCache(d, downloader=fake).get(list(prices.columns), "2000-01-01")
        args = cli.build_parser().parse_args(
            ["backtest", "--tickers", *prices.columns, "--start", "2000-01-01",
             "--offline", "--cache-dir", d, "--top-n", "3"])
        cli.cmd_backtest(args)                     # offline: must not hit the network


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
