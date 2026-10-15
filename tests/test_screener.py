"""Offline test-suite. Runs on deterministic synthetic data, no network.

    py -3.14 tests/test_screener.py          # (or: PYTHONPATH=. python ...)

Plain asserts + a tiny runner so there is no pytest dependency.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

import main as cli
from screener import backtest as bt
from screener import bayes, cache, research, screen, synthetic, universe
from screener import fundamentals as fu
from screener import indicators as ind
from screener import score as sc


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
    # Two monthly periods: CAGR = (1.095 * 1.19) ** (12 / 2) - 1.
    assert np.isclose(paid.stats["cagr"], (1.095 * 1.19) ** 6 - 1)


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


def test_factor_registry_long_lookbacks_stay_exact():
    """All factors, incl. 252-day ones: the tail path must equal the full window."""
    prices = _gappy_prices()
    names = list(screen.FACTORS)
    assert screen.warmup_rows(names) == screen.LONG_MOM_LOOKBACK + 1
    sfn = screen.make_signal_fn(factors=names)
    bfn = screen.make_bayes_signal_fn(horizon=21, min_obs=100, factors=names)
    weights = screen.default_weights(factors=names)
    checked = 0
    for i in range(260, len(prices), 29):
        w = prices.iloc[:i + 1]
        full = screen.build_features(w, factors=names)
        pd.testing.assert_series_equal(sfn(w), sc.composite_score(full, weights).iloc[-1],
                                       check_names=False, rtol=0, atol=1e-10)
        got = bfn(w)
        try:
            post = bayes.fit_weights(full, w, horizon=21)
        except ValueError:
            continue
        if post.n_obs >= 100:
            ref = sc.composite_score(full, post.weights).iloc[-1]
            pd.testing.assert_series_equal(got, ref, check_names=False, rtol=0, atol=1e-9)
            checked += 1
    assert checked > 5


def test_high_proximity_and_unknown_factor():
    s = pd.Series([1.0, 2.0, 4.0, 3.0, 2.0])
    hp = ind.high_proximity(s, window=3)                      # min_periods = int(2.7) = 2
    assert np.isnan(hp.iloc[0]) and hp.iloc[2] == 0.0 and np.isclose(hp.iloc[4], 2 / 4 - 1)
    holey = pd.Series(np.r_[np.arange(1.0, 301.0)])
    holey.iloc[150] = np.nan                                   # one missing day
    assert ind.high_proximity(holey, 252).iloc[260:].notna().all()
    try:
        screen.build_features(synthetic.make_prices(n_days=50), factors=["nope"])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "nope" in str(exc)


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
# fundamentals (step 3): point-in-time SEC companyfacts
# --------------------------------------------------------------------------- #
def _fact(start, end, val, filed, form="10-Q", fy=2099, fp="Q9"):
    # fy/fp deliberately wrong: the parser must identify periods by start/end only.
    f = {"end": end, "val": val, "filed": filed, "form": form, "fy": fy, "fp": fp}
    if start is not None:
        f["start"] = start
    return f


def _company_facts(rev_tag_2020="Revenues", extra_rev=()):
    """Calendar-year filer. Revenue 2019 Q1-Q3 = 100/110/120, FY 460 (Q4 130);
    2020 Q1-Q3 = 120/130/140, FY 540 (Q4 150). Net income 10/qtr in 2019 (FY 40),
    12/qtr in 2020 (FY 48). Equity 200 at every quarter end. 10-Qs carry 3M and
    9M YTD facts; Q4 is never reported discretely."""
    rev19 = [_fact("2019-01-01", "2019-03-31", 100, "2019-05-01"),
             _fact("2019-04-01", "2019-06-30", 110, "2019-08-01"),
             _fact("2019-07-01", "2019-09-30", 120, "2019-11-01"),
             _fact("2019-01-01", "2019-09-30", 330, "2019-11-01"),
             _fact("2019-01-01", "2019-12-31", 460, "2020-02-14", "10-K")]
    rev20 = [_fact("2020-01-01", "2020-03-31", 120, "2020-05-01"),
             _fact("2020-04-01", "2020-06-30", 130, "2020-08-03"),
             _fact("2020-07-01", "2020-09-30", 140, "2020-11-02"),
             _fact("2020-01-01", "2020-09-30", 390, "2020-11-02"),
             _fact("2020-01-01", "2020-12-31", 540, "2021-02-12", "10-K")]
    filings = {"2019-03-31": "2019-05-01", "2019-06-30": "2019-08-01",
               "2019-09-30": "2019-11-01", "2019-12-31": "2020-02-14",
               "2020-03-31": "2020-05-01", "2020-06-30": "2020-08-03",
               "2020-09-30": "2020-11-02", "2020-12-31": "2021-02-12"}
    ni, eq = [], []
    for end, filed in filings.items():
        q = 10 if end.startswith("2019") else 12
        y = end[:4]
        if end.endswith("12-31"):
            ni.append(_fact(f"{y}-01-01", end, 4 * q, filed, "10-K"))
        else:
            ni.append(_fact(f"{y}-{int(end[5:7]) - 2:02d}-01", end, q, filed))
            if end.endswith("09-30"):
                ni.append(_fact(f"{y}-01-01", end, 3 * q, filed))
        eq.append(_fact(None, end, 200, filed, "10-K" if end.endswith("12-31") else "10-Q"))
    gaap = {"NetIncomeLoss": {"units": {"USD": ni}},
            "StockholdersEquity": {"units": {"USD": eq}}}
    rev = {"Revenues": rev19 if rev_tag_2020 != "Revenues" else rev19 + rev20}
    if rev_tag_2020 != "Revenues":
        rev[rev_tag_2020] = rev20
    for tag, facts in rev.items():
        gaap[tag] = {"units": {"USD": facts + [f for t, f in extra_rev if t == tag]}}
    return {"facts": {"us-gaap": gaap}}


def test_fundamentals_q4_derivation_and_ttm():
    rows = fu.extract(_company_facts())
    revq = fu.discrete_quarters(fu._latest_known(rows[rows.concept == "revenue"],
                                                 pd.Timestamp("2021-03-01")))
    assert revq[pd.Timestamp("2019-12-31")] == 130 and revq[pd.Timestamp("2020-12-31")] == 150
    assert fu.ttm(revq, pd.Timestamp("2020-12-31")) == 540     # equals the 10-K FY value
    s = fu.snapshot(rows, pd.Timestamp("2021-02-12"))
    assert s["period_end"] == pd.Timestamp("2020-12-31")
    assert np.isclose(s["rev_growth"], 540 / 460 - 1)
    assert np.isclose(s["roe"], 48 / 200)
    mid = fu.snapshot(rows, pd.Timestamp("2020-11-02"))       # TTM = Q4'19 + Q1-Q3'20
    assert np.isclose(mid["roe"], (10 + 12 + 12 + 12) / 200)
    assert np.isnan(mid["rev_growth"])                         # no TTM a year earlier yet


def test_fundamentals_fast_events_match_reference_snapshots():
    variants = [_company_facts(),
                _company_facts("RevenueFromContractWithCustomerExcludingAssessedTax"),
                _company_facts(extra_rev=[("Revenues", _fact("2020-01-01", "2020-03-31", 125,
                                                             "2021-05-03"))])]
    for facts in variants:
        rows = fu.extract(facts)
        ev = fu.events(rows)
        assert len(ev) == rows["filed"].nunique()
        for d in ev.index:
            ref = fu.snapshot(rows, d)
            assert ev.loc[d, "period_end"] == ref["period_end"]
            for f in fu.FACTORS:
                got = float(ev.loc[d, f])
                assert (np.isnan(got) and np.isnan(ref[f])) or np.isclose(got, ref[f]), (d, f)


def test_fundamentals_ttm_consistency_gate():
    rows = fu.extract(_company_facts())
    assert fu.ttm_consistency(rows, "revenue") == (2, 2)
    bad = _company_facts()                                     # Q2'20 double-reported wrong
    bad["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        _fact("2020-04-01", "2020-06-30", 999, "2020-08-04"))
    assert fu.ttm_consistency(fu.extract(bad), "revenue") == (2, 1)


def test_fundamentals_visible_only_after_filing_day():
    idx = pd.bdate_range("2020-12-01", "2021-03-31")
    daily = fu.to_daily(fu.events(fu.extract(_company_facts())), idx)
    filed = pd.Timestamp("2021-02-12")                         # Friday
    assert np.isclose(daily.loc[filed, "roe"], 46 / 200)       # still the Q3 snapshot
    assert np.isclose(daily.loc[pd.Timestamp("2021-02-15"), "roe"], 48 / 200)


def test_fundamentals_tag_switch_merges_history():
    same = fu.snapshot(fu.extract(_company_facts()), pd.Timestamp("2021-03-01"))
    switched = fu.snapshot(fu.extract(_company_facts(
        "RevenueFromContractWithCustomerExcludingAssessedTax")), pd.Timestamp("2021-03-01"))
    assert np.isclose(same["rev_growth"], switched["rev_growth"])


def test_fundamentals_restatement_does_not_rewrite_past():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    base = fu.to_daily(fu.events(fu.extract(_company_facts())), idx)
    restated = _company_facts(extra_rev=[("Revenues",
                                          _fact("2020-01-01", "2020-03-31", 125, "2021-05-03"))])
    after = fu.to_daily(fu.events(fu.extract(restated)), idx)
    cut = pd.Timestamp("2021-05-03")
    pd.testing.assert_frame_equal(base.loc[:cut], after.loc[:cut])
    assert np.isclose(after.loc[pd.Timestamp("2021-05-04"), "rev_growth"], 545 / 460 - 1)


def test_fundamentals_poisoned_future_filing_ignored():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    base = fu.to_daily(fu.events(fu.extract(_company_facts())), idx)
    poison = _company_facts(extra_rev=[("Revenues",
                                        _fact("2019-01-01", "2019-03-31", 1e9, "2021-06-01"))])
    after = fu.to_daily(fu.events(fu.extract(poison)), idx)
    t = pd.Timestamp("2021-06-01")
    pd.testing.assert_frame_equal(base.loc[:t], after.loc[:t])


def test_fundamentals_stale_values_expire():
    idx = pd.bdate_range("2021-01-01", "2022-06-30")
    daily = fu.to_daily(fu.events(fu.extract(_company_facts())), idx)
    end = pd.Timestamp("2020-12-31")
    assert daily.loc[end + pd.Timedelta(days=400):, "roe"].notna().any()
    assert daily.loc[end + fu.MAX_STALENESS + pd.Timedelta(days=3):, "roe"].isna().all()


def test_fundamentals_cache_and_user_agent():
    with tempfile.TemporaryDirectory() as d:
        class FakeClient:
            calls = 0

            def company_facts(self, cik):
                FakeClient.calls += 1
                return _company_facts()

        fc = fu.FundamentalsCache(d, client=FakeClient())
        try:
            fu.FundamentalsCache(d).rows(1, offline=True)
            raise AssertionError("expected offline miss")
        except LookupError:
            pass
        first = fc.rows(1)
        again = fu.FundamentalsCache(d).rows(1, offline=True)
        pd.testing.assert_frame_equal(first, again)
        assert FakeClient.calls == 1

        panel, unmapped = fu.load_panel(["AAA", "AAB", "ZZZ"], pd.bdate_range("2020-01-01", "2021-06-30"),
                                        {"AAA": 1, "AAB": 1}, cache=fc)
        assert unmapped == ["ZZZ"] and FakeClient.calls == 1     # share classes fetched once
        assert list(panel["roe"].columns) == ["AAA", "AAB", "ZZZ"]
        assert panel["roe"]["ZZZ"].isna().all()

    saved = os.environ.pop("SEC_USER_AGENT", None)
    try:
        fu.SecClient()
        raise AssertionError("expected missing User-Agent error")
    except RuntimeError as exc:
        assert "SEC_USER_AGENT" in str(exc)
    finally:
        if saved is not None:
            os.environ["SEC_USER_AGENT"] = saved


def test_cli_fundamentals_offline_from_cache():
    with tempfile.TemporaryDirectory() as d:
        prices = synthetic.make_prices(n_tickers=6, n_days=400, seed=0)
        cache.PriceCache(d, downloader=lambda t, s, e: prices[t]).get(list(prices.columns),
                                                                      "2000-01-01")
        sec_dir = pathlib.Path(d) / "sec"
        sec_dir.mkdir()
        rows = fu.extract(_company_facts())
        ciks = {t: 1000 + i for i, t in enumerate(prices.columns)}
        for cik in ciks.values():
            rows.to_parquet(sec_dir / f"{cik:010d}.parquet")
        orig = universe.cik_map
        universe.cik_map = lambda path=None: ciks
        try:
            args = cli.build_parser().parse_args(
                ["backtest", "--tickers", *prices.columns, "--start", "2000-01-01", "--offline",
                 "--cache-dir", d, "--sec-cache-dir", str(sec_dir), "--fundamentals",
                 "--top-n", "2"])
            cli.cmd_backtest(args)                # cache only: no SEC_USER_AGENT needed
        finally:
            universe.cik_map = orig


def _fake_fundamentals(prices, seed=5):
    """Step-like PIT panels (values change ~quarterly), with gaps."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("roe", "rev_growth"):
        raw = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        raw.iloc[::63] = rng.normal(size=raw.iloc[::63].shape)
        raw = raw.ffill()
        out[name] = raw.mask(rng.random(raw.shape) < 0.1)
    return out


def test_signals_with_fundamentals_match_reference():
    prices = _gappy_prices()
    fund = _fake_fundamentals(prices)
    weights = screen.default_weights(fund)
    sfn = screen.make_signal_fn(fundamentals=fund)
    bfn = screen.make_bayes_signal_fn(horizon=21, min_obs=100, fundamentals=fund)
    for i in range(150, len(prices), 41):
        w = prices.iloc[:i + 1]
        full = screen.build_features(w, fundamentals=fund)
        ref = sc.composite_score(full, weights).iloc[-1]
        pd.testing.assert_series_equal(sfn(w), ref, check_names=False, rtol=0, atol=1e-10)
        try:
            post = bayes.fit_weights(full, w, horizon=21)
        except ValueError:
            continue
        if post.n_obs >= 100:
            ref_b = sc.composite_score(full, post.weights).iloc[-1]
            pd.testing.assert_series_equal(bfn(w), ref_b, check_names=False, rtol=0, atol=1e-9)
        else:
            bfn(w)
    assert set(screen.build_features(prices, fundamentals=fund)) >= {"roe", "rev_growth"}


# --------------------------------------------------------------------------- #
# factor research
# --------------------------------------------------------------------------- #
def test_rank_ic_matches_pandas_spearman():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=5)
    f = pd.DataFrame(rng.normal(size=(5, 30)), index=idx)
    r = pd.DataFrame(rng.normal(size=(5, 30)), index=idx)
    f.iloc[2, :4] = np.nan                                     # names missing on one side only
    r.iloc[2, 4:7] = np.nan
    ic = research.rank_ic(f, r, idx)
    for d in idx:
        ok = f.loc[d].notna() & r.loc[d].notna()
        # Spearman = Pearson on ranks (pandas' method="spearman" would need scipy).
        ref = f.loc[d][ok].rank().corr(r.loc[d][ok].rank())
        assert np.isclose(ic[d], ref)


def test_research_uses_next_close_timing():
    """Exposure at t drives the t -> t+1 return. Entering at the next close (the
    backtest's timing) must not see it; a same-close entry would."""
    prices, exposure = synthetic.make_factor_panel(true_beta=0.03, seed=1, n_days=800)
    dates = research.sample_dates(prices.index, 1, step=1)
    same_close = research.rank_ic(exposure, research.forward_returns(prices, 1, lag=0), dates)
    next_close = research.rank_ic(exposure, research.forward_returns(prices, 1), dates)
    assert same_close.mean() > 0.5
    assert abs(next_close.mean()) < 0.03


def test_research_finds_momentum_and_spread():
    prices = synthetic.make_prices(n_tickers=40, n_days=900, seed=0, drift_spread=0.002)
    factors = screen.build_features(prices)
    ic = research.ic_table(factors, prices, horizon=21)
    assert ic.loc["momentum", "ic_mean"] > 0.1 and ic.loc["momentum", "t_stat"] > 3
    q = research.quantile_returns(factors["momentum"], prices, horizon=21)
    assert q["spread_mean"] > 0 and q["spread_t"] > 2
    assert q["quantile_mean"].iloc[-1] > q["quantile_mean"].iloc[0]
    # Non-overlapping sampling: one date per 21 rows.
    assert ic.loc["momentum", "n_dates"] <= len(prices) // 21 + 1


def test_factor_correlation_identities():
    prices = synthetic.make_prices(n_tickers=25, n_days=400, seed=3)
    m = screen.build_features(prices)["momentum"]
    dates = research.sample_dates(prices.index, 21)
    corr = research.factor_correlation({"a": m, "b": -m, "c": m}, dates)
    assert np.isclose(corr.loc["a", "c"], 1.0) and np.isclose(corr.loc["a", "b"], -1.0)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)


def test_report_builds_valid_page():
    from screener import report
    prices = synthetic.make_prices(n_tickers=20, n_days=700, seed=0)
    runs = {"A": bt.backtest(prices, screen.make_signal_fn(), top_n=5, cost_bps=10),
            "B": bt.backtest(prices, screen.make_bayes_signal_fn(min_obs=100), top_n=5)}
    bench = prices.mean(axis=1)
    page = report.build("T <x>", "sub", runs, bench,
                        research.report(screen.build_features(prices), prices), ["A", "B"],
                        notes=["n1"])
    assert page.startswith("<!doctype html>") and page.count("<svg") == 2
    assert "T &lt;x&gt;" in page and "SPY buy and hold" in page and "n1" in page
    curve, stats = report.benchmark_curve(bench, runs["A"])
    assert curve.iloc[0] == 1.0 and len(curve) == len(runs["A"].period_returns) + 1
    assert np.isfinite(stats["sharpe"])


def test_panel_signal_equals_per_rebalance_signal():
    """Precomputing the composite once is exact (factors at d use data <= d)."""
    prices = _gappy_prices()
    names = list(screen.FACTORS)
    comp = sc.composite_score(screen.build_features(prices, factors=names),
                              screen.default_weights(factors=names))
    fast, slow = screen.make_panel_signal_fn(comp), screen.make_signal_fn(factors=names)
    for i in range(260, len(prices), 53):
        w = prices.iloc[:i + 1]
        pd.testing.assert_series_equal(fast(w), slow(w), check_names=False, rtol=0, atol=1e-10)
    a = bt.backtest(prices, fast, top_n=4, cost_bps=10)
    b = bt.backtest(prices, slow, top_n=4, cost_bps=10)
    pd.testing.assert_series_equal(a.period_returns, b.period_returns, rtol=0, atol=1e-12)


def test_deflated_sharpe_behaviour():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.02, 0.04, 240))            # monthly Sharpe 0.5, 20 years
    noise = pd.Series(rng.normal(0.0, 0.04, 240))
    assert research.probabilistic_sharpe(good) > 0.99
    assert 0.05 < research.probabilistic_sharpe(noise) < 0.95
    # The same result is less convincing after many trials than after one.
    many = rng.normal(0.0, 0.07, 200)
    assert research.deflated_sharpe(good, many) < research.probabilistic_sharpe(good)
    assert research.expected_max_sharpe(1, 0.01) == 0.0
    assert research.expected_max_sharpe(100, 0.01) > research.expected_max_sharpe(10, 0.01) > 0


def test_cli_analyze_runs_offline():
    args = cli.build_parser().parse_args(["analyze", "--synthetic"])
    cli.cmd_analyze(args)


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


def _membership_fixture(d):
    snap = pathlib.Path(d) / "snap.csv"
    hist = pathlib.Path(d) / "hist.csv"
    # META joined as FB in 2013 and was renamed 2022-06-09; NEW was added 2026-09-21
    # (after the history file's last update), replacing OLD; GONE left in 2018.
    snap.write_text("ticker,name,sector,cik,date_added\n"
                    "META,Meta,Comm,1,2013-12-23\nAAPL,Apple,IT,2,1982-11-30\n"
                    "NEW,New Co,IT,3,2026-09-21\n", "utf-8")
    hist.write_text("# comment\nticker,start_date,end_date\n"
                    "FB,2013-12-23,2022-06-09\nMETA,2022-06-09,\nAAPL,1996-01-02,\n"
                    "OLD,2010-01-04,\nGONE,2005-01-03,2018-03-19\nXFB,2001-01-02,2022-06-09\n",
                    "utf-8")
    return snap, hist


def test_membership_intervals_handle_renames_and_late_changes():
    with tempfile.TemporaryDirectory() as d:
        iv = universe.membership_intervals(*_membership_fixture(d))
        by = {t: g for t, g in iv.groupby("ticker")}
        assert "FB" not in by                                  # rename: covered by META
        assert "XFB" in by                                     # unrelated name, same end date
        assert by["META"]["start"].iloc[0] == pd.Timestamp("2013-12-23")
        assert by["OLD"]["end"].iloc[0] == pd.Timestamp("2026-09-21")
        assert by["GONE"]["end"].iloc[0] == pd.Timestamp("2018-03-19")
        idx = pd.bdate_range("2018-03-15", "2018-03-21")
        m = universe.membership(idx, iv)
        assert m.loc["2018-03-16", "GONE"] and not m.loc["2018-03-19", "GONE"]   # end exclusive
        assert m.loc["2018-03-19", "META"]


def test_bundled_membership_is_sane():
    idx = pd.bdate_range("2015-01-01", "2026-10-01")
    m = universe.membership(idx)
    counts = m.sum(axis=1)
    assert counts.between(490, 515).all(), (counts.min(), counts.max())
    assert m.loc["2016-06-01", "META"] and "FB" not in m.columns or not m["FB"].any()
    assert len(universe.members_between("2015-01-01", "2026-10-01")) > 700


def test_mask_to_members():
    prices = synthetic.make_prices(n_tickers=6, n_days=300, seed=0)
    members = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    members.loc[prices.index[200]:, "T01"] = False             # T01 leaves the index
    sfn = screen.mask_to_members(screen.make_signal_fn(), members)
    before, after = sfn(prices.iloc[:200]), sfn(prices.iloc[:250])
    assert before.notna()["T01"] and np.isnan(after["T01"])
    assert after.drop("T01").notna().all()
    early = screen.mask_to_members(screen.make_signal_fn(), members.iloc[10:])
    assert early(prices.iloc[:5]).isna().all()                 # before membership data: nothing


def test_cache_remembers_dead_symbols():
    with tempfile.TemporaryDirectory() as d:
        fake = _FakeDownloader(["A"])
        cache.PriceCache(d, downloader=fake).get(["A", "DEAD"], "2019-06-01")
        n = len(fake.calls)                                    # initial + one retry for DEAD
        pc = cache.PriceCache(d, downloader=fake)
        out = pc.get(["A", "DEAD"], "2019-06-01")
        assert len(fake.calls) == n and pc.last_missing == ["DEAD"]
        off = cache.PriceCache(d).get(["A", "DEAD"], "2019-06-01", offline=True)
        assert list(off.columns) == ["A"] and list(out.columns) == ["A"]


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
        except Exception as exc:  # noqa: BLE001 - test runner surfaces everything
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            if os.environ.get("GITHUB_ACTIONS"):       # surface in the PR checks UI
                where = traceback.extract_tb(exc.__traceback__)[-1]
                msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
                print(f"::error file=tests/test_screener.py,line={where.lineno},"
                      f"title=FAIL {t.__name__}::{msg} (at {where.filename}:{where.lineno})")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
