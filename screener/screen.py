"""Build the factor table, rank the universe, and expose a backtest signal.

The feature table is a dict of z-scored ``(dates x tickers)`` frames. The same
``build_features`` is reused by ``rank_universe`` (point-in-time ranking for the
CLI) and by ``make_signal_fn`` (per-rebalance scoring for the backtest), so the
research and backtest paths can never diverge.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import bayes
from . import indicators as ind
from . import score as sc

# Factors are oriented so that a *higher* raw value is *more attractive*; the
# default composite weights below are therefore all positive.
DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum": 1.0,
    "trend": 0.5,
    "low_vol": 0.5,
    "reversal": 0.5,
}

# Indicator parameters used by build_features.
MOM_LOOKBACK, MOM_SKIP = 126, 21
LONG_MOM_LOOKBACK = 252                  # 12-1 momentum (Jegadeesh and Titman)
TREND_WINDOW = 50
VOL_WINDOW = 21
REV_WINDOW = 5
HIGH_WINDOW = 252                        # 52-week high

# name -> (raw factor fn, trailing rows needed for its value at the last row).
# Every factor is a fixed-length trailing window and scoring is per row, so the
# last row only depends on that many trailing rows (vol needs VOL_WINDOW returns,
# i.e. VOL_WINDOW + 1 prices).
FACTORS: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], int]] = {
    "momentum": (lambda p: ind.momentum(p, MOM_LOOKBACK, MOM_SKIP), MOM_LOOKBACK + 1),
    "trend": (lambda p: ind.trend(p, TREND_WINDOW), TREND_WINDOW),
    "low_vol": (lambda p: ind.low_volatility(p, VOL_WINDOW), VOL_WINDOW + 1),
    "reversal": (lambda p: ind.reversal(p, REV_WINDOW), REV_WINDOW + 1),
    "mom_12_1": (lambda p: ind.momentum(p, LONG_MOM_LOOKBACK, MOM_SKIP), LONG_MOM_LOOKBACK + 1),
    "high_52w": (lambda p: ind.high_proximity(p, HIGH_WINDOW), HIGH_WINDOW),
}
PRICE_FACTORS = list(DEFAULT_WEIGHTS)    # the default selection
EXTRA_WEIGHTS: dict[str, float] = {"mom_12_1": 1.0, "high_52w": 0.5}


def warmup_rows(factors: list[str] | None = None) -> int:
    return max(FACTORS[f][1] for f in (factors or PRICE_FACTORS))


WARMUP_ROWS = warmup_rows(PRICE_FACTORS)


FUNDAMENTAL_WEIGHTS: dict[str, float] = {"roe": 0.5, "rev_growth": 0.5}

Panel = dict[str, pd.DataFrame]


def build_features(prices: pd.DataFrame,
                   limits: tuple[float, float] = (0.05, 0.05),
                   fundamentals: Panel | None = None,
                   factors: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Compute z-scored factor frames from a ``(dates x tickers)`` price panel.

    ``factors`` selects price factors from ``FACTORS`` (default ``PRICE_FACTORS``).
    Each is lookahead-free (built only from ``shift``/``rolling``) and then
    winsorized + cross-sectionally z-scored. ``fundamentals`` are optional
    precomputed point-in-time raw panels (``fundamentals.load_panel``); they are
    aligned to ``prices`` and scored the same way.
    """
    unknown = set(factors or ()) - set(FACTORS)
    if unknown:
        raise ValueError(f"unknown factors {sorted(unknown)}; choose from {sorted(FACTORS)}")
    raw = {name: FACTORS[name][0](prices) for name in (factors or PRICE_FACTORS)}
    for name, panel in (fundamentals or {}).items():
        raw[name] = panel.reindex(index=prices.index, columns=prices.columns)
    return {name: sc.zscore_factor(frame, limits) for name, frame in raw.items()}


def latest_features(prices: pd.DataFrame,
                    limits: tuple[float, float] = (0.05, 0.05),
                    fundamentals: Panel | None = None,
                    factors: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """``build_features`` at the last row only (one-row frames), from the trailing
    ``warmup_rows(factors)`` rows. Same values, O(1) in history length (fundamental
    panels are already point-in-time per date, so only their last row is read)."""
    tail = prices.iloc[-warmup_rows(factors):]
    return {name: f.iloc[[-1]]
            for name, f in build_features(tail, limits, fundamentals, factors).items()}


def rank_universe(prices: pd.DataFrame,
                  weights: dict[str, float] | None = None,
                  asof=None, fundamentals: Panel | None = None,
                  factors: list[str] | None = None) -> pd.DataFrame:
    """Rank the universe by composite score at a single date (default: last).

    Returns a table indexed by ticker with each factor's z-score, the composite
    score, and the rank (1 = most attractive). Names with no valid score are
    dropped.
    """
    weights = default_weights(fundamentals, factors) if weights is None else weights
    factors = build_features(prices, fundamentals=fundamentals, factors=factors)
    composite = sc.composite_score(factors, weights)
    if asof is None:
        asof = composite.index[-1]

    row = pd.DataFrame({name: f.loc[asof] for name, f in factors.items()})
    row["score"] = composite.loc[asof]
    row = row.dropna(subset=["score"]).sort_values("score", ascending=False)
    row.insert(0, "rank", range(1, len(row) + 1))
    return row


def default_weights(fundamentals: Panel | None = None,
                    factors: list[str] | None = None) -> dict[str, float]:
    """Defaults for the selected price factors plus the fundamental panels supplied."""
    w = {f: DEFAULT_WEIGHTS.get(f, EXTRA_WEIGHTS.get(f, 0.5)) for f in (factors or PRICE_FACTORS)}
    for name in fundamentals or {}:
        w[name] = FUNDAMENTAL_WEIGHTS.get(name, 0.5)
    return w


def make_signal_fn(weights: dict[str, float] | None = None,
                   fundamentals: Panel | None = None,
                   factors: list[str] | None = None) -> Callable[[pd.DataFrame], pd.Series]:
    """Build a ``signal_fn(price_window) -> score per ticker`` for the backtest.

    ``price_window`` is the price history *up to and including* the rebalance
    date. The returned score is the composite at the last row of that window,
    so the signal can only ever use past data. Fundamental panels are
    point-in-time per date, so reading their row at the window's last date is
    equally past-only.
    """
    weights = default_weights(fundamentals, factors) if weights is None else weights
    selected = factors

    def signal_fn(price_window: pd.DataFrame) -> pd.Series:
        feats = latest_features(price_window, fundamentals=fundamentals, factors=selected)
        return sc.composite_score(feats, weights).iloc[-1]

    return signal_fn


def mask_to_members(signal_fn: Callable[[pd.DataFrame], pd.Series],
                    members: pd.DataFrame) -> Callable[[pd.DataFrame], pd.Series]:
    """Only names that are index members on the signal date are eligible.

    ``members`` is a ``(dates x tickers)`` bool frame (``universe.membership``).
    Factors are still computed over every priced name, so cross-sectional
    z-scores are unchanged; the mask only decides who can be bought.
    """
    def masked(price_window: pd.DataFrame) -> pd.Series:
        scores = signal_fn(price_window)
        known = members.loc[:price_window.index[-1]]
        if known.empty:
            return scores.where(pd.Series(False, index=scores.index))
        row = known.iloc[-1].reindex(scores.index, fill_value=False).astype(bool)
        return scores.where(row)

    return masked


def make_bayes_signal_fn(prior_var: float = 1.0, horizon: int = 21,
                         shrink: bool = False, min_obs: int = 250,
                         fundamentals: Panel | None = None,
                         factors: list[str] | None = None,
                         ) -> Callable[[pd.DataFrame], pd.Series]:
    """Build a *walk-forward* Bayesian ``signal_fn`` for the backtest.

    At each rebalance the backtest passes ``window = prices.loc[:r]`` -- the
    price history available as of ``r``. We refit the factor-weight posterior on
    that window only and score the latest row with the learned weights, so the
    weights used at ``r`` depend on nothing after ``r``.

    Lookahead is avoided on two fronts:

    * Training targets: ``bayes.fit_weights`` regresses each factor row on its
      ``horizon``-period *forward* return. Inside the window, the last
      ``horizon`` rows have an unrealized (NaN) forward return and are dropped
      by its ``inner``-join + ``dropna``. Every training row therefore uses only
      data realized on or before ``r``.
    * Scoring: the features at the latest row are built from the window only.

    Until ``min_obs`` realized observations exist the signal is all-NaN, so the
    backtest skips the rebalance (it cannot fit weights on an empty/degenerate
    sample during warm-up).

    Speed: the fit is identical to ``bayes.fit_weights`` on the window, but kept
    as running sufficient statistics, so each call only absorbs the dates whose
    forward return newly realized (see ``_WalkForwardBayes``).
    """
    return _WalkForwardBayes(prior_var, horizon, shrink, min_obs, fundamentals, factors)


class _WalkForwardBayes:
    """Stateful walk-forward Bayes signal.

    Training date ``d`` contributes once ``d + horizon`` is inside the window, and
    its contribution depends only on prices up to ``d + horizon``. So when a new
    window *extends* the previous one (same columns, same index and values on the
    overlap) the old statistics stay valid and only new dates are added. Any other
    window (shorter, reordered, or with changed history) resets the state, so a
    reused instance can never train on rows from beyond the current window.
    """

    def __init__(self, prior_var: float, horizon: int, shrink: bool, min_obs: int,
                 fundamentals: Panel | None = None, factors: list[str] | None = None):
        self.prior_var, self.horizon = prior_var, horizon
        self.shrink, self.min_obs = shrink, min_obs
        self.fundamentals = fundamentals         # fixed for the instance's lifetime
        self.factors = factors
        self._warmup = warmup_rows(factors)
        self._reset()

    def _reset(self) -> None:
        self._prev: tuple | None = None          # (index, columns, values) of last window
        self._done = 0                           # training dates [0, _done) absorbed
        self._stats: bayes.DesignStats | None = None

    def _extends(self, window: pd.DataFrame, values: np.ndarray) -> bool:
        if self._prev is None:
            return False
        index, columns, prev_values = self._prev
        n = len(index)
        return (len(window) >= n and window.columns.equals(columns)
                and window.index[:n].equals(index)
                and np.array_equal(values[:n], prev_values, equal_nan=True))

    def _absorb(self, window: pd.DataFrame, a: int, b: int) -> None:
        """Add training dates ``a .. b-1`` (forward returns realized in ``window``)."""
        lo = max(0, a - self._warmup + 1)
        factors = {k: f.iloc[a - lo:b - lo]
                   for k, f in build_features(window.iloc[lo:b], fundamentals=self.fundamentals,
                                              factors=self.factors).items()}
        prices = window.iloc[a:b + self.horizon]
        stats = bayes.design_stats(factors, prices, self.horizon)
        self._stats = stats if self._stats is None else self._stats + stats

    def __call__(self, price_window: pd.DataFrame) -> pd.Series:
        values = price_window.to_numpy(dtype=float, copy=True)
        if not self._extends(price_window, values):
            self._reset()
        upto = max(len(price_window) - self.horizon, 0)
        if upto > self._done:
            self._absorb(price_window, self._done, upto)
            self._done = upto
        self._prev = (price_window.index, price_window.columns, values)

        nan = pd.Series(float("nan"), index=price_window.columns)
        if self._stats is None or self._stats.n < self.min_obs:
            return nan                           # warm-up: too little history to trust
        post = bayes.posterior_from_stats(self._stats, self.prior_var)
        weights = post.shrunk_weights() if self.shrink else post.weights
        feats = latest_features(price_window, fundamentals=self.fundamentals,
                                factors=self.factors)
        return sc.composite_score(feats, weights).iloc[-1]
