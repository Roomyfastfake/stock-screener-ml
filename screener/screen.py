"""Build the factor table, rank the universe, and expose a backtest signal.

The feature table is a dict of z-scored ``(dates x tickers)`` frames. The same
``build_features`` is reused by ``rank_universe`` (point-in-time ranking for the
CLI) and by ``make_signal_fn`` (per-rebalance scoring for the backtest), so the
research and backtest paths can never diverge.
"""
from __future__ import annotations

from typing import Callable

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


def build_features(prices: pd.DataFrame,
                   limits: tuple[float, float] = (0.05, 0.05)) -> dict[str, pd.DataFrame]:
    """Compute z-scored factor frames from a ``(dates x tickers)`` price panel.

    Each factor is lookahead-free (built only from ``shift``/``rolling``) and
    then winsorized + cross-sectionally z-scored.
    """
    raw = {
        "momentum": ind.momentum(prices),
        "trend": ind.trend(prices),
        "low_vol": ind.low_volatility(prices),
        "reversal": ind.reversal(prices),
    }
    return {name: sc.zscore_factor(frame, limits) for name, frame in raw.items()}


def rank_universe(prices: pd.DataFrame,
                  weights: dict[str, float] | None = None,
                  asof=None) -> pd.DataFrame:
    """Rank the universe by composite score at a single date (default: last).

    Returns a table indexed by ticker with each factor's z-score, the composite
    score, and the rank (1 = most attractive). Names with no valid score are
    dropped.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    factors = build_features(prices)
    composite = sc.composite_score(factors, weights)
    if asof is None:
        asof = composite.index[-1]

    row = pd.DataFrame({name: f.loc[asof] for name, f in factors.items()})
    row["score"] = composite.loc[asof]
    row = row.dropna(subset=["score"]).sort_values("score", ascending=False)
    row.insert(0, "rank", range(1, len(row) + 1))
    return row


def make_signal_fn(weights: dict[str, float] | None = None) -> Callable[[pd.DataFrame], pd.Series]:
    """Build a ``signal_fn(price_window) -> score per ticker`` for the backtest.

    ``price_window`` is the price history *up to and including* the rebalance
    date. The returned score is the composite at the last row of that window,
    so the signal can only ever use past data.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights

    def signal_fn(price_window: pd.DataFrame) -> pd.Series:
        factors = build_features(price_window)
        composite = sc.composite_score(factors, weights)
        return composite.iloc[-1]

    return signal_fn


def make_bayes_signal_fn(prior_var: float = 1.0, horizon: int = 21,
                         shrink: bool = False, min_obs: int = 250,
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
    """

    def signal_fn(price_window: pd.DataFrame) -> pd.Series:
        factors = build_features(price_window)
        try:
            post = bayes.fit_weights(factors, price_window, horizon=horizon,
                                     prior_var=prior_var)
        except ValueError:                       # no realized training rows yet
            return pd.Series(float("nan"), index=price_window.columns)
        if post.n_obs < min_obs:                 # too little history to trust
            return pd.Series(float("nan"), index=price_window.columns)
        weights = post.shrunk_weights() if shrink else post.weights
        return sc.composite_score(factors, weights).iloc[-1]

    return signal_fn
