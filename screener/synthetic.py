"""Deterministic synthetic price panels for offline tests and demos.

No network, fully seeded. Two generators:

* ``make_prices`` -- a panel of names with *persistent* heterogeneous drift, so
  that momentum carries genuine information (past winners keep drifting up).
* ``make_factor_panel`` -- prices whose forward returns load on a *known*
  factor with a known sign, used to test the Bayesian weight estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bdays(n_days: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n_days)


def make_prices(n_tickers: int = 20, n_days: int = 900, seed: int = 0,
                drift_spread: float = 0.0015, vol: float = 0.01,
                start: str = "2015-01-01") -> pd.DataFrame:
    """Geometric random walk with a persistent per-ticker daily drift.

    Drift is fixed per ticker for the whole sample, so trailing momentum is
    predictive of forward returns -- a clean, lookahead-free signal for tests.
    """
    rng = np.random.default_rng(seed)
    idx = _bdays(n_days, start)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    drift = rng.normal(0.0, drift_spread, size=n_tickers)
    shocks = rng.normal(0.0, vol, size=(n_days, n_tickers))
    log_ret = drift[None, :] + shocks
    log_ret[0, :] = 0.0
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def make_factor_panel(n_tickers: int = 40, n_days: int = 600, seed: int = 1,
                      true_beta: float = 0.03, vol: float = 0.01,
                      start: str = "2015-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices whose next-day return loads on a known exposure with sign(true_beta).

    Returns ``(prices, exposure)`` where ``exposure`` is a ``(dates x tickers)``
    standardized factor and each name's next-day return is
    ``true_beta * exposure + noise``. A weight estimator should recover
    ``sign(true_beta)`` on this factor.
    """
    rng = np.random.default_rng(seed)
    idx = _bdays(n_days, start)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    # Standardized cross-sectional exposure, re-drawn each day (stationary factor).
    exp_raw = rng.normal(size=(n_days, n_tickers))
    exposure = pd.DataFrame(exp_raw, index=idx, columns=tickers)
    exposure = exposure.sub(exposure.mean(axis=1), axis=0).div(
        exposure.std(axis=1, ddof=0), axis=0)

    noise = rng.normal(0.0, vol, size=(n_days, n_tickers))
    # Return on day t is driven by the exposure observed at t-1 (no lookahead).
    ret = true_beta * exposure.shift(1).fillna(0.0).to_numpy() + noise
    ret[0, :] = 0.0
    prices = 100.0 * np.exp(np.cumsum(ret, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers), exposure
