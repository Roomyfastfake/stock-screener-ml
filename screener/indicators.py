"""Pure pandas/numpy technical indicators.

Invariants
----------
* No lookahead: every value at time ``t`` depends only on data up to and
  including ``t``. We only ever use ``.shift(k>=0)``, ``.rolling``, ``.ewm``
  and ``.pct_change`` (which all look backward).
* Pure functions: no I/O, no global state, no network.

Inputs are price ``DataFrame``s shaped ``(dates x tickers)`` of (adjusted)
close prices, or a single price ``Series`` indexed by date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(close: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple one-period returns. First row is NaN (no prior price)."""
    return close.pct_change()


def momentum(close: pd.DataFrame | pd.Series, lookback: int = 126,
             skip: int = 21) -> pd.DataFrame | pd.Series:
    """Total return over ``lookback`` periods, skipping the most recent ``skip``.

    The skip avoids contaminating medium-term momentum with short-term
    reversal. ``mom_t = close_{t-skip} / close_{t-lookback} - 1``.
    """
    if skip < 0 or lookback <= skip:
        raise ValueError("require 0 <= skip < lookback")
    return close.shift(skip) / close.shift(lookback) - 1.0


def reversal(close: pd.DataFrame | pd.Series, window: int = 5) -> pd.DataFrame | pd.Series:
    """Short-term reversal factor: negative recent return.

    Recent losers tend to bounce, so we flip the sign: a large negative
    ``window``-period return yields a large positive factor value.
    """
    return -(close / close.shift(window) - 1.0)


def moving_average(close: pd.DataFrame | pd.Series, window: int = 50) -> pd.DataFrame | pd.Series:
    """Trailing simple moving average."""
    return close.rolling(window, min_periods=window).mean()


def trend(close: pd.DataFrame | pd.Series, window: int = 50) -> pd.DataFrame | pd.Series:
    """Price relative to its trailing moving average: ``close/MA - 1``."""
    return close / moving_average(close, window) - 1.0


def volatility(close: pd.DataFrame | pd.Series, window: int = 21) -> pd.DataFrame | pd.Series:
    """Trailing realized volatility of daily returns (std, not annualized)."""
    return daily_returns(close).rolling(window, min_periods=window).std()


def low_volatility(close: pd.DataFrame | pd.Series, window: int = 21) -> pd.DataFrame | pd.Series:
    """Low-volatility factor: negative trailing vol (higher = calmer name)."""
    return -volatility(close, window)


def rsi(close: pd.DataFrame | pd.Series, window: int = 14) -> pd.DataFrame | pd.Series:
    """Wilder's Relative Strength Index in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 (only gains), RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())
