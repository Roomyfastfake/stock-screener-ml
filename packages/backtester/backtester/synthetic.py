"""Deterministic synthetic prices for tests and demos (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(n_tickers: int = 20, n_days: int = 756, seed: int = 0,
                drift_spread: float = 0.0005, vol: float = 0.015,
                start: str = "2018-01-01") -> pd.DataFrame:
    """GBM-ish business-day closes with persistent per-ticker drift."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    drift = np.linspace(-drift_spread, drift_spread, n_tickers)
    rng.shuffle(drift)
    rets = drift + vol * rng.standard_normal((n_days, n_tickers))
    rets[0] = 0.0
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)


def momentum_signal(lookback: int = 126, skip: int = 21):
    """Past-only momentum score: return from t-lookback to t-skip."""
    def signal_fn(window: pd.DataFrame) -> pd.Series:
        if len(window) <= lookback:
            return pd.Series(np.nan, index=window.columns)
        return window.iloc[-1 - skip] / window.iloc[-1 - lookback] - 1.0
    return signal_fn
