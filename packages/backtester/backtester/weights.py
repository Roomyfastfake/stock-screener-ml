"""Weighting schemes for one leg of the book.

A scheme maps ``(scores, window)`` to non-negative weights over ``scores.index``
summing to 1. ``scores`` are already oriented so higher means "more of this leg"
(the engine negates scores for the short leg). ``window`` is the price history
up to and including the signal date, so schemes cannot see the future.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

WeightFn = Callable[[pd.Series, pd.DataFrame], pd.Series]


def _normalise(raw: pd.Series) -> pd.Series:
    raw = raw.clip(lower=0.0).fillna(0.0)
    total = raw.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(raw), index=raw.index)    # fallback: equal
    return raw / total


def equal(scores: pd.Series, window: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / len(scores), index=scores.index)


def score(scores: pd.Series, window: pd.DataFrame) -> pd.Series:
    """Proportional to the positive part of the score; equal if none positive."""
    return _normalise(scores.astype(float))


def make_inverse_vol(lookback: int = 63) -> WeightFn:
    """Weights proportional to 1 / stdev of the last ``lookback`` daily returns."""
    def inverse_vol(scores: pd.Series, window: pd.DataFrame) -> pd.Series:
        px = window[scores.index].ffill().iloc[-(lookback + 1):]
        vol = (px / px.shift(1) - 1.0).iloc[1:].std(ddof=1)
        inv = 1.0 / vol.where(vol > 0)
        return _normalise(inv)
    return inverse_vol


SCHEMES: dict[str, WeightFn] = {
    "equal": equal,
    "score": score,
    "inverse_vol": make_inverse_vol(),
}


def resolve(scheme: str | WeightFn) -> WeightFn:
    if callable(scheme):
        return scheme
    if scheme not in SCHEMES:
        raise ValueError(f"weighting must be callable or one of {sorted(SCHEMES)}")
    return SCHEMES[scheme]
