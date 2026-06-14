"""Cross-sectional scoring: winsorize -> z-score -> weighted composite.

All operations are *cross-sectional*: for each date (row) we standardize
across the universe (columns). This is pure and lookahead-free as long as the
input factors are themselves lookahead-free (see ``indicators``); standardizing
within a single date never peeks into the future.

Factor frames are shaped ``(dates x tickers)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(frame: pd.DataFrame, limits: tuple[float, float] = (0.05, 0.05)) -> pd.DataFrame:
    """Clip each row to its [lower, 1-upper] cross-sectional quantiles.

    Tames outliers before standardizing so a single extreme name cannot
    dominate the z-score.
    """
    lo, hi = limits
    if not (0.0 <= lo < 1.0 and 0.0 <= hi < 1.0 and lo + hi < 1.0):
        raise ValueError("invalid winsor limits")
    lower = frame.quantile(lo, axis=1)
    upper = frame.quantile(1.0 - hi, axis=1)
    return frame.clip(lower=lower, upper=upper, axis=0)


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row to mean 0, std 1 across the universe.

    Rows with fewer than two valid observations (or zero dispersion) become 0.
    """
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1, ddof=0)
    z = frame.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    return z


def zscore_factor(frame: pd.DataFrame, limits: tuple[float, float] = (0.05, 0.05)) -> pd.DataFrame:
    """Winsorize then cross-sectionally z-score a raw factor."""
    return cross_sectional_zscore(winsorize(frame, limits))


def composite_score(factors: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted sum of z-scored factor frames.

    ``factors`` maps name -> already-z-scored ``(dates x tickers)`` frame.
    ``weights`` maps name -> scalar weight. Missing factor values are treated
    as 0 (neutral) so a name is not dropped just because one factor is NaN,
    but a name with *no* valid factors stays NaN.
    """
    if not factors:
        raise ValueError("no factors provided")
    names = list(factors)
    index = factors[names[0]].index
    columns = factors[names[0]].columns

    total = pd.DataFrame(0.0, index=index, columns=columns)
    valid = pd.DataFrame(False, index=index, columns=columns)
    for name in names:
        w = float(weights.get(name, 0.0))
        f = factors[name].reindex(index=index, columns=columns)
        total = total.add(f.fillna(0.0) * w, fill_value=0.0)
        valid = valid | f.notna()
    return total.where(valid)
