"""Rebalance calendars: last available trading date in each period."""
from __future__ import annotations

import pandas as pd

FREQS = {"D": None, "W": "W", "M": "M", "Q": "Q"}


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> list[pd.Timestamp]:
    """Last trading date of each calendar period.

    ``W`` groups by pandas weekly periods (Mon-Sun), which stay contiguous across
    year end, unlike (year, iso_week) keys. ``D`` returns every date.
    """
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {sorted(FREQS)}")
    if freq == "D":
        return list(index)
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period(FREQS[freq])).last())
