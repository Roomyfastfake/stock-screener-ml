"""Long-only top-N monthly-rebalance backtest. Strictly no lookahead.

At each signal date ``r`` the signal is computed from ``prices.loc[:r]``
(past only). The resulting top-N equal-weight book enters on the next available
close and exits on the next signal's following available close, so signals that
use date ``r``'s close never assume a fill at that same close.

The baseline here is frictionless; transaction costs, turnover reporting and
walk-forward weight fitting are layered on in roadmap step 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    period_returns: pd.Series              # net portfolio return per holding period
    equity_curve: pd.Series                # compounded growth of 1 unit
    holdings: dict = field(default_factory=dict)   # entry date -> list[ticker]
    stats: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.stats
        return (f"BacktestResult(periods={len(self.period_returns)}, "
                f"total_return={s.get('total_return', float('nan')):.2%}, "
                f"sharpe={s.get('sharpe', float('nan')):.2f})")


def rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading date within each calendar month."""
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).last())


def _summary_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    if len(returns) == 0:
        return {"total_return": np.nan, "ann_return": np.nan, "ann_vol": np.nan,
                "sharpe": np.nan, "max_drawdown": np.nan}
    equity = (1.0 + returns).cumprod()
    total = equity.iloc[-1] - 1.0
    mean, std = returns.mean(), returns.std(ddof=1)
    ann_return = (1.0 + mean) ** periods_per_year - 1.0
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = np.nan if std == 0 or np.isnan(std) else mean / std * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1.0
    return {"total_return": float(total), "ann_return": float(ann_return),
            "ann_vol": float(ann_vol), "sharpe": float(sharpe),
            "max_drawdown": float(drawdown.min())}


def backtest(prices: pd.DataFrame, signal_fn: SignalFn, top_n: int = 10,
             min_history: int = 126) -> BacktestResult:
    """Run the long-only top-N monthly backtest.

    Parameters
    ----------
    prices : (dates x tickers) close prices.
    signal_fn : maps a price *window* (past only) to a per-ticker score; higher
        is more attractive. NaN scores are ineligible.
    top_n : number of equally weighted names held each period.
    min_history : rows required before the first rebalance (factor warm-up).
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex")
    prices = prices.sort_index()
    dates = rebalance_dates(prices.index)
    dates = [d for d in dates if prices.index.get_loc(d) >= min_history - 1]

    holdings: dict = {}
    period_returns: list[float] = []
    period_index: list[pd.Timestamp] = []

    for r, nxt in zip(dates[:-1], dates[1:]):
        r_loc = prices.index.get_loc(r)
        nxt_loc = prices.index.get_loc(nxt)
        if r_loc + 1 >= len(prices.index) or nxt_loc + 1 >= len(prices.index):
            continue
        entry = prices.index[r_loc + 1]
        exit_ = prices.index[nxt_loc + 1]

        window = prices.loc[:r]                       # signal known after r close
        scores = signal_fn(window).dropna()
        if scores.empty:
            continue
        picks = list(scores.sort_values(ascending=False).head(top_n).index)
        holdings[entry] = picks

        held = prices.loc[[entry, exit_], picks]
        fwd = held.iloc[1] / held.iloc[0] - 1.0       # realized after entry
        period_returns.append(float(fwd.mean()))
        period_index.append(exit_)

    returns = pd.Series(period_returns, index=pd.DatetimeIndex(period_index), name="return")
    equity = (1.0 + returns).cumprod()
    equity.name = "equity"
    return BacktestResult(returns, equity, holdings, _summary_stats(returns))
