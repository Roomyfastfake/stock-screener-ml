"""Long-only top-N monthly-rebalance backtest. Strictly no lookahead.

At each signal date ``r`` the signal is computed from ``prices.loc[:r]``
(past only). The resulting top-N equal-weight book enters on the next available
close and exits on the next signal's following available close, so signals that
use date ``r``'s close never assume a fill at that same close.

Costs: at each rebalance the book's drifted weights are traded to the new
equal-weight targets. Traded notional is ``sum |w_target - w_drifted|`` and is
charged ``cost_bps`` (per unit traded, i.e. per side) against that period's
return. Turnover is reported one-way, ``0.5 * traded``; the first period
buys the full book (turnover 0.5, traded 1.0).
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
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # one-way, per period
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

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
             min_history: int = 126, cost_bps: float = 0.0) -> BacktestResult:
    """Run the long-only top-N monthly backtest.

    Parameters
    ----------
    prices : (dates x tickers) close prices.
    signal_fn : maps a price *window* (past only) to a per-ticker score; higher
        is more attractive. NaN scores are ineligible.
    top_n : number of equally weighted names held each period.
    min_history : rows required before the first rebalance (factor warm-up).
    cost_bps : cost per unit of notional traded, in basis points (per side).
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex")
    prices = prices.sort_index()
    dates = rebalance_dates(prices.index)
    dates = [d for d in dates if prices.index.get_loc(d) >= min_history - 1]

    holdings: dict = {}
    period_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    period_index: list[pd.Timestamp] = []
    drifted = pd.Series(dtype=float)               # weights at end of last period
    cost_rate = cost_bps / 1e4

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

        target = pd.Series(1.0 / len(picks), index=picks)
        traded = float(target.sub(drifted, fill_value=0.0).abs().sum())
        gross = float(fwd.mean())                     # equal weight; NaN legs skipped
        net = gross - traded * cost_rate

        grown = target * (1.0 + fwd.fillna(0.0))
        drifted = grown / grown.sum()

        gross_returns.append(gross)
        period_returns.append(net)
        turnovers.append(0.5 * traded)
        period_index.append(exit_)

    idx = pd.DatetimeIndex(period_index)
    returns = pd.Series(period_returns, index=idx, name="return")
    gross_s = pd.Series(gross_returns, index=idx, name="gross_return")
    turnover = pd.Series(turnovers, index=idx, name="turnover")
    equity = (1.0 + returns).cumprod()
    equity.name = "equity"
    stats = _summary_stats(returns)
    stats["avg_turnover"] = float(turnover.mean()) if len(turnover) else np.nan
    stats["cost_drag_ann"] = (float((gross_s - returns).mean() * 12)
                              if len(returns) else np.nan)
    return BacktestResult(returns, equity, holdings, stats, turnover, gross_s)
