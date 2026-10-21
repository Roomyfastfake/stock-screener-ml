"""Cross-sectional portfolio backtest with daily marking. Strictly no lookahead.

Timing (same as stock-screener-ml's backtest):
  * Rebalance signal dates ``r`` are the last trading day of each period (D/W/M/Q).
  * ``signal_fn(prices.loc[:r])`` scores tickers using the past only.
  * The new book trades at the NEXT close ``e`` (entry) and is held until the
    next entry. The final book is held to the last row. A rebalance whose entry
    would be the last row is skipped (no holding period to earn anything).

Book:
  * Long leg: top ``top_n`` scores, weights sum to +1.
  * Short leg (``short_n > 0``): bottom ``short_n`` scores not in the long leg,
    weights sum to -1. So long-short is gross 2, net 0; remaining NAV is cash at 0%.
  * Weights are fractions of NAV and drift daily:
    ``w' = w * (1 + r) / (1 + w.r - fee)``, valid for long, short and cash.

Costs:
  * At each entry close the drifted book is traded to target. Traded notional
    ``sum |target - drifted|`` is charged ``cost_bps + slippage_bps`` per unit
    (per side), multiplicatively: NAV *= 1 - traded * rate. One-way turnover is
    ``0.5 * traded``. Costs are attributed to the period that starts at that entry.
  * ``borrow_bps`` (annual) accrues daily on short gross exposure.
  * ``cost_model="additive"`` reproduces stock-screener-ml exactly: the book
    drifts as if costs were never paid and each period's net return is
    ``gross - cost - fees`` (simple subtraction). Daily nets subtract the cost on
    the entry day, so compounded equity differs from compounded period nets by
    second-order terms. Use it for parity checks; "nav" is the realistic default.

Missing data: prices are forward-filled, so a NaN close means "no trade, return
0" for a held name. A name is eligible at ``r`` only if it has a price at ``r``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import stats as st
from . import weights as wts
from .schedule import rebalance_dates

SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    returns: pd.Series                 # daily net returns, first entry date onward
    gross_returns: pd.Series           # daily returns before trading costs and borrow
    equity: pd.Series                  # compounded growth of 1 unit (net)
    period_returns: pd.Series          # net return per holding period, indexed by exit date
    period_gross: pd.Series            # gross return per holding period
    turnover: pd.Series                # one-way turnover, indexed by entry date
    costs: pd.Series                   # fraction of NAV paid in trading costs, by entry date
    weights: dict = field(default_factory=dict)    # entry date -> target weights
    signal_dates: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def holdings(self) -> dict:
        """Entry date -> list of tickers held (screener-compatible view)."""
        return {d: list(w.index) for d, w in self.weights.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.stats
        return (f"BacktestResult(days={len(self.returns)}, "
                f"rebalances={len(self.turnover)}, "
                f"cagr={s.get('cagr', float('nan')):.2%}, "
                f"sharpe={s.get('sharpe', float('nan')):.2f})")


def _target(scores: pd.Series, window: pd.DataFrame, top_n: int, short_n: int,
            weight_fn: wts.WeightFn) -> pd.Series:
    ranked = scores.sort_values(ascending=False, kind="mergesort")
    longs = ranked.head(top_n)
    parts = [weight_fn(longs, window)]
    shorts = ranked.drop(longs.index).tail(short_n) if short_n > 0 else ranked.iloc[:0]
    if not shorts.empty:                                    # too few names -> long-only book
        parts.append(-weight_fn(-shorts, window))
    return pd.concat(parts)


def backtest(prices: pd.DataFrame, signal_fn: SignalFn, top_n: int = 10,
             short_n: int = 0, weighting: str | wts.WeightFn = "equal",
             freq: str = "M", min_history: int = 126, cost_bps: float = 0.0,
             slippage_bps: float = 0.0, borrow_bps: float = 0.0,
             cost_model: str = "nav", benchmark: pd.Series | None = None,
             start: pd.Timestamp | str | None = None) -> BacktestResult:
    """Run the backtest.

    Parameters
    ----------
    prices : (dates x tickers) close prices, DatetimeIndex.
    signal_fn : past-only price window -> per-ticker score (higher = long). NaN
        scores are ineligible. Same interface as stock-screener-ml.
    top_n, short_n : names in the long and short legs.
    weighting : "equal", "score", "inverse_vol", or ``f(scores, window) -> weights``.
    freq : rebalance frequency, one of D, W, M, Q.
    min_history : rows required up to and including the first signal date.
    cost_bps, slippage_bps : per unit of notional traded (per side).
    borrow_bps : annual fee on short gross exposure, accrued daily (/252).
    cost_model : "nav" (default) or "additive" (stock-screener-ml parity), see module doc.
    benchmark : optional benchmark price series for relative stats.
    start : no signal dates before this (walk-forward test windows).
    """
    if min(cost_bps, slippage_bps, borrow_bps) < 0:
        raise ValueError("cost_bps, slippage_bps and borrow_bps must be non-negative")
    if top_n < 1 or short_n < 0:
        raise ValueError("need top_n >= 1 and short_n >= 0")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex")
    prices = prices.sort_index()
    index, cols = prices.index, prices.columns
    weight_fn = wts.resolve(weighting)
    if cost_model not in ("nav", "additive"):
        raise ValueError('cost_model must be "nav" or "additive"')
    nav_mode = cost_model == "nav"
    trade_rate = (cost_bps + slippage_bps) / 1e4
    borrow_rate = borrow_bps / 1e4 / st.TRADING_DAYS
    last = len(index) - 1

    # ---- 1. signals and target books (depend only on windows, not on P&L) ----
    targets: dict[int, np.ndarray] = {}
    target_series: dict[pd.Timestamp, pd.Series] = {}
    signal_dates: list[pd.Timestamp] = []
    start_ts = pd.Timestamp(start) if start is not None else None
    for r in rebalance_dates(index, freq):
        loc = index.get_loc(r)
        if loc < min_history - 1 or loc + 1 >= last:
            continue
        if start_ts is not None and r < start_ts:
            continue
        window = prices.iloc[: loc + 1]                     # known after r close
        scores = signal_fn(window)
        scores = scores[scores.notna() & window.iloc[-1].reindex(scores.index).notna()]
        if scores.empty:
            continue                                        # keep drifting book
        tgt = _target(scores, window, top_n, short_n, weight_fn)
        signal_dates.append(r)
        target_series[index[loc + 1]] = tgt
        targets[loc + 1] = tgt.reindex(cols, fill_value=0.0).to_numpy(dtype=float)

    # ---- 2. daily marking ----
    ff = prices.ffill().to_numpy(dtype=float)
    rets = np.zeros_like(ff)
    with np.errstate(invalid="ignore", divide="ignore"):
        rets[1:] = ff[1:] / ff[:-1] - 1.0
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    entries = sorted(targets)
    first = entries[0] if entries else len(index)
    w = np.zeros(len(cols))
    nav = gnav = 1.0
    day_net, day_gross = [], []
    per_idx, per_net, per_gross = [], [], []
    to_idx, turnover, costs = [], [], []
    base = gbase = None
    drag = 0.0                                              # additive: cost + fees this period

    def close_period(i: int) -> None:
        per_idx.append(index[i])
        per_gross.append(gnav / gbase - 1.0)
        per_net.append(nav / base - 1.0 if nav_mode else per_gross[-1] - drag)

    for i in range(first, len(index)):
        r = rets[i]
        gross = float(w @ r)
        fee = borrow_rate * float(-w[w < 0].sum())
        growth = 1.0 + gross - fee
        denom = growth if nav_mode else 1.0 + gross        # additive: book drifts cost-free
        if denom > 0:
            w = w * (1.0 + r) / denom
        prev = nav                                          # NAV at yesterday's close
        nav *= growth
        gnav *= 1.0 + gross
        drag += fee
        day_g = gross
        if i in targets:
            if base is not None:
                close_period(i)
            tgt = targets[i]
            traded = float(np.abs(tgt - w).sum())
            cost = traded * trade_rate
            base, gbase, drag = nav, gnav, cost
            growth = growth * (1.0 - cost) if nav_mode else growth - cost
            nav = prev * growth
            w = tgt.copy()
            to_idx.append(index[i])
            turnover.append(0.5 * traded)
            costs.append(cost)
        day_net.append(growth - 1.0)
        day_gross.append(day_g)
    if base is not None and first < last:                   # final book held to end
        close_period(last)

    didx = index[first:]
    returns = pd.Series(day_net, index=didx, name="return", dtype=float)
    gross_s = pd.Series(day_gross, index=didx, name="gross_return", dtype=float)
    equity = (1.0 + returns).cumprod().rename("equity")
    pidx = pd.DatetimeIndex(per_idx)
    tidx = pd.DatetimeIndex(to_idx)
    turnover_s = pd.Series(turnover, index=tidx, name="turnover", dtype=float)

    stats = st.perf_stats(returns)
    n_years = len(returns) / st.TRADING_DAYS
    stats["avg_turnover"] = float(turnover_s.mean()) if len(turnover_s) else np.nan
    stats["turnover_ann"] = float(turnover_s.sum() / n_years) if n_years else np.nan
    stats["cost_drag_ann"] = (float((gross_s - returns).mean() * st.TRADING_DAYS)
                              if len(returns) else np.nan)
    if benchmark is not None:
        bpx = benchmark.sort_index().ffill()
        bret = (bpx / bpx.shift(1) - 1.0).reindex(didx)
        stats.update(st.benchmark_stats(returns, bret))

    return BacktestResult(
        returns=returns, gross_returns=gross_s, equity=equity,
        period_returns=pd.Series(per_net, index=pidx, name="return", dtype=float),
        period_gross=pd.Series(per_gross, index=pidx, name="gross_return", dtype=float),
        turnover=turnover_s,
        costs=pd.Series(costs, index=tidx, name="cost", dtype=float),
        weights=target_series, signal_dates=signal_dates, stats=stats)
