"""Factor research: does each factor actually predict returns?

All statistics use the backtest's timing: a factor observed at the close of ``t``
is paired with the return from the close of ``t+1`` to ``t+1+h`` (enter next
close), so research and backtest can never disagree about what was knowable.
Dates are sampled every ``step`` rows (default = ``h``) so forward windows do not
overlap and the t-stats are not inflated by autocorrelation.

* Rank IC: Spearman correlation across tickers between factor and forward return,
  per date. Reported as mean, std, IR (mean / std), t-stat, hit rate.
* IC decay: mean IC at several horizons.
* Quantile returns: mean forward return per factor quantile, and the top-minus-
  bottom spread with its t-stat.
* Factor correlation: average cross-sectional rank correlation between factors
  (redundant factors double-count the same bet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame:
    """Return from close ``t+lag`` to ``t+lag+horizon``, indexed at ``t``."""
    return prices.shift(-(lag + horizon)) / prices.shift(-lag) - 1.0


def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int) -> pd.Series:
    """Pearson correlation per row over columns where both are present."""
    a, b = a.align(b, join="inner")
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    both = ~(np.isnan(av) | np.isnan(bv))
    n = both.sum(axis=1)
    av, bv = np.where(both, av, 0.0), np.where(both, bv, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = av.sum(1) / n, bv.sum(1) / n
        da, db = np.where(both, av - ma[:, None], 0.0), np.where(both, bv - mb[:, None], 0.0)
        corr = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
    corr[n < min_names] = np.nan
    return pd.Series(corr, index=a.index)


def _masked_rank(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return frame.where(mask).rank(axis=1)


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, dates: pd.Index,
            min_names: int = 10) -> pd.Series:
    """Spearman IC per date: ranks are taken over names valid in *both* inputs."""
    f, r = factor.reindex(dates), fwd.reindex(index=dates, columns=factor.columns)
    mask = f.notna() & r.notna()
    return _rowwise_corr(_masked_rank(f, mask), _masked_rank(r, mask), min_names).dropna()


def summarize_ic(ic: pd.Series) -> dict:
    n = len(ic)
    mean, std = (float(ic.mean()), float(ic.std(ddof=1))) if n > 1 else (np.nan, np.nan)
    return {"ic_mean": mean, "ic_std": std,
            "ic_ir": mean / std if std else np.nan,
            "t_stat": mean / std * np.sqrt(n) if std else np.nan,
            "hit_rate": float((ic > 0).mean()) if n else np.nan, "n_dates": n}


def sample_dates(index: pd.DatetimeIndex, horizon: int, step: int | None = None,
                 lag: int = 1) -> pd.DatetimeIndex:
    """Every ``step`` rows (default ``horizon``) whose forward window is complete."""
    step = horizon if step is None else step
    last = len(index) - 1 - lag - horizon
    return index[np.arange(0, max(last + 1, 0), step)]


def ic_table(factors: dict[str, pd.DataFrame], prices: pd.DataFrame, horizon: int = 21,
             step: int | None = None) -> pd.DataFrame:
    fwd = forward_returns(prices, horizon)
    dates = sample_dates(prices.index, horizon, step)
    rows = {name: summarize_ic(rank_ic(f, fwd, dates)) for name, f in factors.items()}
    return pd.DataFrame(rows).T


def ic_decay(factors: dict[str, pd.DataFrame], prices: pd.DataFrame,
             horizons=(1, 5, 21, 63)) -> pd.DataFrame:
    out = {}
    for h in horizons:
        fwd = forward_returns(prices, h)
        dates = sample_dates(prices.index, h)
        out[f"{h}d"] = {name: float(rank_ic(f, fwd, dates).mean()) for name, f in factors.items()}
    return pd.DataFrame(out)


def quantile_returns(factor: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21,
                     n_quantiles: int = 5, step: int | None = None) -> dict:
    """Mean forward return per quantile (1 = lowest factor) and the top-bottom spread."""
    fwd = forward_returns(prices, horizon)
    dates = sample_dates(prices.index, horizon, step)
    f, r = factor.reindex(dates), fwd.reindex(index=dates, columns=factor.columns)
    mask = f.notna() & r.notna()
    pct = f.where(mask).rank(axis=1, pct=True)
    bucket = np.ceil(pct * n_quantiles).clip(1, n_quantiles)
    per_date = pd.DataFrame({q: r.where(bucket == q).mean(axis=1)
                             for q in range(1, n_quantiles + 1)}).dropna()
    spread = per_date[n_quantiles] - per_date[1]
    n = len(spread)
    t = float(spread.mean() / spread.std(ddof=1) * np.sqrt(n)) if n > 1 else np.nan
    return {"quantile_mean": per_date.mean(), "spread_mean": float(spread.mean()),
            "spread_t": t, "n_dates": n}


def factor_correlation(factors: dict[str, pd.DataFrame], dates: pd.Index) -> pd.DataFrame:
    """Average cross-sectional rank correlation between each pair of factors."""
    names = list(factors)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = factors[a].reindex(dates), factors[b].reindex(dates)
            mask = fa.notna() & fb.notna()
            c = float(_rowwise_corr(_masked_rank(fa, mask), _masked_rank(fb, mask), 10).mean())
            out.loc[a, b] = out.loc[b, a] = c
    return out


def report(factors: dict[str, pd.DataFrame], prices: pd.DataFrame, horizon: int = 21) -> dict:
    """Everything above in one dict (used by the CLI ``analyze`` command)."""
    dates = sample_dates(prices.index, horizon)
    quant = {name: quantile_returns(f, prices, horizon) for name, f in factors.items()}
    qtable = pd.DataFrame({n: q["quantile_mean"] for n, q in quant.items()}).T
    qtable.columns = [f"Q{c}" for c in qtable.columns]
    qtable["Q5-Q1"] = [quant[n]["spread_mean"] for n in qtable.index]
    qtable["spread_t"] = [quant[n]["spread_t"] for n in qtable.index]
    return {"ic": ic_table(factors, prices, horizon), "decay": ic_decay(factors, prices),
            "quantiles": qtable, "correlation": factor_correlation(factors, dates)}
