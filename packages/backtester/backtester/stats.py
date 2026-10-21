"""Performance and benchmark-relative statistics on a periodic return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def perf_stats(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    keys = ("total_return", "cagr", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "hit_rate")
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in keys}
    equity = (1.0 + r).cumprod()
    total = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (periods_per_year / len(r)) - 1.0
    mean, std = r.mean(), r.std(ddof=1)
    ann = np.sqrt(periods_per_year)
    sharpe = mean / std * ann if std > 0 else np.nan
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    sortino = mean / downside * ann if downside > 0 else np.nan
    mdd = float((equity / equity.cummax().clip(lower=1.0) - 1.0).min())
    calmar = cagr / -mdd if mdd < 0 else np.nan
    vals = (total, cagr, std * ann, sharpe, sortino, mdd, calmar, (r > 0).mean())
    return {k: float(v) for k, v in zip(keys, vals)}


def benchmark_stats(returns: pd.Series, bench: pd.Series,
                    periods_per_year: int = TRADING_DAYS) -> dict:
    """OLS beta/alpha of ``returns`` on ``bench`` plus active-return stats.

    alpha_ann is the per-period OLS intercept times ``periods_per_year``.
    """
    keys = ("beta", "alpha_ann", "correlation", "active_return_ann",
            "tracking_error", "info_ratio")
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if len(df) < 3:
        return {k: np.nan for k in keys}
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    var_x = x.var(ddof=1)
    beta = np.cov(y, x, ddof=1)[0, 1] / var_x if var_x > 0 else np.nan
    alpha = (y.mean() - beta * x.mean()) * periods_per_year
    corr = np.corrcoef(y, x)[0, 1] if y.std() > 0 and x.std() > 0 else np.nan
    active = y - x
    te = active.std(ddof=1) * np.sqrt(periods_per_year)
    act_ann = active.mean() * periods_per_year
    ir = act_ann / te if te > 1e-12 else np.nan
    return {k: float(v) for k, v in zip(keys, (beta, alpha, corr, act_ann, te, ir))}
