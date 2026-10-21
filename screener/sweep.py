"""Variant grid judged on active return vs a benchmark, with the deflated Sharpe.

Every variant starts on the same date (the longest factor warm-up in the grid) and
is compared with the benchmark marked on the same rebalance dates. The deflated
Sharpe of the best variant's active return accounts for how many variants were
tried, which is the honest way to report "the best of N".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import backtest as bt
from . import research, screen
from . import score as sc
from .report import benchmark_curve

FACTOR_SETS: dict[str, list[str] | None] = {
    "default": None, "momentum": ["momentum"], "mom_12_1": ["mom_12_1"],
    "high_52w": ["high_52w"], "trend": ["trend"], "low_vol": ["low_vol"],
    "reversal": ["reversal"], "mom_12_1+reversal": ["mom_12_1", "reversal"],
}
TOP_N = (10, 25, 50)


@dataclass
class SweepResult:
    table: pd.DataFrame          # one row per variant, sorted by information ratio
    best: str
    psr: float                   # P[true active return > 0], best variant alone
    dsr: float                   # same, deflated for the number of trials
    n_trials: int
    benchmark_stats: dict


def run(prices: pd.DataFrame, benchmark: pd.Series, members: pd.DataFrame | None = None,
        factor_sets: dict | None = None, top_ns=TOP_N, cost_bps: float = 10.0,
        progress=None) -> SweepResult:
    factor_sets = FACTOR_SETS if factor_sets is None else factor_sets
    warm = max(screen.warmup_rows(f) for f in factor_sets.values())
    rows, active, bench_stats = {}, {}, {}
    for name, factors in factor_sets.items():
        comp = sc.composite_score(screen.build_features(prices, factors=factors),
                                  screen.default_weights(factors=factors))
        if members is not None:
            comp = comp.where(members.reindex(index=comp.index, columns=comp.columns,
                                              fill_value=False))
        sfn = screen.make_panel_signal_fn(comp)
        for n in top_ns:
            res = bt.backtest(prices, sfn, top_n=n, cost_bps=cost_bps, min_history=warm)
            curve, bench_stats = benchmark_curve(benchmark, res)
            key = f"{name}, top {n}"
            act = res.period_returns - curve.pct_change().dropna().reindex(res.period_returns.index)
            active[key] = act
            ir = act.mean() / act.std(ddof=1) * 12 ** 0.5
            rows[key] = {**{k: res.stats[k] for k in ("cagr", "ann_vol", "sharpe", "max_drawdown")},
                         "excess_vs_benchmark": res.stats["cagr"] - bench_stats["cagr"],
                         "info_ratio": ir}
            if progress:
                progress(key, rows[key])
    table = pd.DataFrame(rows).T.sort_values("info_ratio", ascending=False)
    best = table.index[0]
    trial_ir = [a.mean() / a.std(ddof=1) for a in active.values()]
    return SweepResult(table, best, research.probabilistic_sharpe(active[best], 0.0),
                       research.deflated_sharpe(active[best], trial_ir), len(trial_ir), bench_stats)
