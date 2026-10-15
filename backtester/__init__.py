"""Reusable cross-sectional backtesting engine (pandas/numpy only)."""
from .engine import BacktestResult, SignalFn, backtest
from .report import tearsheet
from .research import (
                       bootstrap_sharpe_ci,
                       deflated_sharpe,
                       expected_max_sharpe,
                       probabilistic_sharpe,
                       walk_forward,
                       walk_forward_splits,
)
from .schedule import rebalance_dates
from .stats import benchmark_stats, perf_stats

__all__ = [
    "BacktestResult", "SignalFn", "backtest", "rebalance_dates",
    "perf_stats", "benchmark_stats", "walk_forward", "walk_forward_splits",
    "bootstrap_sharpe_ci", "probabilistic_sharpe", "expected_max_sharpe",
    "deflated_sharpe", "tearsheet",
]
