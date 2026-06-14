"""CLI for the stock screener.

Examples
--------
    # Offline demo on synthetic data (no network needed):
    py -3.14 main.py demo

    # Rank a live universe (requires yfinance):
    py -3.14 main.py rank --tickers AAPL MSFT NVDA --start 2020-01-01

    # Backtest with fixed, interpretable weights:
    py -3.14 main.py backtest --tickers AAPL MSFT NVDA --start 2018-01-01

    # Backtest with walk-forward Bayesian weights (refit each rebalance, no lookahead):
    py -3.14 main.py backtest --synthetic --bayes
"""
from __future__ import annotations

import argparse

import pandas as pd

from screener import backtest as bt
from screener import bayes
from screener import screen
from screener import synthetic


def _load_prices(args) -> pd.DataFrame:
    if getattr(args, "synthetic", False) or not getattr(args, "tickers", None):
        return synthetic.make_prices(n_tickers=20, n_days=900, seed=0)
    from screener import data  # lazy: only import (and need yfinance) on demand
    return data.download_prices(args.tickers, start=args.start, end=args.end)


def _weights(args, prices: pd.DataFrame) -> dict[str, float]:
    factors = screen.build_features(prices)
    if getattr(args, "bayes", False):
        scorer = bayes.BayesianScorer(prior_var=args.prior_var, horizon=args.horizon)
        scorer.fit(factors, prices)
        print("learned weights:", {k: round(float(v), 4) for k, v in scorer.weights.items()})
        print("signal/noise   :", {k: round(float(v), 2) for k, v in scorer.posterior.signal_to_noise.items()})
        return scorer.weights
    return screen.DEFAULT_WEIGHTS


def cmd_rank(args) -> None:
    prices = _load_prices(args)
    table = screen.rank_universe(prices, weights=_weights(args, prices))
    print(table.head(args.top_n).round(3).to_string())


def cmd_backtest(args) -> None:
    prices = _load_prices(args)
    if getattr(args, "bayes", False):
        # Walk-forward: weights are refit on each rebalance window (past only),
        # so this is genuinely out-of-sample -- no full-sample lookahead.
        signal_fn = screen.make_bayes_signal_fn(prior_var=args.prior_var,
                                                horizon=args.horizon)
    else:
        signal_fn = screen.make_signal_fn(weights=_weights(args, prices))
    result = bt.backtest(prices, signal_fn, top_n=args.top_n)
    print(result)
    for k, v in result.stats.items():
        print(f"  {k:>14}: {v: .4f}")


def cmd_demo(args) -> None:
    args.synthetic = True
    print("== Ranking (synthetic universe) ==")
    cmd_rank(args)
    print("\n== Backtest (synthetic universe) ==")
    cmd_backtest(args)


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they may appear *after* the
    # subcommand (e.g. `main.py backtest --bayes --top-n 5`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tickers", nargs="*", help="universe; omit for synthetic data")
    common.add_argument("--start", default="2018-01-01")
    common.add_argument("--end", default=None)
    common.add_argument("--synthetic", action="store_true", help="force offline synthetic data")
    common.add_argument("--top-n", dest="top_n", type=int, default=10)
    common.add_argument(
        "--bayes",
        action="store_true",
        help="use learned Bayesian weights (full-sample for rank; walk-forward, refit per rebalance, for backtest)",
    )
    common.add_argument("--prior-var", dest="prior_var", type=float, default=1.0)
    common.add_argument("--horizon", type=int, default=21)

    p = argparse.ArgumentParser(description="ML stock screener", parents=[common])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("rank", parents=[common]).set_defaults(func=cmd_rank)
    sub.add_parser("backtest", parents=[common]).set_defaults(func=cmd_backtest)
    sub.add_parser("demo", parents=[common]).set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
