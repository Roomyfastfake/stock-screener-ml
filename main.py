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

    # S&P 500 with the local price cache (first run downloads, reruns are offline):
    py -3.14 main.py fetch --universe sp500 --start 2015-01-01
    py -3.14 main.py backtest --universe sp500 --start 2015-01-01 --offline
"""
from __future__ import annotations

import argparse

import pandas as pd

from screener import backtest as bt
from screener import bayes
from screener import cache
from screener import screen
from screener import synthetic
from screener import universe


def _tickers(args) -> list[str]:
    if getattr(args, "universe", None):
        return universe.resolve(args.universe, sector=args.sector,
                                start=args.start, end=args.end)
    return [universe.to_yahoo(t) for t in (getattr(args, "tickers", None) or [])]


def _is_pit(args) -> bool:
    return (getattr(args, "universe", None) or "").lower() == "sp500-pit"


def _members(args, prices: pd.DataFrame) -> pd.DataFrame | None:
    """Point-in-time membership mask for ``--universe sp500-pit``, else None."""
    if not _is_pit(args):
        return None
    members = universe.membership(prices.index).reindex(columns=prices.columns,
                                                        fill_value=False)
    n_mem = universe.membership(prices.index).sum(axis=1)
    priced = (members & prices.notna()).sum(axis=1)
    cov = (priced / n_mem).resample("YE").mean()
    print("members with price data: " + ", ".join(
        f"{d.year} {v:.0%}" for d, v in cov.items()))
    return members


def _load_prices(args) -> pd.DataFrame:
    tickers = _tickers(args)
    if getattr(args, "synthetic", False) or not tickers:
        return synthetic.make_prices(n_tickers=20, n_days=900, seed=0)
    if args.no_cache:
        from screener import data  # lazy: only import (and need yfinance) on demand
        return data.download_prices(tickers, start=args.start, end=args.end)
    pc = cache.PriceCache(args.cache_dir, max_age_days=args.max_age_days)
    prices = pc.get(tickers, start=args.start, end=args.end,
                    offline=args.offline, refresh=args.refresh)
    if pc.last_missing:
        print(f"no data for {len(pc.last_missing)} tickers: {pc.last_missing[:10]}")
    print(f"prices: {prices.shape[1]} tickers x {prices.shape[0]} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    return prices


def _fundamentals(args, prices: pd.DataFrame):
    """Point-in-time fundamental panels for ``--fundamentals``, else None."""
    if not getattr(args, "fundamentals", False):
        return None
    from screener import fundamentals as fu
    ciks = universe.cik_map()
    fc = fu.FundamentalsCache(args.sec_cache_dir)
    panel, unmapped = fu.load_panel(list(prices.columns), prices.index, ciks, cache=fc,
                                    offline=args.offline, refresh=args.refresh)
    if unmapped:
        print(f"fundamentals: no CIK for {len(unmapped)} tickers: {unmapped[:10]}")
    for d in (prices.index[len(prices) // 4], prices.index[len(prices) // 2], prices.index[-1]):
        cov = {k: f"{v:.0%}" for k, v in fu.coverage(panel, d).items()}
        print(f"fundamentals coverage {d.date()}: {cov}")
    return panel


def _weights(args, prices: pd.DataFrame, fund=None) -> dict[str, float]:
    if getattr(args, "bayes", False):
        factors = screen.build_features(prices, fundamentals=fund, factors=args.factors)
        scorer = bayes.BayesianScorer(prior_var=args.prior_var, horizon=args.horizon)
        scorer.fit(factors, prices)
        print("learned weights:", {k: round(float(v), 4) for k, v in scorer.weights.items()})
        print("signal/noise   :", {k: round(float(v), 2) for k, v in scorer.posterior.signal_to_noise.items()})
        return scorer.weights
    return screen.default_weights(fund, args.factors)


def cmd_rank(args) -> None:
    prices = _load_prices(args)
    fund = _fundamentals(args, prices)
    table = screen.rank_universe(prices, weights=_weights(args, prices, fund),
                                 fundamentals=fund, factors=args.factors)
    members = _members(args, prices)
    if members is not None:                       # only current members are investable
        table = table[members.iloc[-1].reindex(table.index, fill_value=False).astype(bool)]
        table["rank"] = range(1, len(table) + 1)
    print(table.head(args.top_n).round(3).to_string())


def cmd_backtest(args) -> None:
    prices = _load_prices(args)
    fund = _fundamentals(args, prices)
    if getattr(args, "bayes", False):
        # Walk-forward: weights are refit on each rebalance window (past only),
        # so this is genuinely out-of-sample -- no full-sample lookahead.
        signal_fn = screen.make_bayes_signal_fn(prior_var=args.prior_var,
                                                horizon=args.horizon, fundamentals=fund,
                                                factors=args.factors)
    else:
        signal_fn = screen.make_signal_fn(weights=_weights(args, prices, fund),
                                          fundamentals=fund, factors=args.factors)
    members = _members(args, prices)
    if members is not None:
        signal_fn = screen.mask_to_members(signal_fn, members)
    if getattr(args, "engine", "builtin") == "backtester":
        result = _run_external(args, prices, signal_fn)
    else:
        result = bt.backtest(prices, signal_fn, top_n=args.top_n, cost_bps=args.cost_bps)
    print(result)
    for k, v in result.stats.items():
        print(f"  {k:>14}: {v: .4f}")


def _run_external(args, prices: pd.DataFrame, signal_fn):
    """Route through the sibling `backtester` package (pip install -e ../backtester)."""
    try:
        import backtester
    except ImportError as exc:
        raise SystemExit("--engine backtester needs the package: "
                         "pip install -e ../backtester") from exc
    bench = None
    if args.benchmark:
        pc = cache.PriceCache(args.cache_dir, max_age_days=args.max_age_days)
        bench = pc.get([args.benchmark], start=args.start, end=args.end,
                       offline=args.offline)[args.benchmark]
    result = backtester.backtest(
        prices, signal_fn, top_n=args.top_n, short_n=args.short_n,
        weighting=args.weighting, freq=args.freq, cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps, cost_model=args.cost_model, benchmark=bench)
    if args.tearsheet:
        backtester.tearsheet(result, args.tearsheet)
        print(f"tearsheet written to {args.tearsheet}")
    return result


def cmd_fetch(args) -> None:
    if not _tickers(args):
        raise SystemExit("fetch needs --universe or --tickers")
    _load_prices(args)


def cmd_analyze(args) -> None:
    """Factor research: rank IC, IC decay, quintile spreads, factor correlation."""
    from screener import research
    prices = _load_prices(args)
    fund = _fundamentals(args, prices)
    factors = screen.build_features(prices, fundamentals=fund, factors=args.factors)
    members = _members(args, prices)
    if members is not None:                       # research only on investable names
        factors = {k: f.where(members) for k, f in factors.items()}
    rep = research.report(factors, prices, horizon=args.horizon)
    pd.set_option("display.width", 140)
    print(f"\n== Rank IC, {args.horizon}d forward return from next close "
          f"(non-overlapping dates) ==")
    print(rep["ic"].round(3).to_string())
    print("\n== Mean IC by horizon (decay) ==")
    print(rep["decay"].round(3).to_string())
    print(f"\n== Mean {args.horizon}d forward return by factor quintile ==")
    print((rep["quantiles"].drop(columns="spread_t") * 100).round(2).astype(str).add("%")
          .join(rep["quantiles"]["spread_t"].round(2)).to_string())
    print("\n== Average cross-sectional rank correlation between factors ==")
    print(rep["correlation"].round(2).to_string())
    if args.out:
        import pathlib
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in rep.items():
            table.to_csv(out / f"{name}.csv")
        print(f"\ntables written to {out}")


def cmd_check_fundamentals(args) -> None:
    """Fetch/cache SEC fundamentals for the universe and run the TTM validation gate."""
    from screener import fundamentals as fu
    tickers = _tickers(args)
    if not tickers:
        raise SystemExit("check-fundamentals needs --universe or --tickers")
    ciks = universe.cik_map()
    fc = fu.FundamentalsCache(args.sec_cache_dir)
    seen, tally = set(), {c: [0, 0, 0] for c in ("revenue", "net_income")}  # checked, passed, cos ok
    for i, t in enumerate(tickers, 1):
        cik = ciks.get(t)
        if cik is None or cik in seen:
            continue
        seen.add(cik)
        rows = fc.rows(cik, offline=args.offline, refresh=args.refresh)
        for c in tally:
            n, ok = fu.ttm_consistency(rows, c)
            tally[c][0] += n
            tally[c][1] += ok
            tally[c][2] += int(n > 0 and ok == n)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}")
    for c, (n, ok, cos) in tally.items():
        print(f"{c}: {ok}/{n} 10-K checks pass ({ok / max(n, 1):.1%}); "
              f"{cos}/{len(seen)} companies pass every check")


def cmd_refresh_universe(args) -> None:
    df = universe.refresh_sp500()
    print(f"wrote {len(df)} constituents to {universe.SP500_CSV}")


def cmd_demo(args) -> None:
    args.synthetic = True
    print("== Ranking (synthetic universe) ==")
    cmd_rank(args)
    print("\n== Backtest (synthetic universe) ==")
    cmd_backtest(args)


def _factor_list(text: str) -> list[str]:
    names = [f.strip() for f in text.split(",") if f.strip()]
    unknown = [f for f in names if f not in screen.FACTORS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown factors {unknown}; choose from {sorted(screen.FACTORS)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they may appear *after* the
    # subcommand (e.g. `main.py backtest --bayes --top-n 5`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tickers", nargs="*", help="explicit tickers; omit (and --universe) for synthetic data")
    common.add_argument("--universe", default=None,
                        help="'sp500' (bundled snapshot) or a CSV path with a 'ticker' column")
    common.add_argument("--sector", default=None, help="filter --universe by GICS sector")
    common.add_argument("--cache-dir", dest="cache_dir", default=str(cache.DEFAULT_DIR))
    common.add_argument("--max-age-days", dest="max_age_days", type=float, default=1.0,
                        help="re-download open-ended (no --end) histories older than this")
    common.add_argument("--offline", action="store_true", help="cache only, never download")
    common.add_argument("--refresh", action="store_true", help="re-download every ticker")
    common.add_argument("--no-cache", dest="no_cache", action="store_true",
                        help="download directly, bypassing the cache")
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
    common.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0,
                        help="backtest cost per unit traded, in bps per side (0 = frictionless)")
    common.add_argument("--engine", choices=["builtin", "backtester"], default="builtin",
                        help="builtin = screener/backtest.py; backtester = sibling package")
    # Options below only apply to --engine backtester.
    common.add_argument("--short-n", dest="short_n", type=int, default=0)
    common.add_argument("--weighting", choices=["equal", "score", "inverse_vol"], default="equal")
    common.add_argument("--freq", choices=["D", "W", "M", "Q"], default="M")
    common.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=0.0)
    common.add_argument("--cost-model", dest="cost_model", choices=["additive", "nav"],
                        default="additive", help="additive matches the builtin engine exactly")
    common.add_argument("--tearsheet", default=None, help="write an HTML tearsheet to this path")
    common.add_argument("--factors", type=_factor_list, default=None,
                        help="comma list of price factors (default momentum,trend,low_vol,"
                             "reversal; also mom_12_1, high_52w)")
    common.add_argument("--fundamentals", action="store_true",
                        help="add point-in-time SEC factors (roe, rev_growth); needs SEC_USER_AGENT "
                             "unless cached")
    common.add_argument("--sec-cache-dir", dest="sec_cache_dir",
                        default=str(universe.DATA_DIR / "cache" / "sec"))
    common.add_argument("--benchmark", default=None,
                        help="benchmark ticker for relative stats, e.g. SPY (cached like prices)")

    p = argparse.ArgumentParser(description="ML stock screener", parents=[common])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("rank", parents=[common]).set_defaults(func=cmd_rank)
    sub.add_parser("backtest", parents=[common]).set_defaults(func=cmd_backtest)
    sub.add_parser("demo", parents=[common]).set_defaults(func=cmd_demo)
    sub.add_parser("fetch", parents=[common],
                   help="warm the price cache for --universe/--tickers").set_defaults(func=cmd_fetch)
    analyze = sub.add_parser("analyze", parents=[common],
                             help="factor research: IC, decay, quintiles, correlation")
    analyze.add_argument("--out", default=None, help="directory for CSV tables")
    analyze.set_defaults(func=cmd_analyze)
    sub.add_parser("check-fundamentals", parents=[common],
                   help="fetch SEC fundamentals and run the TTM validation gate"
                   ).set_defaults(func=cmd_check_fundamentals)
    sub.add_parser("refresh-universe",
                   help="rebuild data/sp500.csv from Wikipedia").set_defaults(func=cmd_refresh_universe)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
