"""Self-contained HTML research report (inline SVG charts, no dependencies).

``build(...)`` takes already-computed results so it is pure and testable; the
CLI ``report`` command runs the backtests and research and calls it.
"""
from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

from . import backtest as bt

PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"]

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e2db;--card:#ffffff;
--c1:#2f6fbf;--c2:#d0702a;--c3:#5b8c3a;--c4:#8a5cb8;--c5:#777;--pos:47,111,191;--neg:208,112,42}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;
--muted:#9c9a92;--line:#33322f;--card:#1f1f1d;--c1:#6fa3e6;--c2:#ec9657;--c3:#8cc06a;--c4:#b993e0;
--c5:#aaa;--pos:111,163,230;--neg:236,150,87}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#9c9a92;--line:#33322f;--card:#1f1f1d;
--c1:#6fa3e6;--c2:#ec9657;--c3:#8cc06a;--c4:#b993e0;--c5:#aaa;--pos:111,163,230;--neg:236,150,87}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
.sub{color:var(--muted);margin:0 0 20px}
.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr.hl td{font-weight:650}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:13px;margin:4px 0 0}
.legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}
svg text{fill:var(--muted);font-size:11px}svg .grid{stroke:var(--line)}
.note{color:var(--muted);font-size:13px}
ul{padding-left:20px}
"""


def _fmt(v, kind="pct") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return {"pct": f"{v:.1%}", "num": f"{v:.2f}", "ic": f"{v:+.3f}", "int": f"{int(v)}"}[kind]


def _table(df: pd.DataFrame, kinds: dict[str, str], highlight: set | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, row in df.iterrows():
        cls = ' class="hl"' if highlight and idx in highlight else ""
        cells = "".join(f"<td>{_fmt(row[c], kinds.get(c, 'num'))}</td>" for c in df.columns)
        rows.append(f"<tr{cls}><td>{html.escape(str(idx))}</td>{cells}</tr>")
    return f'<div class="card"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


def _legend(names: list[str]) -> str:
    items = "".join(f'<span><i style="background:{PALETTE[i % len(PALETTE)]}"></i>'
                    f"{html.escape(n)}</span>" for i, n in enumerate(names))
    return f'<div class="legend">{items}</div>'


def line_chart(series: dict[str, pd.Series], log: bool = True, pct: bool = False,
               width: int = 860, height: int = 300) -> str:
    """Multi-series SVG line chart; x = dates, y = value (log scale if ``log``)."""
    pad_l, pad_r, pad_t, pad_b = 52, 12, 10, 26
    data = {k: s.dropna() for k, s in series.items() if len(s.dropna())}
    t0 = min(s.index[0] for s in data.values())
    t1 = max(s.index[-1] for s in data.values())
    lo = min(float(s.min()) for s in data.values())
    hi = max(float(s.max()) for s in data.values())
    tf = (lambda v: math.log(v)) if log else (lambda v: v)
    ylo, yhi = tf(lo), tf(hi)
    if yhi == ylo:
        yhi = ylo + 1.0
    span = (t1 - t0).total_seconds() or 1.0

    def px(t):
        return pad_l + (width - pad_l - pad_r) * (t - t0).total_seconds() / span

    def py(v):
        return pad_t + (height - pad_t - pad_b) * (1 - (tf(v) - ylo) / (yhi - ylo))

    parts = []
    if log:
        ticks = [2.0 ** k for k in range(math.floor(math.log2(lo)), math.ceil(math.log2(hi)) + 1)]
        ticks = [t for t in ticks if lo * 0.98 <= t <= hi * 1.02] or [lo, hi]
        label = lambda t: f"{t:g}x"
    else:
        step = (hi - lo) / 4 or 1.0
        ticks = [lo + i * step for i in range(5)]
        label = (lambda t: f"{t:.0%}") if pct else (lambda t: f"{t:.2f}")
    for t in ticks:
        y = py(t)
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{label(t)}</text>')
    for year in range(t0.year + 1, t1.year + 1):
        x = px(pd.Timestamp(year=year, month=1, day=1))
        if year % 2 == 0 or t1.year - t0.year <= 6:
            parts.append(f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle">{year}</text>')
    for i, (name, s) in enumerate(data.items()):
        pts = " ".join(f"{px(t):.1f},{py(float(v)):.1f}" for t, v in s.items())
        parts.append(f'<polyline fill="none" stroke="{PALETTE[i % len(PALETTE)]}" '
                     f'stroke-width="1.8" points="{pts}"/>')
    svg = (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="{html.escape(", ".join(data))}">{"".join(parts)}</svg>')
    return f'<div class="card">{svg}{_legend(list(data))}</div>'


def heatmap(corr: pd.DataFrame) -> str:
    names = list(corr.index)
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    rows = []
    for a in names:
        cells = []
        for b in names:
            v = float(corr.loc[a, b])
            rgb = "var(--pos)" if v >= 0 else "var(--neg)"
            alpha = 0.0 if a == b else min(abs(v), 1.0) * 0.75
            cells.append(f'<td style="background:rgba({rgb},{alpha:.2f})">{v:+.2f}</td>')
        rows.append(f"<tr><td>{html.escape(a)}</td>{''.join(cells)}</tr>")
    return f'<div class="card"><table><tr><th></th>{head}</tr>{"".join(rows)}</table></div>'


def benchmark_curve(bench: pd.Series, result: bt.BacktestResult) -> tuple[pd.Series, dict]:
    """Benchmark marked at the strategy's entry/exit dates, plus its stats."""
    dates = [min(result.holdings)] + list(result.period_returns.index)
    b = bench.reindex(pd.DatetimeIndex(dates), method="ffill")
    rets = b.pct_change().dropna()
    curve = (1.0 + rets).cumprod()
    return pd.concat([pd.Series([1.0], index=b.index[:1]), curve]), bt._summary_stats(rets)


def _curve(result: bt.BacktestResult) -> pd.Series:
    start = pd.Series([1.0], index=[min(result.holdings)])
    return pd.concat([start, result.equity_curve])


def build(title: str, subtitle: str, results: dict[str, bt.BacktestResult],
          benchmark: pd.Series, research: dict, primary: list[str],
          coverage: str = "", notes: list[str] | None = None) -> str:
    """Assemble the report. ``results`` maps label -> builtin BacktestResult."""
    first = results[primary[0]]
    bench_curve, bench_stats = benchmark_curve(benchmark, first)
    cols = {"cagr": "CAGR", "ann_vol": "Vol", "sharpe": "Sharpe", "max_drawdown": "Max DD",
            "avg_turnover": "Turnover"}
    stats = {k: r.stats for k, r in results.items()}
    stats["SPY buy and hold"] = bench_stats
    perf = pd.DataFrame({lab: {cols[c]: s.get(c, np.nan) for c in cols} for lab, s in stats.items()}).T
    perf.index.name = "Strategy"
    kinds = {"CAGR": "pct", "Vol": "pct", "Max DD": "pct", "Turnover": "pct", "Sharpe": "num"}

    curves = {k: _curve(results[k]) for k in primary}
    curves["SPY"] = bench_curve
    dd = {k: c / c.cummax() - 1.0 for k, c in curves.items()}

    best = max(primary, key=lambda k: results[k].stats.get("sharpe", -9))
    bs, ss = results[best].stats, bench_stats
    lead = (f"On point-in-time index membership the best variant, <b>{html.escape(best)}</b>, "
            f"returned {_fmt(bs['cagr'])} a year (Sharpe {_fmt(bs['sharpe'], 'num')}) against "
            f"{_fmt(ss['cagr'])} (Sharpe {_fmt(ss['sharpe'], 'num')}) for SPY. "
            f"{'It does not beat' if bs['sharpe'] < ss['sharpe'] else 'It beats'} the index "
            f"on a risk-adjusted basis.")

    ic = research["ic"][["ic_mean", "t_stat", "hit_rate", "n_dates"]].copy()
    ic.columns = ["Mean IC", "t-stat", "IC > 0", "Dates"]
    ic.index.name = "Factor"
    q = research["quantiles"].copy()
    q.index.name = "Factor"
    qk = {c: "pct" for c in q.columns}
    qk["spread_t"] = "num"
    q = q.rename(columns={"spread_t": "Spread t"})
    qk["Spread t"] = "num"
    decay = research["decay"].copy()
    decay.index.name = "Factor"

    body = [
        f"<h1>{html.escape(title)}</h1><p class='sub'>{html.escape(subtitle)}</p>",
        f"<p class='lead'>{lead}</p>",
        "<h2>Performance</h2>",
        _table(perf, kinds, highlight=set(primary)),
        "<p class='note'>Builtin engine: monthly rebalance, signal at the close, fill at the next "
        "close, 10 bps per side on traded notional. Stats from monthly periods; the benchmark is "
        "marked on the same dates.</p>",
        "<h2>Growth of $1 (log scale)</h2>", line_chart(curves, log=True),
        "<h2>Drawdown</h2>", line_chart(dd, log=False, pct=True, height=200),
        "<h2>Do the factors predict returns?</h2>",
        "<p>Rank IC is the cross-sectional Spearman correlation between a factor and the next "
        "21-day return (entered at the next close), on non-overlapping monthly dates. |t| above "
        "about 2 would be evidence of a real effect.</p>",
        _table(ic, {"Mean IC": "ic", "t-stat": "num", "IC > 0": "pct", "Dates": "int"}),
        "<h3>Mean 21-day return by factor quintile</h3>",
        _table(q, qk),
        "<h3>IC decay by horizon</h3>",
        _table(decay, {c: "ic" for c in decay.columns}),
        "<h3>Factor correlation</h3>",
        "<p class='note'>Average cross-sectional rank correlation. Highly correlated factors are "
        "the same bet counted twice.</p>",
        heatmap(research["correlation"]),
        "<h2>Method and caveats</h2><ul>",
        *[f"<li>{n}</li>" for n in (notes or [])],
        f"<li>{html.escape(coverage)}</li>" if coverage else "",
        "</ul>",
        "<p class='note'>Generated by stock-screener-ml. Research only, not investment advice.</p>",
    ]
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>")
