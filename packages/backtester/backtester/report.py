"""Self-contained HTML tearsheet (inline SVG, no JS, no external assets)."""
from __future__ import annotations

import html
import pathlib

import numpy as np
import pandas as pd

from .engine import BacktestResult

_PCT = {"total_return", "cagr", "ann_vol", "max_drawdown", "hit_rate", "avg_turnover",
        "turnover_ann", "cost_drag_ann", "alpha_ann", "active_return_ann", "tracking_error"}

_CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#6b7385;--line:#e3e6ec;--accent:#2f6fde;--neg:#c2410c;--pos:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--fg:#e6e9ef;--muted:#949bab;--line:#2a2f3a;--accent:#6ea0ff;--neg:#fb923c;--pos:#4ade80}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:960px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}
.sub{color:var(--muted);margin:0 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.k{border:1px solid var(--line);border-radius:6px;padding:8px 10px}.k b{display:block;font-size:17px}.k span{color:var(--muted);font-size:12px}
svg{width:100%;height:auto;display:block}.wrap{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:3px 7px;text-align:right;border-bottom:1px solid var(--line);font-size:12px}th{color:var(--muted);font-weight:500}
.n{color:var(--neg)}.p{color:var(--pos)}
"""


def _fmt(key: str, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.2%}" if key in _PCT else f"{v:.2f}"


def _line_svg(series: pd.Series, color: str, fill: bool = False,
              w: int = 900, h: int = 220, pad: int = 6) -> str:
    y = series.to_numpy(dtype=float)
    if len(y) < 2:
        return ""
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    lo, hi = (lo - 1e-9, hi + 1e-9) if hi == lo else (lo, hi)
    xs = np.linspace(pad, w - pad, len(y))
    ys = h - pad - (y - lo) / (hi - lo) * (h - 2 * pad)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    area = ""
    if fill:
        top = h - pad - (0 - lo) / (hi - lo) * (h - 2 * pad)
        area = (f'<polygon points="{xs[0]:.1f},{top:.1f} {pts} {xs[-1]:.1f},{top:.1f}" '
                f'fill="var({color})" opacity="0.18"/>')
    label = (f'<text x="{pad}" y="14" font-size="11" fill="var(--muted)">{hi:.3g}</text>'
             f'<text x="{pad}" y="{h - pad - 2}" font-size="11" fill="var(--muted)">{lo:.3g}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img">{area}'
            f'<polyline points="{pts}" fill="none" stroke="var({color})" stroke-width="1.5"/>'
            f"{label}</svg>")


def _monthly_table(returns: pd.Series) -> str:
    m = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    grid = m.unstack()
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    head = "".join(f"<th>{c}</th>" for c in
                   ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                    "Sep", "Oct", "Nov", "Dec", "Year"])
    rows = []
    for yr, row in grid.iterrows():
        cells = []
        for mo in range(1, 13):
            v = row.get(mo, np.nan)
            cells.append("<td></td>" if pd.isna(v) else
                         f'<td class="{"n" if v < 0 else "p"}">{v:.1%}</td>')
        y = yearly[yr]
        cells.append(f'<td class="{"n" if y < 0 else "p"}"><b>{y:.1%}</b></td>')
        rows.append(f"<tr><th>{yr}</th>{''.join(cells)}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def tearsheet(result: BacktestResult, path: str | pathlib.Path,
              title: str = "Backtest tearsheet", notes: dict | None = None) -> pathlib.Path:
    """Write an HTML tearsheet for ``result``. ``notes`` adds extra key/value rows
    (e.g. bootstrap CI or deflated Sharpe) to the headline grid."""
    r = result.returns
    dd = result.equity / result.equity.cummax().clip(lower=1.0) - 1.0
    stats = dict(result.stats)
    order = ["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate",
             "avg_turnover", "turnover_ann", "cost_drag_ann", "beta", "alpha_ann",
             "tracking_error", "info_ratio"]
    tiles = [(k, _fmt(k, stats[k])) for k in order if k in stats]
    tiles += [(k, v if isinstance(v, str) else _fmt(k, v)) for k, v in (notes or {}).items()]
    grid = "".join(f'<div class="k"><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>'
                   for k, v in tiles)
    span = f"{r.index[0]:%Y-%m-%d} to {r.index[-1]:%Y-%m-%d}" if len(r) else "empty"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">{span} &middot; {len(r)} days &middot; {len(result.turnover)} rebalances</p>
<div class="grid">{grid}</div>
<h2>Equity (net, growth of 1)</h2>{_line_svg(result.equity, "--accent")}
<h2>Drawdown</h2>{_line_svg(dd, "--neg", fill=True, h=140)}
<h2>Monthly returns</h2><div class="wrap">{_monthly_table(r) if len(r) else ""}</div>
</main></body></html>"""
    out = pathlib.Path(path)
    out.write_text(doc, encoding="utf-8")
    return out
