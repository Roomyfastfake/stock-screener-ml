"""Synthetic demo: momentum long-short, stats, research metrics, tearsheet.

    py -3 examples/demo.py [out.html]

Synthetic data only, so the numbers say nothing about real markets.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import backtester as bt
from backtester import synthetic

prices = synthetic.make_prices(n_tickers=40, n_days=1260, seed=0)
bench = prices.mean(axis=1)
sfn = synthetic.momentum_signal()

trial_sharpes = []
for w in ("equal", "score", "inverse_vol"):
    r = bt.backtest(prices, sfn, top_n=8, short_n=8, weighting=w, cost_bps=10,
                    slippage_bps=5, borrow_bps=50, benchmark=bench)
    trial_sharpes.append(r.returns.mean() / r.returns.std())
    print(f"{w:12s} {r}")

res = bt.backtest(prices, sfn, top_n=8, short_n=8, weighting="inverse_vol", cost_bps=10,
                  slippage_bps=5, borrow_bps=50, benchmark=bench)
ci = bt.bootstrap_sharpe_ci(res.returns)
dsr = bt.deflated_sharpe(res.returns, trial_sharpes)
print(f"Sharpe {ci['sharpe']:.2f}  95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}]  DSR {dsr:.3f}")

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tearsheet.html")
bt.tearsheet(res, out, title="Synthetic momentum L/S (inverse-vol)",
             notes={"sharpe 95% CI": f"[{ci['lo']:.2f}, {ci['hi']:.2f}]",
                    "deflated sharpe": f"{dsr:.3f}"})
print(f"wrote {out}")
