# backtester

A small, tested cross-sectional backtesting engine for equity strategies (pandas and numpy only). It lives inside [stock-screener-ml](https://github.com/Roomyfastfake/stock-screener-ml) as its evaluation layer, but has no dependency on it: any strategy that can score tickers from past prices plugs in.

## The contract

```python
signal_fn(prices_up_to_t: pd.DataFrame) -> pd.Series   # ticker -> score, higher = long, NaN = ineligible
```

The signal is computed after the close of `t` and traded at the next close. Tests poison every row after `t` and check that nothing moves.

## Features

* Rebalancing daily, weekly, monthly or quarterly
* Long-only or long-short (`top_n`, `short_n`), weighting `equal`, `score`, `inverse_vol` or any callable
* Costs: commission and slippage per unit traded on drifted-to-target weights, borrow fee on shorts, two cost models (`nav`, and `additive` for exact parity with the screener)
* Daily marking, so drawdowns inside a holding period are visible
* Stats: CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover, cost drag; vs a benchmark: beta, alpha, tracking error, information ratio
* Research: walk-forward splits and runner, stationary-bootstrap Sharpe confidence interval, probabilistic and deflated Sharpe (Bailey and Lopez de Prado 2014)
* Self-contained HTML tearsheet

## Usage

```python
import backtester as bt

res = bt.backtest(prices, signal_fn, top_n=25, short_n=0, weighting="equal",
                  freq="M", cost_bps=10, slippage_bps=5, benchmark=spy)
print(res.stats)
print(bt.bootstrap_sharpe_ci(res.returns))
bt.tearsheet(res, "tearsheet.html")
```

From the stock-screener-ml root:

```powershell
py -3 -m pip install -e .\packages\backtester
py -3 main.py backtest --universe sp500-pit --start 2015-01-01 --engine backtester `
    --cost-model nav --benchmark SPY --tearsheet reports\sp500_pit.html
```

## Tests

```powershell
py -3 packages\backtester\tests\test_backtester.py
```

Includes hand-checked timing fixtures shared with the screener and a live parity test that reproduces the screener's own backtest (`screener/backtest.py` in the enclosing repo, or `SCREENER_PATH`) to 1e-12.

Design decisions are recorded in [AGENTS.md](AGENTS.md).

## Disclaimer

For education and research. Not financial advice.
