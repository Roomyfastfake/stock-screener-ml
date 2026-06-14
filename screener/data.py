"""The *only* networked module: download prices via yfinance.

Everything else in the package is pure and offline-testable. ``yfinance`` is
imported lazily so the rest of the package (and the test-suite) works without
it installed.
"""
from __future__ import annotations

import pandas as pd


def _require_yfinance():
    try:
        import yfinance as yf  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "yfinance is required for downloads. Install with `pip install yfinance`."
        ) from exc
    return yf


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    auto_adjust: bool = True) -> pd.DataFrame:
    """Download adjusted close prices as a ``(dates x tickers)`` frame.

    Network call. Columns are tickers; rows are dates. Tickers that return no
    data are dropped.
    """
    yf = _require_yfinance()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust,
                      progress=False)
    # yfinance returns a column MultiIndex (field, ticker) for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
        close = raw[field]
    else:  # single ticker
        col = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[col]]
        close.columns = [tickers[0] if isinstance(tickers, list) else tickers]
    return close.dropna(how="all").sort_index()
