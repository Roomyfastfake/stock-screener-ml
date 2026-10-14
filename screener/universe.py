"""Ticker universes loaded from local CSV files.

``data/sp500.csv`` is a committed snapshot of *current* S&P 500 members
(columns ``ticker,name,sector,cik``). Backtesting history on today's members has
survivorship bias: names that left the index (often after falling) are missing.
Treat results on this universe as optimistic until point-in-time membership exists.

``refresh_sp500`` rebuilds the snapshot from Wikipedia (network + lxml).
"""
from __future__ import annotations

import datetime as _dt
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
SP500_CSV = DATA_DIR / "sp500.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def to_yahoo(ticker: str) -> str:
    """Exchange symbol -> Yahoo symbol (class shares use '-': BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def load_universe(path: str | pathlib.Path, sector: str | None = None) -> list[str]:
    """Tickers from a CSV with a ``ticker`` column, Yahoo-normalised, de-duplicated.

    ``sector`` filters on the optional ``sector`` column (case-insensitive).
    """
    df = pd.read_csv(path, comment="#")
    if "ticker" not in df.columns:
        raise ValueError(f"{path} needs a 'ticker' column")
    if sector is not None:
        if "sector" not in df.columns:
            raise ValueError(f"{path} has no 'sector' column to filter on")
        df = df[df["sector"].str.lower() == sector.lower()]
    tickers = [to_yahoo(t) for t in df["ticker"].dropna().astype(str)]
    return list(dict.fromkeys(t for t in tickers if t))


def load_sp500(sector: str | None = None) -> list[str]:
    return load_universe(SP500_CSV, sector=sector)


def cik_map(path: str | pathlib.Path = SP500_CSV) -> dict[str, int]:
    """Yahoo ticker -> SEC CIK from a universe CSV's optional ``cik`` column."""
    df = pd.read_csv(path, comment="#")
    if "cik" not in df.columns:
        return {}
    df = df.dropna(subset=["cik"])
    return {to_yahoo(t): int(c) for t, c in zip(df["ticker"].astype(str), df["cik"])}


def resolve(name_or_path: str, sector: str | None = None) -> list[str]:
    """``"sp500"`` -> the bundled snapshot; anything else is a CSV path."""
    if name_or_path.lower() == "sp500":
        return load_sp500(sector=sector)
    return load_universe(name_or_path, sector=sector)


def refresh_sp500(path: str | pathlib.Path = SP500_CSV) -> pd.DataFrame:
    """Download current constituents from Wikipedia and write the snapshot CSV."""
    import io
    import urllib.request

    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "stock-screener-ml"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed URL
        html = resp.read().decode("utf-8")
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    out = pd.DataFrame({"ticker": table["Symbol"].astype(str).str.strip(),
                        "name": table["Security"].astype(str).str.strip(),
                        "sector": table["GICS Sector"].astype(str).str.strip(),
                        "cik": pd.to_numeric(table["CIK"], errors="coerce").astype("Int64")})
    out = out.sort_values("ticker").reset_index(drop=True)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# S&P 500 constituents snapshot from Wikipedia, as of {_dt.date.today()}\n")
        out.to_csv(fh, index=False, lineterminator="\n")
    return out
