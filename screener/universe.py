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

import numpy as np
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


def resolve(name_or_path: str, sector: str | None = None,
            start=None, end=None) -> list[str]:
    """``"sp500"`` -> the bundled snapshot; ``"sp500-pit"`` -> every name that was a
    member at any time (use with ``membership``); anything else is a CSV path."""
    key = name_or_path.lower()
    if key == "sp500":
        return load_sp500(sector=sector)
    if key == "sp500-pit":
        if sector is not None:
            raise ValueError("--sector is not supported for sp500-pit (no historical sectors)")
        return members_between(start, end)
    return load_universe(name_or_path, sector=sector)


def members_between(start=None, end=None) -> list[str]:
    """Tickers that were index members at any time in ``[start, end]``."""
    iv = membership_intervals()
    lo = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    hi = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    overlap = (iv["start"] <= hi) & (iv["end"].isna() | (iv["end"] > lo))
    return list(dict.fromkeys(iv.loc[overlap, "ticker"]))


# --------------------------------------------------------------------------- #
# point-in-time membership
# --------------------------------------------------------------------------- #
HISTORY_CSV = DATA_DIR / "sp500_history.csv"
HISTORY_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"


def membership_intervals(snapshot: str | pathlib.Path = SP500_CSV,
                         history: str | pathlib.Path = HISTORY_CSV) -> pd.DataFrame:
    """Membership as ``ticker, start, end`` intervals (``end`` NaT = still a member).

    * Current members come from the Wikipedia snapshot, starting at its per-company
      ``date_added``. That survives ticker renames (META's start is FB's 2013 add),
      which a per-ticker history cannot.
    * Former members are the history file's *closed* intervals.
    * Names the history still lists as open but the newer snapshot does not are
      treated as removed on the snapshot's most recent addition date (the history
      file lags the snapshot by a few weeks).
    """
    snap = pd.read_csv(snapshot, comment="#")
    hist = pd.read_csv(history, comment="#", parse_dates=["start_date", "end_date"])
    hist["ticker"] = hist["ticker"].map(to_yahoo)
    current = {to_yahoo(t): pd.Timestamp(d) for t, d in zip(snap["ticker"], snap["date_added"])}

    closed = hist[hist["end_date"].notna()]
    closed = closed.drop(index=_renamed_predecessors(hist, closed, current))
    dropped = hist[hist["end_date"].isna() & ~hist["ticker"].isin(current)].copy()
    newest = hist["start_date"].max()
    late_adds = [d for d in current.values() if d > newest]
    dropped["end_date"] = min(late_adds) if late_adds else pd.Timestamp.today().normalize()
    out = pd.concat([
        pd.DataFrame({"ticker": list(current), "start": list(current.values()), "end": pd.NaT}),
        closed.rename(columns={"start_date": "start", "end_date": "end"}),
        dropped.rename(columns={"start_date": "start", "end_date": "end"}),
    ], ignore_index=True)
    return out.sort_values(["ticker", "start"]).reset_index(drop=True)


def _renamed_predecessors(hist: pd.DataFrame, closed: pd.DataFrame,
                          current: dict[str, pd.Timestamp]) -> list:
    """Closed intervals that are an earlier *ticker* of a current member.

    If a current member's open interval in the per-ticker history starts after its
    per-company ``date_added``, it was renamed on that start date (FB -> META on
    2022-06-09). The closed interval ending that day under the old ticker is the
    same company, already covered from ``date_added``; prefer the one starting at
    ``date_added``, else the only one ending that day.
    """
    drop = []
    opens = hist[hist["end_date"].isna() & hist["ticker"].isin(current)]
    for t, start in zip(opens["ticker"], opens["start_date"]):
        added = current[t]
        if pd.isna(added) or start <= added + pd.Timedelta(days=7):
            continue
        cands = closed[(closed["end_date"] == start) & (closed["ticker"] != t)]
        match = cands[(cands["start_date"] - added).abs() <= pd.Timedelta(days=7)]
        if len(match) >= 1:
            drop.extend(match.index[:1])
        elif len(cands) == 1:
            drop.extend(cands.index)
    return drop


def membership(index: pd.DatetimeIndex, intervals: pd.DataFrame | None = None) -> pd.DataFrame:
    """``(dates x tickers)`` bool: was the name an index member on that date?

    A name is a member from its ``start`` date (inclusive) to its ``end`` date
    (exclusive): on the change date the replacement is already in.
    """
    iv = membership_intervals() if intervals is None else intervals
    tickers = list(dict.fromkeys(iv["ticker"]))
    out = np.zeros((len(index), len(tickers)), dtype=bool)
    col = {t: i for i, t in enumerate(tickers)}
    for t, s, e in zip(iv["ticker"], iv["start"], iv["end"]):
        lo = index.searchsorted(s, side="left")
        hi = len(index) if pd.isna(e) else index.searchsorted(e, side="left")
        out[lo:hi, col[t]] = True
    return pd.DataFrame(out, index=index, columns=tickers)


def refresh_history(path: str | pathlib.Path = HISTORY_CSV) -> int:
    """Re-download the membership history file (network)."""
    import datetime
    import urllib.request

    req = urllib.request.Request(HISTORY_URL, headers={"User-Agent": "stock-screener-ml"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed URL
        body = resp.read().decode("utf-8")
    header = (f"# S&P 500 membership intervals by ticker from github.com/fja05680/sp500 "
              f"(file sp500_ticker_start_end.csv, fetched {datetime.date.today()}). "
              f"MIT License, see data/LICENSE-sp500-history.txt\n")
    pathlib.Path(path).write_text(header + body.replace("\r\n", "\n"), "utf-8")
    return body.count("\n") - 1


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
                        "cik": pd.to_numeric(table["CIK"], errors="coerce").astype("Int64"),
                        "date_added": pd.to_datetime(table["Date added"], errors="coerce").dt.date})
    out = out.sort_values("ticker").reset_index(drop=True)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# S&P 500 constituents snapshot from Wikipedia, as of {_dt.date.today()}\n")
        out.to_csv(fh, index=False, lineterminator="\n")
    return out
