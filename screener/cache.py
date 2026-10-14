"""Local parquet price cache in front of the downloader.

Layout under ``cache_dir`` (default ``data/cache``, git-ignored)::

    prices/<TICKER>.parquet   one adjusted-close series per ticker
    manifest.json             ticker -> {fetched_at, start, first, last}

A ticker is a hit when it was fetched from a ``start`` at or before the request
and is fresh enough: either its last *complete* session at fetch time (New York
close, see ``last_complete_session``) is on or after the requested ``end``, or
(open-ended requests) it was fetched within ``max_age_days``. A miss re-downloads the ticker's
*whole* history instead of appending, because adjusted closes are restated on
every split/dividend; splicing old and new downloads would mix adjustment bases.

``offline=True`` never touches the network and fails loudly on a miss.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Callable

import pandas as pd

from . import universe

DEFAULT_DIR = universe.DATA_DIR / "cache"
Downloader = Callable[[list[str], str, "str | None"], pd.DataFrame]


class CacheMiss(LookupError):
    """Requested tickers are not cached (raised in offline mode)."""


def _now() -> pd.Timestamp:
    return pd.Timestamp(_dt.datetime.now())


class PriceCache:
    def __init__(self, cache_dir: str | pathlib.Path = DEFAULT_DIR,
                 downloader: Downloader | None = None, max_age_days: float = 1.0,
                 chunk_size: int = 100):
        self.dir = pathlib.Path(cache_dir)
        self.prices_dir = self.dir / "prices"
        self.manifest_path = self.dir / "manifest.json"
        self._downloader = downloader
        self.max_age = pd.Timedelta(days=max_age_days)
        self.chunk_size = chunk_size
        self.manifest: dict = (json.loads(self.manifest_path.read_text("utf-8"))
                               if self.manifest_path.exists() else {})

    # ------------------------------------------------------------------ io
    def _path(self, ticker: str) -> pathlib.Path:
        return self.prices_dir / f"{ticker}.parquet"

    def _read(self, ticker: str) -> pd.Series:
        s = pd.read_parquet(self._path(ticker))[ticker]
        s.index = pd.DatetimeIndex(s.index)
        return s

    def _write(self, ticker: str, series: pd.Series, start: str,
               cutoff: pd.Timestamp) -> None:
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        series.rename(ticker).to_frame().to_parquet(self._path(ticker))
        self.manifest[ticker] = {"fetched_at": _now().isoformat(timespec="seconds"),
                                 "complete_through": str(cutoff.date()),
                                 "start": str(pd.Timestamp(start).date()),
                                 "first": str(series.index[0].date()),
                                 "last": str(series.index[-1].date())}

    def _save_manifest(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1, sort_keys=True), "utf-8")
        tmp.replace(self.manifest_path)

    # ------------------------------------------------------------ freshness
    def is_fresh(self, ticker: str, start: str, end: str | None) -> bool:
        meta = self.manifest.get(ticker)
        if meta is None or not self._path(ticker).exists():
            return False
        if pd.Timestamp(meta["start"]) > pd.Timestamp(start):
            return False
        if end is not None and "complete_through" in meta and \
                pd.Timestamp(meta["complete_through"]) >= pd.Timestamp(end).normalize():
            return True                      # every close up to `end` was final when fetched
        return _now() - pd.Timestamp(meta["fetched_at"]) <= self.max_age

    # ---------------------------------------------------------------- main
    def get(self, tickers: list[str], start: str, end: str | None = None,
            offline: bool = False, refresh: bool = False) -> pd.DataFrame:
        """``(dates x tickers)`` adjusted closes for ``[start, end]``.

        Tickers the downloader returns nothing for are dropped (and reported via
        ``self.last_missing``) rather than failing the whole universe.
        """
        tickers = list(dict.fromkeys(tickers))
        if offline:
            stale = [t for t in tickers if t not in self.manifest or not self._path(t).exists()
                     or pd.Timestamp(self.manifest[t]["start"]) > pd.Timestamp(start)]
            if stale:
                raise CacheMiss(f"{len(stale)} tickers not cached from {start}, e.g. {stale[:5]}")
        else:
            todo = tickers if refresh else [t for t in tickers if not self.is_fresh(t, start, end)]
            self._fetch(todo, start, end)

        cols = {t: self._read(t) for t in tickers if self._path(t).exists()}
        self.last_missing = [t for t in tickers if t not in cols]
        if not cols:
            return pd.DataFrame()
        joined = pd.concat(cols, axis=1).sort_index()
        # Concatenating per-ticker series yields one memory block per column, which
        # makes every later frame op loop column by column (~50x slower for 500
        # names). Rebuild as a single contiguous float block.
        frame = pd.DataFrame(joined.to_numpy(dtype=float), index=joined.index,
                             columns=joined.columns)
        frame = frame.loc[pd.Timestamp(start):]
        if end is not None:
            frame = frame.loc[:pd.Timestamp(end)]
        return frame.dropna(how="all")

    def _fetch(self, tickers: list[str], start: str, end: str | None) -> None:
        if not tickers:
            return
        download = self._downloader or _default_downloader
        # Always fetch through today so the cached history is reusable for any `end`.
        cutoff = last_complete_session(_now_et())
        for i in range(0, len(tickers), self.chunk_size):
            chunk = tickers[i:i + self.chunk_size]
            failed = self._store(download(chunk, start, None), chunk, start, cutoff)
            if failed:                       # one retry: yfinance drops names transiently
                self._store(download(failed, start, None), failed, start, cutoff)
            self._save_manifest()            # persist progress per chunk

    def _store(self, raw: pd.DataFrame, chunk: list[str], start: str,
               cutoff: pd.Timestamp) -> list[str]:
        failed = []
        for t in chunk:
            s = raw[t].dropna() if t in raw.columns else pd.Series(dtype=float)
            s = s.loc[:cutoff]
            if s.empty:
                failed.append(t)
            else:
                self._write(t, s, start, cutoff)
        return failed


MARKET_TZ = "America/New_York"
CLOSE_BUFFER = pd.Timedelta(hours=16, minutes=30)   # 16:00 close + time for final prints


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MARKET_TZ)


def last_complete_session(now_et: pd.Timestamp) -> pd.Timestamp:
    """Latest date whose close is final. Before ~16:30 ET today's bar is intraday."""
    today = now_et.tz_localize(None).normalize()
    if now_et.tz_localize(None) - today >= CLOSE_BUFFER:
        return today
    return today - pd.Timedelta(days=1)


def _default_downloader(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    from . import data  # lazy: yfinance only when we actually download
    return data.download_prices(tickers, start=start, end=end)
