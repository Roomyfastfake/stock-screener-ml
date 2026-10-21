"""Point-in-time fundamentals from SEC EDGAR XBRL ``companyfacts``.

Why SEC and not ``yfinance.Ticker.info``: ``info`` is today's snapshot, so using it
in a backtest leaks the future. Every companyfacts value carries the date it was
``filed``; a value is only used from the first trading day *after* that date.

Parsing rules (each has a test):

* Periods are identified by ``start``/``end``, never ``fy``/``fp`` (those describe
  the filing a fact came from, so comparatives carry the newer filing's labels).
* Flow concepts are classified by duration: quarter, 9 months, full year (with
  slack for 52/53-week years). Q4 is rarely reported, so it is ``FY - 9M``.
* The same period appears in several filings (original, comparatives,
  restatements). As of date ``t`` its value is the copy *filed latest on or before
  t*; ties go to the higher-priority tag. Tag fallbacks are merged per period, so a
  history that switches tag (e.g. revenue to ASC 606 in 2018) stays continuous.
* Snapshots are computed only at a company's filing dates and forward-filled; a
  snapshot whose period ended more than ``MAX_STALENESS`` ago is dropped.

Factors (higher = more attractive, z-scored downstream like the price factors):

* ``roe``: trailing-twelve-month net income / latest stockholders' equity
  (NaN when equity <= 0).
* ``rev_growth``: TTM revenue / TTM revenue one year earlier - 1 (NaN when the
  earlier TTM <= 0).

Network: SEC requires a ``User-Agent`` naming you and a contact email, read from
the ``SEC_USER_AGENT`` environment variable. Only extracted concepts are cached
(``data/cache/sec/<CIK>.parquet``), not the raw multi-MB JSON.
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from . import universe

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DEFAULT_DIR = universe.DATA_DIR / "cache" / "sec"
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-KT/A"}

# concept -> (us-gaap tags in priority order, unit, kind)
CONCEPTS: dict[str, tuple[list[str], str, str]] = {
    "revenue": (["Revenues",
                 "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax",
                 "SalesRevenueNet",
                 "SalesRevenueGoodsNet"], "USD", "flow"),
    "net_income": (["NetIncomeLoss",
                    "NetIncomeLossAvailableToCommonStockholdersBasic",
                    "ProfitLoss"], "USD", "flow"),
    "equity": (["StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
               "USD", "stock"),
}
FACTORS = ["roe", "rev_growth"]

QUARTER_DAYS = (75, 105)
NINE_MONTH_DAYS = (255, 290)
YEAR_DAYS = (350, 380)
TTM_WINDOW = pd.Timedelta(days=330)       # 4 quarter ends fit, a 5th does not
MAX_STALENESS = pd.Timedelta(days=456)    # ~15 months after the period end
COLUMNS = ["concept", "priority", "start", "end", "val", "filed", "form"]


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def extract(facts: dict) -> pd.DataFrame:
    """companyfacts JSON -> long table of the concepts we use (see ``COLUMNS``)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for concept, (tags, unit, kind) in CONCEPTS.items():
        for priority, tag in enumerate(tags):
            for f in gaap.get(tag, {}).get("units", {}).get(unit, []):
                if f.get("form") not in FORMS or "val" not in f:
                    continue
                if kind == "flow" and "start" not in f:
                    continue
                rows.append((concept, priority, f.get("start"), f["end"], float(f["val"]),
                             f["filed"], f["form"]))
    out = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("start", "end", "filed"):
        out[c] = pd.to_datetime(out[c])
    return out


def _latest_known(rows: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """One row per period: the latest copy filed on or before ``asof``."""
    known = rows[rows["filed"] <= asof]
    known = known.sort_values(["filed", "priority"], ascending=[True, False])
    return known.drop_duplicates(["concept", "start", "end"], keep="last")


def _duration_between(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    days = (df["end"] - df["start"]).dt.days
    return df[(days >= lo) & (days <= hi)]


def discrete_quarters(flow: pd.DataFrame) -> pd.Series:
    """Quarter values by period end, with Q4 derived as FY - 9M when not reported."""
    q = _duration_between(flow, *QUARTER_DAYS)
    quarters = q.groupby("end")["val"].last()
    nine = _duration_between(flow, *NINE_MONTH_DAYS).set_index("start")
    for _, fy in _duration_between(flow, *YEAR_DAYS).iterrows():
        if fy["end"] in quarters.index:
            continue
        ytd = nine.loc[[fy["start"]]] if fy["start"] in nine.index else nine.iloc[:0]
        ytd = ytd[ytd["end"] < fy["end"]].sort_values("end")
        if len(ytd):
            quarters.loc[fy["end"]] = fy["val"] - ytd["val"].iloc[-1]
    return quarters.sort_index()


def ttm(quarters: pd.Series, end: pd.Timestamp) -> float:
    """Sum of the 4 quarters ending at ``end``; NaN unless exactly 4 are known."""
    if end not in quarters.index:
        return np.nan
    window = quarters[(quarters.index > end - TTM_WINDOW) & (quarters.index <= end)]
    return float(window.sum()) if len(window) == 4 else np.nan


def _year_before(quarters: pd.Series, end: pd.Timestamp) -> pd.Timestamp | None:
    target = end - pd.Timedelta(days=364)
    near = quarters.index[abs(quarters.index - target) <= pd.Timedelta(days=20)]
    return None if len(near) == 0 else near[np.argmin(abs(near - target))]


def snapshot(rows: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Factor values knowable on ``asof`` (filings with ``filed <= asof``)."""
    out = {"period_end": pd.NaT, "roe": np.nan, "rev_growth": np.nan}
    known = _latest_known(rows, asof)
    by = {c: known[known["concept"] == c] for c in CONCEPTS}
    ni_q = discrete_quarters(by["net_income"])
    rev_q = discrete_quarters(by["revenue"])
    ends = ni_q.index.union(rev_q.index)
    if len(ends) == 0:
        return out
    end = ends[-1]
    out["period_end"] = end

    ni = ttm(ni_q, end)
    eq = by["equity"].set_index("end")["val"]
    eq = eq[~eq.index.duplicated(keep="last")]
    if end in eq.index and eq[end] > 0 and np.isfinite(ni):
        out["roe"] = ni / eq[end]

    prev = _year_before(rev_q, end)
    if prev is not None:
        now, before = ttm(rev_q, end), ttm(rev_q, prev)
        if np.isfinite(now) and np.isfinite(before) and before > 0:
            out["rev_growth"] = now / before - 1.0
    return out


def _in(days: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= days <= bounds[1]


def _quarters_from(periods: dict) -> dict:
    """Dict version of ``discrete_quarters``: {(start, end): val} -> {end: val}."""
    q, nine, years = {}, {}, []
    for (s, e), v in periods.items():
        d = (e - s).days
        if _in(d, QUARTER_DAYS):
            q[e] = v
        elif _in(d, NINE_MONTH_DAYS):
            nine.setdefault(s, []).append((e, v))
        elif _in(d, YEAR_DAYS):
            years.append((s, e, v))
    for s, e, v in years:
        if e not in q:
            ytd = [(ee, vv) for ee, vv in nine.get(s, []) if ee < e]
            if ytd:
                q[e] = v - max(ytd, key=lambda x: x[0])[1]
    return q


def _ttm_from(q: dict, end) -> float:
    if end not in q:
        return np.nan
    vals = [v for e, v in q.items() if end - TTM_WINDOW < e <= end]
    return float(sum(vals)) if len(vals) == 4 else np.nan


def _snapshot_from(state: dict) -> dict:
    out = {"period_end": pd.NaT, "roe": np.nan, "rev_growth": np.nan}
    ni_q, rev_q = _quarters_from(state["net_income"]), _quarters_from(state["revenue"])
    ends = set(ni_q) | set(rev_q)
    if not ends:
        return out
    end = max(ends)
    out["period_end"] = end
    ni = _ttm_from(ni_q, end)
    eq = {e: v for (_, e), v in state["equity"].items()}
    if end in eq and eq[end] > 0 and np.isfinite(ni):
        out["roe"] = ni / eq[end]
    target = end - pd.Timedelta(days=364)
    near = [e for e in rev_q if abs(e - target) <= pd.Timedelta(days=20)]
    if near:
        prev = min(near, key=lambda e: abs(e - target))
        now, before = _ttm_from(rev_q, end), _ttm_from(rev_q, prev)
        if np.isfinite(now) and np.isfinite(before) and before > 0:
            out["rev_growth"] = now / before - 1.0
    return out


def events(rows: pd.DataFrame) -> pd.DataFrame:
    """Snapshot at every distinct filing date: index ``filed``, factor columns.

    One pass in filing order keeping the latest value per period, so it is linear
    in the number of facts; equal to ``snapshot(rows, d)`` at each date ``d``.
    """
    cols = ["filed", "period_end", *FACTORS]
    if rows.empty:
        return pd.DataFrame(columns=cols).set_index("filed")
    r = rows.sort_values(["filed", "priority"], ascending=[True, False], kind="mergesort")
    state: dict = {c: {} for c in CONCEPTS}
    recs = []
    filed = r["filed"].to_numpy()
    keys = list(zip(r["concept"], r["start"], r["end"], r["val"]))
    for i, (concept, start, end, val) in enumerate(keys):
        state[concept][(start, end)] = val           # later filing / higher priority wins
        if i + 1 == len(keys) or filed[i + 1] != filed[i]:
            recs.append({"filed": pd.Timestamp(filed[i]), **_snapshot_from(state)})
    out = pd.DataFrame(recs, columns=cols).set_index("filed")
    out["period_end"] = pd.to_datetime(out["period_end"])
    return out.astype({f: float for f in FACTORS})


def to_daily(ev: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Events -> trading-day panel. A filing on day F is usable from the next day."""
    if ev.empty:
        return pd.DataFrame(np.nan, index=index, columns=FACTORS)
    pos = index.searchsorted(ev.index, side="right")          # first trading day > filed
    ok = pos < len(index)
    ev = ev[ok].copy()
    ev.index = index[pos[ok]]
    ev = ev[~ev.index.duplicated(keep="last")]                # same-day filings: last wins
    daily = ev.reindex(index, method="ffill")
    stale = (index.to_series() - daily["period_end"]) > MAX_STALENESS
    daily.loc[stale.to_numpy(), FACTORS] = np.nan
    return daily[FACTORS]


def build_panel(rows_by_ticker: dict[str, pd.DataFrame],
                index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Raw factor panels ``{factor: (dates x tickers)}`` aligned on ``index``."""
    cols = {t: to_daily(events(r), index) for t, r in rows_by_ticker.items() if not r.empty}
    panel = {}
    for f in FACTORS:
        frame = pd.DataFrame({t: d[f] for t, d in cols.items()}, index=index)
        panel[f] = pd.DataFrame(frame.to_numpy(dtype=float), index=index, columns=frame.columns)
    return panel


def ttm_consistency(rows: pd.DataFrame, concept: str, rtol: float = 0.005) -> tuple[int, int]:
    """Validation gate: at every 10-K, TTM rebuilt from quarters must equal the FY value.

    Returns ``(checked, passed)``. Q4 is ``FY - 9M`` when not reported, so this
    checks that the reported quarters sum to the 9M YTD (catches duplicated,
    mis-tagged or missing quarters), and checks reported Q4s directly.
    """
    flow = rows[rows["concept"] == concept]
    checked = passed = 0
    for filed in sorted(flow.loc[flow["form"].str.startswith("10-K"), "filed"].unique()):
        known = _latest_known(flow, filed)
        fy = _duration_between(known, *YEAR_DAYS)
        fy = fy[fy["filed"] == filed]
        if fy.empty:
            continue
        fy = fy.sort_values("end").iloc[-1]
        q = _duration_between(known, *QUARTER_DAYS).groupby("end")["val"].last()
        nine = _duration_between(known, *NINE_MONTH_DAYS)
        nine = nine[(nine["start"] == fy["start"]) & (nine["end"] < fy["end"])]
        if fy["end"] not in q.index and not nine.empty:
            q.loc[fy["end"]] = fy["val"] - nine.sort_values("end")["val"].iloc[-1]
            # Derived Q4 makes TTM == FY tautological unless Q1-Q3 match the 9M YTD.
            ytd = nine.sort_values("end").iloc[-1]
            got = q[(q.index > ytd["start"]) & (q.index <= ytd["end"])].sum()
            ok = np.isclose(got, ytd["val"], rtol=rtol)
        else:
            got = ttm(q.sort_index(), fy["end"])
            ok = np.isfinite(got) and np.isclose(got, fy["val"], rtol=rtol)
        checked += 1
        passed += bool(ok)
    return checked, passed


def coverage(panel: dict[str, pd.DataFrame], date) -> dict[str, float]:
    """Share of names with a value per factor at ``date``."""
    return {f: float(p.loc[date].notna().mean()) for f, p in panel.items()}


# --------------------------------------------------------------------------- #
# fetching + cache
# --------------------------------------------------------------------------- #
class SecClient:
    """Throttled companyfacts fetcher (SEC allows at most 10 requests/second)."""

    def __init__(self, user_agent: str | None = None, min_interval: float = 0.12,
                 retries: int = 4):
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT")
        if not self.user_agent:
            raise RuntimeError(
                "SEC requires a User-Agent naming you and a contact email. Set it, e.g.\n"
                '  $env:SEC_USER_AGENT = "Your Name your.email@example.com"')
        self.min_interval, self.retries, self._last = min_interval, retries, 0.0

    def company_facts(self, cik: int) -> dict:
        url = FACTS_URL.format(cik=cik)
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent,
                                                       "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed host
                    body = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return json.loads(body)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return {}
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                    raise
            time.sleep(2 ** attempt)
        return {}


class FundamentalsCache:
    """Extracted-concept cache per CIK, refreshed after ``max_age_days``."""

    def __init__(self, cache_dir: str | pathlib.Path = DEFAULT_DIR,
                 client: SecClient | None = None, max_age_days: float = 7.0):
        self.dir = pathlib.Path(cache_dir)
        self._client = client
        self.max_age = max_age_days * 86400

    def _path(self, cik: int) -> pathlib.Path:
        return self.dir / f"{cik:010d}.parquet"

    def rows(self, cik: int, offline: bool = False, refresh: bool = False) -> pd.DataFrame:
        path = self._path(cik)
        fresh = path.exists() and time.time() - path.stat().st_mtime <= self.max_age
        if offline or (fresh and not refresh):
            if path.exists():
                return pd.read_parquet(path)
            if offline:
                raise LookupError(f"CIK {cik} fundamentals not cached")
        if self._client is None:
            self._client = SecClient()
        out = extract(self._client.company_facts(cik))
        self.dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
        return out


def load_panel(tickers: list[str], index: pd.DatetimeIndex, ciks: dict[str, int],
               cache: FundamentalsCache | None = None, offline: bool = False,
               refresh: bool = False) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch (or read cached) fundamentals and build the factor panels.

    Returns ``(panel, unmapped)``; ``unmapped`` are tickers without a CIK.
    Share classes (GOOG/GOOGL) share a CIK and are fetched once.
    """
    cache = cache or FundamentalsCache()
    unmapped = [t for t in tickers if t not in ciks]
    by_cik: dict[int, pd.DataFrame] = {}
    rows = {}
    for t in tickers:
        if t in ciks:
            cik = ciks[t]
            if cik not in by_cik:
                by_cik[cik] = cache.rows(cik, offline=offline, refresh=refresh)
            rows[t] = by_cik[cik]
    panel = build_panel(rows, index)
    return ({f: p.reindex(columns=tickers) for f, p in panel.items()}, unmapped)
