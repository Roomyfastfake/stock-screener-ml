"""Strategy-evaluation tools: walk-forward runs, bootstrap Sharpe CI, deflated Sharpe.

Sharpe inputs here are per-period (not annualised) unless a function says so.
Deflated Sharpe follows Bailey & Lopez de Prado (2014):
  PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )
  with kurt the raw (non-excess) kurtosis, and the deflated benchmark
  SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))),
  g = Euler-Mascheroni, V = variance of the N trial Sharpes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from . import stats as st
from .engine import BacktestResult, SignalFn, backtest

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(index: pd.DatetimeIndex, train_size: int, test_size: int,
                        expanding: bool = True) -> list[Split]:
    """Consecutive, non-overlapping test windows of ``test_size`` rows.

    Train is the ``train_size`` rows before each test window (rolling) or
    everything before it (expanding). Train always ends strictly before test.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    out = []
    t0 = train_size
    while t0 < len(index):
        t1 = min(t0 + test_size, len(index)) - 1
        tr0 = 0 if expanding else t0 - train_size
        out.append(Split(index[tr0], index[t0 - 1], index[t0], index[t1]))
        t0 += test_size
    return out


@dataclass
class WalkForwardResult:
    returns: pd.Series
    folds: list                         # list[(Split, BacktestResult)]
    stats: dict


def walk_forward(prices: pd.DataFrame, fit_fn: Callable[[pd.DataFrame], SignalFn],
                 splits: list[Split], **bt_kwargs) -> WalkForwardResult:
    """Fit on each train window, trade only inside the following test window.

    ``fit_fn(train_prices) -> signal_fn``. The signal_fn may still read history
    before the test start (that is the past), but whatever was *fit* only saw
    train rows. Each fold starts flat, so fold boundaries pay a full rebuild.
    """
    folds, parts = [], []
    for sp in splits:
        signal_fn = fit_fn(prices.loc[sp.train_start:sp.train_end])
        res = backtest(prices.loc[:sp.test_end], signal_fn, start=sp.test_start, **bt_kwargs)
        folds.append((sp, res))
        parts.append(res.returns.loc[sp.test_start:])
    returns = pd.concat(parts) if parts else pd.Series(dtype=float)
    return WalkForwardResult(returns, folds, st.perf_stats(returns))


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def _sharpe(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, x.mean(axis=-1) / sd, np.nan)


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, wrap-around."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, mean_block: float = 20.0,
                        alpha: float = 0.05, seed: int = 0,
                        periods_per_year: int = st.TRADING_DAYS) -> dict:
    """Annualised Sharpe with a stationary-bootstrap percentile CI.

    Block resampling keeps short-range autocorrelation that an iid bootstrap
    would destroy (which would understate the CI width).
    """
    x = returns.dropna().to_numpy(dtype=float)
    ann = math.sqrt(periods_per_year)
    rng = np.random.default_rng(seed)
    boots = _sharpe(x[stationary_bootstrap_indices(len(x), n_boot, mean_block, rng)]) * ann
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"sharpe": float(_sharpe(x) * ann), "lo": float(lo), "hi": float(hi),
            "p_le_zero": float((boots <= 0).mean())}


# --------------------------------------------------------------------------- #
# probabilistic / deflated Sharpe
# --------------------------------------------------------------------------- #
def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > ``sr_benchmark``) given skew and fat tails."""
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    sd = x.std(ddof=1)
    if n < 3 or sd == 0:
        return float("nan")
    sr = x.mean() / sd
    z = (x - x.mean()) / x.std(ddof=0)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected max per-period SR of ``n_trials`` skill-less strategies."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1 or sr_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(returns: pd.Series, trial_sharpes) -> float:
    """PSR against the expected best of all trials tried (per-period Sharpes).

    ``trial_sharpes`` holds the per-period Sharpe of every configuration you
    evaluated, including the chosen one. Report this, not the raw Sharpe, when
    the strategy was picked from a search.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    var = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    return probabilistic_sharpe(returns, expected_max_sharpe(len(trials), var))
