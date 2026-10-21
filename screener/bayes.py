"""Step 1: a Bayesian scoring layer that *learns* factor weights.

Instead of hard-coding composite weights, we run a conjugate Bayesian linear
regression of next-period cross-sectional (relative) returns on the z-scored
factors:

    y = X w + e,      e ~ N(0, sigma^2 I),      prior  w ~ N(0, tau^2 I)

The posterior is closed-form (no sampler needed):

    S = (X^T X / sigma^2 + I / tau^2)^{-1}        # posterior covariance
    m = S X^T y / sigma^2                         # posterior mean  (= weights)

The Normal prior is exactly ridge regression with penalty ``sigma^2 / tau^2``:
a factor with little signal is pulled toward zero, and its posterior variance
(large) flags it as low-confidence. ``BayesianScorer`` is a drop-in alternative
to :func:`screener.score.composite_score`.

Lookahead note: the *estimator* is agnostic to time -- it just takes ``X`` and
``y``. Avoiding lookahead is the caller's job: fit only on data whose forward
window has already realized. ``screen.make_bayes_signal_fn`` does this walk-forward
for the backtest; a full-sample ``fit_weights`` is in-sample research only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import score as sc


@dataclass
class Posterior:
    """Posterior over factor weights from a conjugate Bayesian regression."""
    names: list[str]
    mean: np.ndarray          # posterior mean weight per factor
    cov: np.ndarray           # posterior covariance matrix
    noise_var: float          # sigma^2 actually used
    prior_var: float          # tau^2 used
    n_obs: int                # rows used in the fit

    @property
    def std(self) -> np.ndarray:
        """Posterior standard deviation per weight."""
        return np.sqrt(np.diag(self.cov))

    @property
    def weights(self) -> dict[str, float]:
        """Posterior-mean weights keyed by factor name (the new composite weights)."""
        return dict(zip(self.names, self.mean))

    @property
    def signal_to_noise(self) -> dict[str, float]:
        """``mean / std`` per factor -- a t-like confidence score."""
        return dict(zip(self.names, self.mean / np.where(self.std == 0, np.nan, self.std)))

    def shrunk_weights(self) -> dict[str, float]:
        """Confidence-weighted weights: ``mean * mean^2 / (mean^2 + var)``.

        Pushes low-confidence (high-variance) factors further toward zero than
        the raw posterior mean. Useful when you want extra robustness.
        """
        var = np.diag(self.cov)
        m = self.mean
        shrink = m * (m ** 2) / (m ** 2 + var)
        return dict(zip(self.names, shrink))


@dataclass
class DesignStats:
    """Sufficient statistics of a regression sample: ``X'X, X'y, y'y, n``.

    The posterior depends on the data only through these, so samples can be
    accumulated incrementally with ``+`` (used by the walk-forward signal).
    """
    names: list[str]
    xtx: np.ndarray
    xty: np.ndarray
    yty: float
    n: int

    @classmethod
    def from_arrays(cls, X: np.ndarray, y: np.ndarray, names: list[str]) -> "DesignStats":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        return cls(list(names), X.T @ X, X.T @ y, float(y @ y), X.shape[0])

    def __add__(self, other: "DesignStats") -> "DesignStats":
        if self.names != other.names:
            raise ValueError("cannot add stats over different factors")
        return DesignStats(self.names, self.xtx + other.xtx, self.xty + other.xty,
                           self.yty + other.yty, self.n + other.n)


def _solve_posterior(s: DesignStats, prior_var: float,
                     noise_var: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    p = len(s.names)
    if s.n == 0:
        raise ValueError("no observations to fit")
    if noise_var is None:
        # Cheap, stable estimate: ridge fit at the prior scale, then residual var.
        # ||y - X b||^2 expanded so it needs only the sufficient statistics.
        beta0 = np.linalg.solve(s.xtx + (1.0 / prior_var) * np.eye(p), s.xty)
        rss = s.yty - 2.0 * beta0 @ s.xty + beta0 @ s.xtx @ beta0
        noise_var = max(float(rss) / max(s.n - p, 1), 1e-12)
    precision = s.xtx / noise_var + np.eye(p) / prior_var
    cov = np.linalg.inv(precision)
    mean = cov @ s.xty / noise_var
    return mean, cov, float(noise_var)


def posterior_from_stats(stats: DesignStats, prior_var: float = 1.0,
                         noise_var: float | None = None) -> "Posterior":
    mean, cov, nv = _solve_posterior(stats, prior_var, noise_var)
    return Posterior(stats.names, mean, cov, nv, prior_var, n_obs=stats.n)


def bayesian_linear_regression(X: np.ndarray, y: np.ndarray,
                               prior_var: float = 1.0,
                               noise_var: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Conjugate Normal-prior linear regression. Returns ``(mean, cov, noise_var)``.

    ``prior_var`` (tau^2) controls shrinkage: smaller -> stronger pull to zero.
    ``noise_var`` (sigma^2); if ``None`` it is estimated from ridge residuals.
    """
    X = np.asarray(X, dtype=float)
    stats = DesignStats.from_arrays(X, y, [str(i) for i in range(X.shape[1])])
    return _solve_posterior(stats, prior_var, noise_var)


def _stack_factors(factors: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Stack ``(dates x tickers)`` factor frames into a long ``(date,ticker) x factor`` table."""
    names = list(factors)
    cols = {name: factors[name].stack() for name in names}
    long = pd.concat(cols, axis=1)
    long.columns = names
    return long, names


def design_stats(factors: dict[str, pd.DataFrame], prices: pd.DataFrame,
                 horizon: int = 21) -> DesignStats:
    """Regression sample from z-scored factors and forward returns, as statistics.

    ``y`` is the cross-sectionally demeaned forward return over ``horizon``
    periods (relative performance), matching the screener's cross-sectional
    ranking objective and absorbing the market move. Rows whose forward return
    has not realized inside ``prices`` are NaN and dropped.
    """
    long, names = _stack_factors(factors)

    fwd = prices.shift(-horizon) / prices - 1.0
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)          # cross-sectional demean
    y = fwd.stack()
    y.name = "y"

    data = long.join(y, how="inner").dropna()
    return DesignStats.from_arrays(data[names].to_numpy(), data["y"].to_numpy(), names)


def fit_weights(factors: dict[str, pd.DataFrame], prices: pd.DataFrame,
                horizon: int = 21, prior_var: float = 1.0,
                noise_var: float | None = None) -> Posterior:
    """Estimate factor-weight posterior from z-scored factors and forward returns."""
    return posterior_from_stats(design_stats(factors, prices, horizon), prior_var, noise_var)


class BayesianScorer:
    """Drop-in alternative to ``composite_score`` with learned weights.

    Usage mirrors the hard-coded path::

        scorer = BayesianScorer(prior_var=1.0, horizon=21).fit(factors, prices)
        composite = scorer.composite(factors)        # (dates x tickers)
        weights   = scorer.posterior.weights         # learned weights
    """

    def __init__(self, prior_var: float = 1.0, horizon: int = 21,
                 noise_var: float | None = None, shrink: bool = False):
        self.prior_var = prior_var
        self.horizon = horizon
        self.noise_var = noise_var
        self.shrink = shrink
        self.posterior: Posterior | None = None

    def fit(self, factors: dict[str, pd.DataFrame], prices: pd.DataFrame) -> "BayesianScorer":
        self.posterior = fit_weights(factors, prices, self.horizon,
                                     self.prior_var, self.noise_var)
        return self

    @property
    def weights(self) -> dict[str, float]:
        if self.posterior is None:
            raise RuntimeError("call fit() first")
        return self.posterior.shrunk_weights() if self.shrink else self.posterior.weights

    def composite(self, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Weighted composite using the learned weights -- same shape as the inputs."""
        return sc.composite_score(factors, self.weights)
