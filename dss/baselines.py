"""Standard portfolio construction rules used as baselines for the risky part.

All rules are long-only, fully invested and respect the same per-asset cap
w_max as the DSS optimiser (rules without a native cap are capped by
water-filling, :func:`dss.portfolio.signal_weighted`). Inputs are known at the
rebalancing date t: the trailing window of daily returns (252 days) and, for
Black-Litterman, the Module IV ranks at t.

* ``1/N``          - equal weights (DeMiguel, Garlappi and Uppal, 2009);
* ``min_variance`` - minimum variance, Ledoit-Wolf covariance;
* ``erc``          - equal risk contributions (Maillard, Roncalli and Teiletche, 2010),
                     Spinu's convex formulation;
* ``hrp``          - hierarchical risk parity (Lopez de Prado, 2016): single linkage on
                     the distance sqrt(2 (1 - rho)), quasi-diagonal order, recursive
                     bisection with inverse-variance allocation;
* ``markowitz``    - maximum Sharpe ratio with historical mean excess returns of the window;
* ``black_litterman`` - equal-weight equilibrium prior pi = delta Sigma w_eq (delta = 2.5,
                     tau = 0.05); absolute views from the Module IV ranks by Grinold's rule
                     alpha_i = IC * sigma_i * z_i with IC = 0.05 fixed in advance; view
                     uncertainty Omega = diag(tau P Sigma P'); weights maximise
                     w' mu_BL - delta / 2 w' Sigma w.

The constants (delta, tau, IC) are textbook values fixed before the experiment, not tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from . import portfolio

BASELINES = ("1/N", "min_variance", "erc", "hrp", "markowitz", "black_litterman")
LABELS = {"1/N": "1/N", "min_variance": "Минимальная дисперсия", "erc": "Паритет риска (ERC)",
          "hrp": "HRP", "markowitz": "Марковиц (макс. Шарп)", "black_litterman": "Блэк–Литтерман"}
BL_DELTA, BL_TAU, BL_IC = 2.5, 0.05, 0.05


def _cap(w: np.ndarray, w_max: float) -> np.ndarray:
    return portfolio.signal_weighted(np.clip(w, 0, None), w_max)


def erc(cov: np.ndarray, w_max: float = 1.0) -> np.ndarray:
    """Equal risk contributions: argmin 0.5 y'Sigma y - sum(log y) / n, w = y / sum(y)."""
    cov = np.asarray(cov, float)
    n = len(cov)
    scale = np.sqrt(np.diag(cov).mean())
    c = cov / scale ** 2
    obj = lambda y: 0.5 * y @ c @ y - np.log(y).sum() / n          # noqa: E731
    grad = lambda y: c @ y - 1.0 / (n * y)                          # noqa: E731
    y0 = 1.0 / np.sqrt(np.diag(c))
    res = minimize(obj, y0 / y0.sum(), jac=grad, method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    w = res.x / res.x.sum()
    return _cap(w, w_max)


def risk_contributions(w, cov) -> np.ndarray:
    w, cov = np.asarray(w, float), np.asarray(cov, float)
    rc = w * (cov @ w)
    return rc / rc.sum()


def hrp(cov: np.ndarray, w_max: float = 1.0) -> np.ndarray:
    cov = np.asarray(cov, float)
    n = len(cov)
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip(2 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), "single")))
    w = np.ones(n)
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]

        def var(ix):
            sub = cov[np.ix_(ix, ix)]
            ivp = 1 / np.diag(sub)
            ivp /= ivp.sum()
            return float(ivp @ sub @ ivp)
        vl, vr = var(left), var(right)
        alpha = 1 - vl / (vl + vr)
        w[left] *= alpha
        w[right] *= 1 - alpha
        stack += [left, right]
    return _cap(w / w.sum(), w_max)


def markowitz(mu: np.ndarray, cov: np.ndarray, w_max: float = 1.0) -> np.ndarray:
    """Maximum Sharpe ratio for expected excess returns ``mu``; if no asset has a positive
    expected excess return the tangency portfolio does not exist and minimum variance is used."""
    mu = np.asarray(mu, float)
    if not (mu > 0).any():
        return portfolio.min_variance(cov, w_max)
    return portfolio.max_signal_to_risk(mu, cov, w_max)


def black_litterman_mu(cov: np.ndarray, ranks: np.ndarray, delta: float = BL_DELTA, tau: float = BL_TAU,
                       ic: float = BL_IC) -> np.ndarray:
    """Posterior expected excess returns with absolute views from the ranks (P = I)."""
    cov = np.asarray(cov, float)
    n = len(cov)
    pi = delta * cov @ np.full(n, 1.0 / n)
    s = np.asarray(ranks, float)
    z = (s - s.mean()) / s.std() if s.std() > 0 else np.zeros(n)
    q = pi + ic * np.sqrt(np.diag(cov)) * z
    tc = tau * cov
    omega = np.diag(np.diag(tc))
    a = np.linalg.inv(tc) + np.linalg.inv(omega)
    return np.linalg.solve(a, np.linalg.solve(tc, pi) + np.linalg.solve(omega, q))


def mean_variance(mu: np.ndarray, cov: np.ndarray, delta: float, w_max: float = 1.0) -> np.ndarray:
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    n = len(mu)
    w_max = max(w_max, 1.0 / n)
    obj = lambda w: -(w @ mu - 0.5 * delta * w @ cov @ w)        # noqa: E731
    res = minimize(obj, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0.0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}], options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0, None) if res.success else np.full(n, 1.0 / n)
    return w / w.sum()


def weights(rule: str, window: pd.DataFrame, w_max: float = 1.0, ranks=None, rf_annual: float = 0.0) -> np.ndarray:
    """Weights of the risky part by ``rule`` from the trailing daily returns ``window``."""
    n = window.shape[1]
    if rule == "1/N":
        return np.full(n, 1.0 / n)
    cov = portfolio.covariance(window, "ledoit_wolf")
    if rule == "min_variance":
        return portfolio.min_variance(cov, w_max)
    if rule == "erc":
        return erc(cov, w_max)
    if rule == "hrp":
        return hrp(cov, w_max)
    if rule == "markowitz":
        mu = window.dropna().mean().to_numpy() * 252 - rf_annual
        return markowitz(mu, cov, w_max)
    if rule == "black_litterman":
        s = np.full(n, 0.5) if ranks is None else np.asarray(ranks, float)
        return mean_variance(black_litterman_mu(cov, s), cov, BL_DELTA, w_max)
    raise ValueError(f"unknown baseline {rule!r}")
