"""Module IV, stage B - signal-based portfolio optimisation and backtest.  [A4]

    max_w  w's / sqrt(w' Sigma w)   s.t.  sum w = 1,  0 <= w_i <= w_max      ([A4] Eq. 3-4)

Extensions requested by the limitations section of [A4] and slide 26 of the
defence: per-asset cap, cluster caps from Module III (hard labels or fuzzy
memberships U: sum_i u_ik w_i <= cap_k), Ledoit-Wolf covariance shrinkage,
weekly/monthly rebalancing and transaction costs in the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from . import config
from .clustering import performance


def covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """Annualised covariance of daily returns (x252)."""
    r = returns.dropna()
    if method == "ledoit_wolf":
        return LedoitWolf().fit(r.to_numpy()).covariance_ * 252
    return np.cov(r.to_numpy().T) * 252


def feasible_limits(groups, w_max: float, cap: float | None) -> tuple[float, float | None, bool]:
    """Relax (w_max, cap) until sum_k min(cap, n_k * w_max) >= 1, i.e. until a
    fully invested portfolio satisfying both limits exists."""
    n = len(groups)
    if cap is None:
        return max(w_max, 1.0 / n), cap, False
    _, counts = np.unique(np.asarray(groups), return_counts=True)
    w, c, relaxed = max(w_max, 1.0 / n), max(cap, 1.0 / len(counts)), False
    while np.minimum(c, counts * w).sum() < 1 - 1e-9:
        w, c, relaxed = min(w + 0.01, 1.0), min(c + 0.02, 1.0), True
    return w, c, relaxed


def _feasible_start(groups, w_max, cap, n):
    if groups is None or cap is None:
        return np.full(n, 1.0 / n)
    g = np.asarray(groups)
    labs, counts = np.unique(g, return_counts=True)
    allowed = np.minimum(cap, counts * w_max)
    share = allowed / allowed.sum()
    x0 = np.zeros(n)
    for lab, cnt, sh in zip(labs, counts, share):
        x0[g == lab] = sh / cnt
    return x0


def _group_constraints(n, groups=None, U=None, cap=None):
    cons = []
    if cap is None:
        return cons
    if U is not None:
        U = np.asarray(U)
        k = U.shape[1]
        cap_k = max(cap, 1.0 / k + 1e-6)
        for j in range(k):
            cons.append({"type": "ineq", "fun": lambda w, u=U[:, j], c=cap_k: c - u @ w})
    elif groups is not None:
        g = np.asarray(groups)
        for lab in np.unique(g):
            mask = (g == lab).astype(float)
            cons.append({"type": "ineq", "fun": lambda w, m=mask, c=cap: c - m @ w})
    return cons


def _solve(obj, n, w_max, groups, U, cap) -> tuple[np.ndarray, bool]:
    """SLSQP from a feasible start; if it fails, the feasible start itself is
    returned (a valid portfolio) together with ``ok=False``."""
    if U is None and groups is not None:
        w_max, cap_eff, _ = feasible_limits(groups, w_max, cap)
    else:
        w_max, cap_eff = max(w_max, 1.0 / n), cap
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + _group_constraints(n, groups, U, cap_eff)
    x0 = _feasible_start(groups if U is None else None, w_max, cap_eff, n)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0.0, w_max)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    ok = res.success and all(c["fun"](res.x) >= -1e-6 for c in cons if c["type"] == "ineq")
    w = np.clip(res.x if ok else x0, 0, None)
    return w / w.sum(), bool(ok)


def max_signal_to_risk(s, cov, w_max=1.0, groups=None, U=None, cluster_cap=None, return_ok=False):
    """Weights maximising w's / sqrt(w'Sigma w); with ``return_ok`` also whether SLSQP converged."""
    s, cov = np.asarray(s, float), np.asarray(cov, float)
    obj = lambda w: -(w @ s) / np.sqrt(max(w @ cov @ w, 1e-12))
    w, ok = _solve(obj, len(s), w_max, groups, U, cluster_cap)
    return (w, ok) if return_ok else w


def min_variance(cov, w_max=1.0, groups=None, U=None, cluster_cap=None):
    cov = np.asarray(cov, float)
    return _solve(lambda w: w @ cov @ w, len(cov), w_max, groups, U, cluster_cap)[0]


def signal_weighted(s, w_max=1.0):
    """w_i proportional to s_i, capped and renormalised (water-filling)."""
    s = np.clip(np.asarray(s, float), 0, None)
    w = s / s.sum() if s.sum() > 0 else np.full(len(s), 1 / len(s))
    w_max = max(w_max, 1.0 / len(s))
    for _ in range(50):
        over = w > w_max + 1e-12
        if not over.any():
            break
        excess = (w[over] - w_max).sum()
        w[over] = w_max
        free = ~over & (w < w_max)
        w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else 0
    return w / w.sum()


STRATEGIES = ("Signal Weighted", "Max Signal-to-Risk", "Min Variance", "Equal Weighted")


def target_weights(strategy, s, cov, w_max, groups=None, U=None, cluster_cap=None):
    n = len(s)
    if strategy == "Signal Weighted":
        return signal_weighted(s, w_max)
    if strategy == "Max Signal-to-Risk":
        return max_signal_to_risk(s, cov, w_max, groups, U, cluster_cap)
    if strategy == "Min Variance":
        return min_variance(cov, w_max, groups, U, cluster_cap)
    return np.full(n, 1.0 / n)


@dataclass
class BacktestResult:
    curves: pd.DataFrame        # cumulative value, base = 100
    daily: pd.DataFrame
    metrics: pd.DataFrame
    weights_last: pd.DataFrame  # strategy x asset
    turnover: pd.Series


def backtest(prices: pd.DataFrame, ranks: pd.DataFrame, rebalance: str = "W", tc_bps: float = 10.0,
             cov_method: str = "ledoit_wolf", w_max: float = 1.0, groups=None, U=None,
             cluster_cap=None, benchmark: pd.Series | None = None,
             risk_free: float = config.RISK_FREE, strategies=STRATEGIES) -> BacktestResult:
    """Strictly out-of-sample: weights at the close of t use signals and
    covariance known at t and earn the return from t to t+1."""
    rets = prices.pct_change()
    dates = ranks.dropna(how="all").index
    dates = dates[dates.isin(rets.index)]
    if rebalance == "D":
        rebal = set(dates)
    else:
        per = pd.Series(dates, index=dates).groupby(pd.PeriodIndex(dates, freq=rebalance[0])).last()
        rebal = set(per.values)
    cols = list(prices.columns)
    w = {s: np.full(len(cols), 1.0 / len(cols)) for s in strategies}
    out = {s: [] for s in strategies}
    turn = {s: [] for s in strategies}
    pos = {d: i for i, d in enumerate(rets.index)}
    used = []
    for d in dates:
        i = pos[d]
        if i + 1 >= len(rets.index):
            break
        cost = dict.fromkeys(strategies, 0.0)
        if d in rebal:
            hist = rets.iloc[max(0, i - config.COV_WINDOW + 1): i + 1].dropna()
            cov = covariance(hist, cov_method)
            s = ranks.loc[d, cols].fillna(0.5).to_numpy()
            for st in strategies:
                new = target_weights(st, s, cov, w_max, groups, U, cluster_cap)
                to = float(np.abs(new - w[st]).sum() / 2)
                turn[st].append(to)
                cost[st] = to * tc_bps / 1e4
                w[st] = new
        r_next = rets.iloc[i + 1][cols].fillna(0).to_numpy()
        for st in strategies:
            out[st].append(float(w[st] @ r_next) - cost[st])
            grown = w[st] * (1 + r_next)
            w[st] = grown / grown.sum()
        used.append(rets.index[i + 1])
    daily = pd.DataFrame({s: out[s] for s in strategies}, index=pd.Index(used, name="Date"))
    if benchmark is not None:
        daily["Benchmark"] = benchmark.pct_change().reindex(daily.index).fillna(0)
    else:
        daily["Рынок (равновзв. buy&hold)"] = (prices.reindex(daily.index).div(prices.loc[dates[0]])
                                              .mean(axis=1).pct_change().fillna(0))
    curves = 100 * (1 + daily).cumprod()
    rows = []
    for s in daily.columns:
        rows.append({"strategy": s} | performance(daily[s], risk_free)
                    | {"turnover": float(np.mean(turn[s])) if s in turn and turn[s] else 0.0})
    metrics = pd.DataFrame(rows).set_index("strategy")
    wl = pd.DataFrame({s: w[s] for s in strategies}, index=cols).T
    return BacktestResult(curves, daily, metrics, wl, pd.Series({s: np.mean(turn[s]) for s in strategies}))
