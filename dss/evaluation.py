"""Contribution of the DSS modules: full factorial ablation and Shapley values.

Each module (I macro rule, II currency rule, III cluster limits, IV signals) is
a player of a cooperative game; the value v(S) of a coalition S is a metric of
the backtest in which only the modules in S are switched on. With four modules
there are 2^4 = 16 configurations; v(empty set) is the "EW + KZT" portfolio
(d = d0, h = 0, equal weights in the risky part). The Shapley value of module i

    phi_i = sum_{S not containing i} |S|! (n - |S| - 1)! / n! * (v(S + i) - v(S))

is the average marginal contribution of i over all orders in which the modules
can be added; the values add up to v(all) - v(empty set) (efficiency). Unlike a
one-at-a-time ablation it accounts for interactions (e.g. the currency rule
only matters when the defensive part is large).

Uncertainty: the 16 daily return series are resampled with the same circular
block bootstrap indices (:mod:`dss.significance`), the metric and the Shapley
values are recomputed on every resample, and the percentile interval is
reported. A fixed-window metric (return in a stress window) has no bootstrap
interval: resampling days destroys the window.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from . import significance

MODULES = ("I", "II", "III", "IV")


def coalitions(players=MODULES) -> list[frozenset]:
    """All subsets of ``players``, ordered by size, then lexicographically."""
    out = []
    for k in range(len(players) + 1):
        out += [frozenset(c) for c in combinations(players, k)]
    return out


def coalition_name(s, players=MODULES) -> str:
    return "+".join(p for p in players if p in s) or "none"


def shapley(values: dict, players=MODULES) -> pd.Series:
    """Exact Shapley values from v(S) for every subset S (keys: frozensets)."""
    n = len(players)
    phi = {}
    for i in players:
        tot = 0.0
        for s in coalitions([p for p in players if p != i]):
            w = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
            tot += w * (values[s | {i}] - values[s])
        phi[i] = tot
    return pd.Series(phi, name="shapley")


def efficiency_gap(values: dict, phi: pd.Series, players=MODULES) -> float:
    """sum(phi) - (v(all) - v(empty)); zero up to rounding."""
    return float(phi.sum() - (values[frozenset(players)] - values[frozenset()]))


# ----------------------------------------------------------------------------- metrics of daily returns
def sharpe_metric(excess: np.ndarray) -> np.ndarray:
    """Annualised Sharpe of daily excess returns (last axis), as in Table 4."""
    return significance.sharpe(excess, 0.0)


def max_drawdown_metric(r: np.ndarray) -> np.ndarray:
    """Maximum drawdown (negative fraction) of the compounded daily returns along the last axis."""
    curve = np.cumprod(1 + np.asarray(r, float), axis=-1)
    peak = np.maximum.accumulate(curve, axis=-1)
    return (curve / peak - 1).min(axis=-1)


def window_return(r: pd.Series, a, b) -> float:
    seg = r.loc[a:b]
    return float((1 + seg).prod() - 1) if len(seg) else np.nan


def shapley_table(daily: pd.DataFrame, rf: pd.Series, windows: dict | None = None, players=MODULES,
                  n_boot: int = significance.N_BOOT, seed: int | None = None, level: float = 0.95) -> pd.DataFrame:
    """Shapley values of the modules for Sharpe, maximum drawdown and stress-window returns.

    ``daily`` has one column per coalition, named by :func:`coalition_name`. Returns a long
    table: metric, module, phi, ci_low, ci_high, and rows module = 'sum' (sum of phi),
    'v(all)-v(none)' and 'efficiency gap'."""
    names = {s: coalition_name(s, players) for s in coalitions(players)}
    missing = [n for n in names.values() if n not in daily]
    if missing:
        raise ValueError(f"no returns for coalitions {missing}")
    daily = daily[list(names.values())].dropna()
    excess = daily.sub(rf.reindex(daily.index).fillna(0.0), axis=0)
    a = (1 - level) / 2
    idx = significance.block_bootstrap_indices(len(daily), n_boot, seed=seed if seed is not None else
                                               significance.config.RANDOM_SEED)
    rows = []

    def add(metric, point: dict, boot: np.ndarray | None):
        phi = shapley(point, players)
        bphi = None
        if boot is not None:
            # boot: coalitions x n_boot, in the order of names
            keys = list(names)
            bphi = np.array([shapley({k: boot[j, b] for j, k in enumerate(keys)}, players).to_numpy()
                             for b in range(boot.shape[1])])
        for m_i, p in enumerate(players):
            lo = hi = np.nan
            if bphi is not None:
                lo, hi = np.nanquantile(bphi[:, m_i], a), np.nanquantile(bphi[:, m_i], 1 - a)
            rows.append({"metric": metric, "module": p, "phi": phi[p], "ci_low": lo, "ci_high": hi})
        total = point[frozenset(players)] - point[frozenset()]
        rows.append({"metric": metric, "module": "sum", "phi": float(phi.sum()), "ci_low": np.nan, "ci_high": np.nan})
        rows.append({"metric": metric, "module": "v(all)-v(none)", "phi": total, "ci_low": np.nan, "ci_high": np.nan})
        rows.append({"metric": metric, "module": "efficiency gap", "phi": efficiency_gap(point, phi, players),
                     "ci_low": np.nan, "ci_high": np.nan})

    ex = excess.to_numpy().T
    raw = daily.to_numpy().T
    keys = list(names)
    add("sharpe", {k: float(sharpe_metric(ex[j])) for j, k in enumerate(keys)},
        np.vstack([sharpe_metric(ex[j][idx]) for j in range(len(keys))]))
    add("max_drawdown", {k: float(max_drawdown_metric(raw[j])) for j, k in enumerate(keys)},
        np.vstack([max_drawdown_metric(raw[j][idx]) for j in range(len(keys))]))
    for wname, (wa, wb) in (windows or {}).items():
        point = {k: window_return(daily[names[k]], wa, wb) for k in keys}
        if all(np.isfinite(v) for v in point.values()):
            add(f"return {wname}", point, None)
    out = pd.DataFrame(rows)
    out.attrs |= {"n_days": len(daily), "n_boot": n_boot, "block": significance.block_length(len(daily))}
    return out


def paired_difference_test(a: pd.Series, b: pd.Series, metric, null: float = 0.0, n_boot: int = significance.N_BOOT,
                           seed: int | None = None, level: float = 0.95) -> dict:
    """Paired circular block bootstrap of metric(a) - metric(b) for two daily return series.

    ``metric`` maps an array (n_boot x n or n) of daily returns to values along the last axis.
    Returns the difference, its percentile interval and the shift-method p-values: two-sided for
    H0: difference = ``null`` and one-sided for H1: difference > ``null``."""
    both = pd.concat([a, b], axis=1).dropna()
    x, y = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    idx = significance.block_bootstrap_indices(len(both), n_boot, seed=seed if seed is not None
                                               else significance.config.RANDOM_SEED)
    diff = float(metric(x) - metric(y))
    bd = metric(x[idx]) - metric(y[idx])
    centred = bd - diff + null
    q = (1 - level) / 2
    return {"diff": diff, "low": float(np.nanquantile(bd, q)), "high": float(np.nanquantile(bd, 1 - q)),
            "p_two_sided": float(np.nanmean(np.abs(centred - null) >= abs(diff - null))),
            "p_greater": float(np.nanmean(centred >= diff)), "n_days": len(both)}
