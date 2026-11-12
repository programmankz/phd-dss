"""Property tests of the feasibility condition of the cluster limits (T1) and of the relaxation
loop, and of the liquidity features of Module III."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import clustering, portfolio  # noqa: E402


def _lp_feasible(groups, w_max, cap) -> bool:
    g = np.asarray(groups)
    n = len(g)
    labs = np.unique(g)
    A = np.array([(g == k).astype(float) for k in labs])
    res = linprog(np.zeros(n), A_ub=A, b_ub=np.full(len(labs), cap), A_eq=np.ones((1, n)), b_eq=[1.0],
                  bounds=[(0, w_max)] * n, method="highs")
    return res.status == 0


def test_t1_condition_equals_lp_feasibility():
    rng = np.random.default_rng(0)
    for _ in range(400):
        n = int(rng.integers(2, 12))
        k = int(rng.integers(1, n + 1))
        groups = rng.integers(0, k, n)
        w_max, cap = float(rng.uniform(0.05, 0.6)), float(rng.uniform(0.1, 0.9))
        _, counts = np.unique(groups, return_counts=True)
        cond = np.minimum(cap, counts * w_max).sum() >= 1 - 1e-9
        assert cond == _lp_feasible(groups, w_max, cap), (groups, w_max, cap)


def test_relaxation_terminates_with_feasible_limits():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(2, 15))
        groups = rng.integers(0, int(rng.integers(1, n + 1)), n)
        w_max, cap = float(rng.uniform(0.01, 0.5)), float(rng.uniform(0.01, 0.9))
        w, c, relaxed = portfolio.feasible_limits(groups, w_max, cap)
        assert w <= 1.0 and c <= 1.0 and w >= w_max - 1e-12 and c >= min(cap, 1.0) - 1e-12
        assert _lp_feasible(groups, w, c)
        # the loop moves w by 0.01 and c by 0.02, so it ends in at most 100 steps
        assert w - max(w_max, 1 / n) <= 1.0 + 1e-9
        if not relaxed:
            assert np.isclose(w, max(w_max, 1 / n)) and np.isclose(c, max(cap, 1 / len(np.unique(groups))))


def test_liquidity_features():
    d = pd.bdate_range("2024-01-01", periods=6)
    px = pd.DataFrame({"A": [100, 100, 101, 101, 102, 102], "B": [50, 51, 52, 53, 54, 55]}, index=d, dtype=float)
    vol = pd.DataFrame({"A": [0, 0, 1e6, 0, 2e6, 0], "B": 1e8}, index=d)
    f = clustering.liquidity_features(px, vol)
    assert np.isclose(f.loc["A", "zero_ret_share"], 3 / 5) and f.loc["B", "zero_ret_share"] == 0
    # A: days with trades are 3rd and 5th: |r| = 0.01, 0.0099 over 1e6, 2e6 tenge
    amihud_a = np.mean([0.01 / 1e6, (102 / 101 - 1) / 2e6]) * 1e9
    assert np.isclose(f.loc["A", "log_amihud"], np.log(amihud_a))
    assert f.loc["A", "log_amihud"] > f.loc["B", "log_amihud"]
