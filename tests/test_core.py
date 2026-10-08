"""Unit tests of the scientific invariants of the DSS.

Run with pytest (``pytest -q``) or without it (``python tests/run.py``).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import config, fx, macro, portfolio, signals  # noqa: E402
from dss.clustering import dtw_matrix, fuzzy_cmeans, kmedoids_pam  # noqa: E402


def close(a, b, tol=1e-6):
    return bool(np.allclose(a, b, atol=tol))


# ---------------------------------------------------------------- Module I
def _toy_panel():
    years = list(range(2000, 2012))
    return pd.DataFrame([np.arange(12, dtype=float), np.arange(12, dtype=float)[::-1] - 3],
                        index=pd.Index(["A", "B"], name="country"), columns=years)


def test_features_use_only_past_values():
    f = macro.build_features(macro.to_long(_toy_panel())).set_index(["country", "year"])
    row = f.loc[("A", 2006)]               # growth of A in year t is t - 2000
    assert row["lag1"] == 5 and row["lag3"] == 3
    assert close(row["rolling_mean3"], (5 + 4 + 3) / 3)
    assert close(row["rolling_mean5"], (5 + 4 + 3 + 2 + 1) / 5)
    assert close(row["cumulative_mean"], np.mean([0, 1, 2, 3, 4, 5]))
    assert row["growth"] == 6                # the target enters no predictor


def test_arima_forecast_has_horizon_and_interval():
    y = np.cumsum(np.random.default_rng(0).normal(0, 1, 40)) + 3
    best, table = macro.arima_select(y, 5)
    assert len(best.forecast) == 5
    assert np.all(best.lower <= best.forecast) and np.all(best.forecast <= best.upper)
    assert "ARIMA(3, 1, 0)" in set(table["order"])


def test_sensitivity_reproduces_table_10_endpoints():
    assert close(macro.sensitivity({"brent": 45})["brent"], -2.4)
    assert close(macro.sensitivity({"brent": 110})["brent"], 1.8)
    assert close(macro.sensitivity({"cpc": 12})["cpc"], -2.3)
    assert close(macro.sensitivity({})["total"], 0.0)


def test_scenario_offsets_match_table_9():
    off = np.array(config.SCENARIOS["stress"]["offsets"])
    base = np.array([4.34, 4.64, 4.99, 5.23, 5.45])
    assert close(base + off, [1.04, 1.34, 2.09, 2.43, 2.75], tol=0.01)


def test_plausibility_rule():
    hist = np.array([4, 5, 3, 4, 6, 5, 4, 3, 5, 4, 2, 4, 5, 3, 4], float)
    assert macro.plausible(np.array([4.0, 4.5, 5.0]), hist)
    assert not macro.plausible(np.array([-3.0, -5.0, -7.0]), hist)


# ---------------------------------------------------------------- Module II
def test_fx_describe_and_trend():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    df = pd.DataFrame({"USD": np.linspace(400, 500, 400), "RUB": 5 + np.sin(np.arange(400) / 20)}, index=idx)
    st = fx.describe(df)
    assert list(st.index[:2]) == ["Среднее", "Стандартная ошибка"] and st.loc["Счёт", "USD"] == 400
    reg, _ = fx.trend_regression(df)
    assert close(reg.loc["USD/KZT", "R2"], 1.0) and reg.loc["RUB/KZT", "R2"] < 0.5


# ---------------------------------------------------------------- Module III
def test_kmedoids_separates_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (10, 2)), rng.normal(5, 0.1, (10, 2))])
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    lab, med, _ = kmedoids_pam(D, 2)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[-1]
    assert set(med.tolist()) <= set(range(20))       # medoids are real objects


def test_fcm_memberships_are_a_partition():
    X = np.random.default_rng(2).normal(size=(30, 3))
    _, U, _ = fuzzy_cmeans(X, 3)
    assert close(U.sum(axis=1), 1) and (U >= 0).all() and (U <= 1).all()


def test_dtw_is_a_symmetric_dissimilarity():
    S = np.array([[0, 1, 2, 1, 0], [0, 0, 1, 2, 1], [5, 5, 5, 5, 5]], float)
    D = dtw_matrix(S)
    assert close(D, D.T) and close(np.diag(D), 0) and D[0, 1] < D[0, 2]


# ---------------------------------------------------------------- Module IV
def test_rank_signal_in_unit_interval_with_average_ties():
    pred = pd.DataFrame([[0.1, 0.3, 0.3, -0.2]], columns=list("abcd"))
    s = signals.rank_signal(pred).iloc[0]
    assert close(s["d"], 0.25) and close(s["b"], 0.875) and close(s["c"], 0.875) and s.max() <= 1


def test_optimizer_respects_all_constraints():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 8))
    cov = A @ A.T / 8 * 0.04 + np.eye(8) * 0.01
    s = np.linspace(0.1, 1, 8)
    groups = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    w = portfolio.max_signal_to_risk(s, cov, w_max=0.3, groups=groups, cluster_cap=0.5)
    assert close(w.sum(), 1) and (w >= -1e-9).all() and (w <= 0.3 + 1e-6).all()
    for g in range(3):
        assert w[groups == g].sum() <= 0.5 + 1e-6


def test_infeasible_limits_are_relaxed():
    w_max, cap, relaxed = portfolio.feasible_limits(np.array([0] * 10 + [1, 2]), 0.2, 0.5)
    assert relaxed and np.minimum(cap, np.array([10, 1, 1]) * w_max).sum() >= 1


def test_backtest_is_out_of_sample():
    """Ranks that foresee the t+1 return must beat equal weights; this only holds
    if weights set at t earn the return from t to t+1 (no look-ahead shift)."""
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2020-01-01", periods=300)
    rets = pd.DataFrame(rng.normal(0, 0.02, (300, 5)), index=idx, columns=list("abcde"))
    prices = 100 * (1 + rets).cumprod()
    oracle = signals.rank_signal(rets.shift(-1)).iloc[260:-1]
    bt = portfolio.backtest(prices, oracle, rebalance="D", tc_bps=0,
                            strategies=("Signal Weighted", "Equal Weighted"))
    m = bt.metrics
    assert m.loc["Signal Weighted", "total_return"] > m.loc["Equal Weighted", "total_return"]
