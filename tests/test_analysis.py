"""Tests of the integration rules (coefficients, sensitivity, monotonicity),
the optimiser fallback and the significance tests of Module IV (FR-IV-13).

Run with pytest (``pytest -q``) or without it (``python tests/run.py``).
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import config, integration, portfolio, significance  # noqa: E402
from dss.clustering import performance  # noqa: E402

MOD = config.PROFILES["moderate"]


# ---------------------------------------------------------------- integration rules
def test_rules_are_read_from_config():
    d0, trail = integration.risk_budget(MOD, 0.5, "base", "Восстановление", 2.0)
    assert np.isclose(d0, 0.60 * (1 - 0.6 * 0.5))
    assert "0,6·P(рецессии)" in trail[1]
    soft = replace(config.RULES, rec_penalty=0.3)
    assert np.isclose(integration.risk_budget(MOD, 0.5, "base", "Восстановление", 2.0, soft)[0], 0.60 * 0.85)
    assert np.isclose(integration.fx_hedge(0.5), 0.15 + 0.6 * 0.5)
    assert integration.fx_hedge(1.0, replace(config.RULES, hedge_slope=2.0)) == config.RULES.hedge_max
    assert integration.fx_hedge(-1.0) == 0.0


def test_sensitivity_at_factor_one_reproduces_the_decision_rule():
    sens = integration.sensitivity_analysis(MOD, 0.3, "bear", "Замедление", -0.5, 0.4)
    co = sens["coefficients"]
    d = integration.risk_budget(MOD, 0.3, "bear", "Замедление", -0.5)[0]
    h = integration.fx_hedge(0.4)
    assert all(np.isclose(v, d if out == "доля рисковых" else h) for v, out in zip(co["×1"], co["Выход"]))
    # a heavier recession penalty lowers the risky share, a steeper hedge raises the hedge
    rec = co.loc["Штраф за P(рецессии), правило 1"]
    assert rec["×1,5"] < rec["×1"] < rec["×0,5"] and rec["Эластичность"] < 0
    assert co.loc["Наклон хеджа по индексу FX-риска, правило 2", "Эластичность"] > 0
    # the grid is monotone: down along P(recession), down from Bull to Stress
    g = sens["risk_grid"].to_numpy()
    assert (np.diff(g, axis=0) <= 1e-12).all() and (np.diff(g, axis=1) <= 1e-12).all()


def test_default_rules_are_monotone_and_violations_are_detected():
    mono = integration.sensitivity_analysis(MOD, 0.2, "base", "Расширение", 2.0, 0.5)["monotonicity"]
    assert (mono["При текущих коэффициентах"] == "да").all()
    broken = replace(config.RULES, scenario_mult=config.RULES.scenario_mult | {"stress": 1.2})
    mono = integration.sensitivity_analysis(MOD, 0.2, "base", "Расширение", 2.0, 0.5, broken)["monotonicity"]
    row = "Доля рисковых не растёт при ухудшении сценария (Bull → Stress)"
    assert mono.loc[row, "При текущих коэффициентах"] == "нет" and mono.loc[row, "Худший случай, п.п."] < 0


# ---------------------------------------------------------------- optimiser fallback
def test_optimizer_reports_failure_and_returns_a_feasible_portfolio():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 6))
    cov = A @ A.T / 6 * 0.04 + np.eye(6) * 0.01
    s = np.linspace(0.2, 1, 6)
    groups = np.array([0, 0, 1, 1, 2, 2])
    w, ok = portfolio.max_signal_to_risk(s, cov, 0.4, groups, None, 0.5, return_ok=True)
    assert ok and np.isclose(w.sum(), 1)
    bad = cov.copy()
    bad[0, 0] = np.nan                                   # the solver cannot evaluate the objective
    w, ok = portfolio.max_signal_to_risk(s, bad, 0.4, groups, None, 0.5, return_ok=True)
    assert not ok and np.isclose(w.sum(), 1) and (w >= 0).all() and (w <= 0.4 + 1e-9).all()
    for g in range(3):
        assert w[groups == g].sum() <= 0.5 + 1e-9


# ---------------------------------------------------------------- significance (FR-IV-13)
def _strategies(n=750, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0003, 0.01, n)
    return pd.DataFrame({"Equal Weighted": base, "Copy": base,
                         "Better": base + 0.0015, "Other": rng.normal(0.0003, 0.01, n)},
                        index=pd.bdate_range("2021-01-01", periods=n))


def test_bootstrap_sharpe_matches_backtest_metric_and_is_reproducible():
    df = _strategies()
    a, b = significance.sharpe_tests(df, 0.05, n_boot=500), significance.sharpe_tests(df, 0.05, n_boot=500)
    pd.testing.assert_frame_equal(a, b)                                     # seed 42: same numbers
    for col in df:
        assert np.isclose(a.loc[col, "sharpe"], performance(df[col], 0.05)["sharpe"])
        assert a.loc[col, "ci_low"] <= a.loc[col, "sharpe"] <= a.loc[col, "ci_high"]
    assert np.isnan(a.loc["Equal Weighted", "p_value"])
    assert a.loc["Copy", "diff"] == 0 and a.loc["Copy", "p_value"] == 1.0   # paired resampling
    assert a.loc["Better", "diff_low"] > 0 and a.loc["Better", "p_value"] < 0.01
    assert a.attrs["block"] == significance.block_length(750) == 9


def test_block_bootstrap_indices_are_circular_blocks():
    idx = significance.block_bootstrap_indices(20, n_boot=50, block=4, seed=0)
    assert idx.shape == (50, 20) and idx.min() >= 0 and idx.max() < 20
    first = idx[:, :4]
    assert (((first - first[:, :1]) % 20) == np.arange(4)).all()            # consecutive days, wrapping


def test_diebold_mariano_detects_a_better_forecast():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.01, 500)
    good, naive = (y - 0.8 * y) ** 2, y ** 2
    dm, p = significance.diebold_mariano(good, naive)
    assert dm < 0 and p < 0.001
    dm, p = significance.diebold_mariano(naive + rng.normal(0, 1e-8, 500), naive)
    assert p > 0.05
    assert np.isnan(significance.diebold_mariano(naive, naive)[0])          # zero loss differential


def test_forecast_tests_against_naive_forecast():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=300)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 4)), axis=0)), index=idx,
                          columns=list("abcd"))
    target = np.log(prices).diff().shift(-1).iloc[50:-1]
    preds = {"oracle": target * 0.9, "noise": target * 0 + rng.normal(0, 0.02, target.shape)}
    out = significance.forecast_tests(preds, prices, {"oracle": "Оракул"})
    assert out.loc["Оракул", "verdict"] == "точнее наивного" and out.loc["Оракул", "dm"] < 0
    assert out.loc["noise", "verdict"] == "хуже наивного"
    assert out.loc["Оракул", "n_days"] == len(target)


# ---------------------------------------------------------------- evaluation protocol (FR-IV-04)
def test_paper_protocol_is_one_chronological_split():
    from dss import signals
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2021-01-04", periods=400)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 4)), axis=0)), index=idx,
                          columns=list("abcd"))
    p = config.PROTOCOLS["paper"]
    r = signals.run(prices, ("ridge",), p["train_share"], p["retrain_every"])
    dates = r.predictions["ridge"].index
    assert np.isclose((dates[0] - idx[0]).days / (idx[-1] - idx[0]).days, 0.8, atol=0.05)
    assert r.predictions["ridge"].index.is_monotonic_increasing and len(dates) > 50
    wf = config.PROTOCOLS["walk_forward"]
    assert (wf["train_share"], wf["retrain_every"]) == (0.5, 126) and p["retrain_every"] == 0
