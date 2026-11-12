"""Tests of the derived integration rules d*, h* (dss.integration, rules="derived")."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import config, integration  # noqa: E402

PROF = config.PROFILES["moderate"]
WORSE = integration.RegimeMoments(mu_n=0.20, var_n=0.04, mu_r=-0.10, var_r=0.12)


def test_mixture_moments_limits_and_variance():
    assert np.allclose(integration.mixture_moments(0.0, WORSE), (0.20, 0.04))
    assert np.allclose(integration.mixture_moments(1.0, WORSE), (-0.10, 0.12))
    mu, var = integration.mixture_moments(0.5, WORSE)
    assert np.isclose(mu, 0.05) and np.isclose(var, 0.08 + 0.25 * 0.09)


def test_derived_d_monotone_when_adverse_is_worse():
    ds = [integration.derived_risk_budget(p, WORSE, 0.10, 3.0, PROF) for p in np.linspace(0, 1, 21)]
    assert all(b <= a + 1e-12 for a, b in zip(ds, ds[1:]))
    assert ds[0] > ds[-1] == PROF.min_risky_share


def test_derived_d_counterexample_when_adverse_has_higher_mean():
    # as on KASE: low-growth years had higher tenge returns -> d* rises with pi
    better = integration.RegimeMoments(mu_n=0.09, var_n=0.05, mu_r=0.28, var_r=0.20)
    d0, d1 = (integration.derived_risk_budget(p, better, 0.092, 5.0, PROF) for p in (0.0, 0.3))
    assert d1 > d0


def test_derived_d_bounds():
    for p in np.linspace(0, 1, 11):
        for g in (0.01, 1.0, 100.0):
            d = integration.derived_risk_budget(p, WORSE, 0.10, g, PROF)
            assert PROF.min_risky_share <= d <= PROF.max_risky_share
    # negative premium -> d_min for every gamma
    assert integration.derived_risk_budget(1.0, WORSE, 0.10, 0.01, PROF) == PROF.min_risky_share


def test_derived_h_zero_without_carry_and_bounds():
    assert integration.derived_fx_hedge(0.6, -0.02, 0.0, 0.01, 2.0) == 0.0
    assert integration.derived_fx_hedge(0.6, 0.0, 1e-12, 0.01, 2.0) == 0.0
    assert integration.derived_fx_hedge(1.0, 0.05, 0.0, 0.01, 2.0) == 0.0      # no defensive part
    h = integration.derived_fx_hedge(0.5, 0.01, 0.0, 0.02, 2.0)
    assert np.isclose(h, (0.01 / 2.0) / (0.5 * 0.02))
    assert integration.derived_fx_hedge(0.5, 1.0, 0.0, 0.01, 0.1) == config.RULES.hedge_max
    # a negative covariance of stocks with the dollar (stocks fall when the tenge weakens) raises the hedge
    assert integration.derived_fx_hedge(0.5, 0.01, -0.002, 0.02, 2.0) > h


def test_calibrate_gamma_attains_target_and_reports_bounds():
    cases = [(p, WORSE, 0.08) for p in (0.0, 0.05, 0.1, 0.2)]
    g = integration.calibrate_gamma(0.6, cases, PROF)
    md = np.mean([integration.derived_risk_budget(p, m, r, g, PROF) for p, m, r in cases])
    assert np.isclose(md, 0.6, atol=1e-4) and integration.calibration_attainable(0.6, cases, PROF, g)
    # half of the cases with a negative premium: mean d* <= 0.5 (0.95 + 0.10) < 0.6 for any gamma
    bad = cases[:2] + [(1.0, WORSE, 0.08)] * 2
    gb = integration.calibrate_gamma(0.6, bad, PROF)
    assert gb == 0.05 and not integration.calibration_attainable(0.6, bad, PROF, gb)


def test_regime_moments_split_and_minimum():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(np.r_[np.full(150, 0.001), np.full(150, -0.002)], index=idx)
    adv = pd.Series(np.r_[np.zeros(150), np.ones(150)].astype(bool), index=idx)
    m = integration.regime_moments(r, adv)
    assert np.isclose(m.mu_n, 0.252) and np.isclose(m.mu_r, -0.504) and m.n_n == m.n_r == 150
    assert integration.regime_moments(r, adv, min_days=200) is None


def test_derived_h_non_decreasing_in_expected_depreciation():
    hs = [integration.derived_fx_hedge(0.6, m, 0.001, 0.01, 3.0) for m in np.linspace(-0.05, 0.10, 16)]
    assert all(b >= a - 1e-12 for a, b in zip(hs, hs[1:]))
