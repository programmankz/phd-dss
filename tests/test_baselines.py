"""Tests of the baseline portfolio rules (dss.baselines) and of the deflated Sharpe ratio."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import baselines, significance  # noqa: E402

COV = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.01], [0.0, 0.01, 0.16]])


def test_erc_equalises_risk_contributions():
    w = baselines.erc(COV)
    assert np.isclose(w.sum(), 1.0) and (w > 0).all()
    assert np.allclose(baselines.risk_contributions(w, COV), 1 / 3, atol=1e-4)


def test_hrp_with_uncorrelated_assets_is_inverse_variance():
    cov = np.diag([0.04, 0.09, 0.16, 0.25])
    w = baselines.hrp(cov)
    ivp = 1 / np.diag(cov)
    # recursive bisection of inverse-variance portfolios equals inverse variance for a diagonal matrix
    assert np.allclose(w, ivp / ivp.sum())


def test_caps_are_respected():
    for rule in ("min_variance", "erc", "hrp", "markowitz", "black_litterman"):
        rng = np.random.default_rng(1)
        win = pd.DataFrame(rng.normal(0.001, [0.01, 0.02, 0.03, 0.01, 0.015], (300, 5)))
        w = baselines.weights(rule, win, 0.3, ranks=np.linspace(0, 1, 5))
        assert np.isclose(w.sum(), 1.0) and (w >= -1e-12).all() and (w <= 0.3 + 1e-6).all(), rule


def test_black_litterman_without_views_returns_the_prior():
    flat = np.full(3, 0.5)
    mu = baselines.black_litterman_mu(COV, flat)
    pi = baselines.BL_DELTA * COV @ np.full(3, 1 / 3)
    assert np.allclose(mu, pi)
    w = baselines.mean_variance(mu, COV, baselines.BL_DELTA)
    assert np.allclose(w, 1 / 3, atol=1e-4)
    up = baselines.black_litterman_mu(COV, np.array([1.0, 0.5, 0.0]))
    assert up[0] > pi[0] and up[2] < pi[2]


def test_markowitz_falls_back_to_min_variance():
    from dss import portfolio
    assert np.allclose(baselines.markowitz(np.array([-0.1, -0.2, -0.05]), COV), portfolio.min_variance(COV))
    w = baselines.markowitz(np.array([0.10, 0.0, 0.0]), COV)
    assert w[0] > 0.99


def test_deflated_sharpe_decreases_with_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0008, 0.01, 900)
    trials = rng.normal(0.03, 0.02, 20)
    d1, s1 = significance.deflated_sharpe(r, 1)
    d20, s20 = significance.deflated_sharpe(r, 20, trials)
    assert 0 <= d20 < d1 <= 1 and s1 == 0 and s20 > 0
