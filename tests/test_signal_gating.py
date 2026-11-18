"""Tests of the Module IV additions: Pesaran-Timmermann test, IC t-statistic and daily IC,
signal admission (gate), turnover penalty and the rank-invariance lemma."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import portfolio, significance, signals  # noqa: E402


def test_pesaran_timmermann_detects_skill_and_its_absence():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 2000)
    good = y + rng.normal(0, 1.5, 2000)
    bad = rng.normal(0, 1, 2000)
    g, b = significance.pesaran_timmermann(good, y), significance.pesaran_timmermann(bad, y)
    assert g["p_value"] < 0.01 and g["hit_rate"] > 0.6
    assert b["p_value"] > 0.05
    # HAC version agrees in direction
    assert significance.pesaran_timmermann(good, y, nw_lag=5)["p_value"] < 0.01


def test_pt_base_rate_versus_half_z_test():
    # always-up forecast in a rising market: 60 % hits, z-test against 1/2 "significant",
    # PT undefined/insignificant because the forecast has no variation
    rng = np.random.default_rng(1)
    y = np.where(rng.random(1000) < 0.6, 1.0, -1.0)
    r = significance.pesaran_timmermann(np.ones(1000), y)
    assert r["hit_rate"] > 0.55 and r["p_half"] < 0.01
    assert not (r["p_value"] < 0.05)


def test_overlap_inflates_classic_pt():
    # 21-day overlapping sums of noise and of an unrelated noise: classic PT over-rejects, HAC much less
    rng = np.random.default_rng(2)
    rej_classic = rej_hac = 0
    for _ in range(60):
        e, u = rng.normal(size=1300), rng.normal(size=1300)
        y = pd.Series(e).rolling(21).sum().to_numpy()[21:]
        f = pd.Series(u).rolling(21).sum().to_numpy()[21:]
        rej_classic += significance.pesaran_timmermann(f, y)["p_value"] < 0.05
        rej_hac += significance.pesaran_timmermann(f, y, nw_lag=20)["p_value"] < 0.05
    assert rej_classic > rej_hac and rej_hac <= 12


def test_ic_t_stat_and_daily_ic():
    t, n = significance.ic_t_stat([0.1, 0.2, 0.0, 0.1, np.nan])
    assert n == 4 and np.isclose(t, 0.1 * 2 / np.std([0.1, 0.2, 0.0, 0.1], ddof=1))
    d = pd.bdate_range("2024-01-01", periods=3)
    sc = pd.DataFrame([[1, 2, 3], [3, 2, 1], [1, 2, np.nan]], index=d, columns=list("abc"), dtype=float)
    y = pd.DataFrame([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], index=d, columns=list("abc"))
    ic = signals.daily_ic(sc, y)
    assert np.isclose(ic.iloc[0], 1.0) and np.isclose(ic.iloc[1], -1.0) and np.isnan(ic.iloc[2])


def test_turnover_penalty_moves_less():
    cov = np.diag([0.04, 0.05, 0.06, 0.05])
    s = np.array([0.25, 0.5, 0.75, 1.0])
    prev = np.array([0.4, 0.3, 0.2, 0.1])
    w0 = portfolio.max_signal_to_risk(s, cov, 0.5)
    wk = portfolio.max_signal_to_risk(s, cov, 0.5, w_prev=prev, kappa=2.0)
    assert np.abs(wk - prev).sum() < np.abs(w0 - prev).sum() - 1e-3
    assert np.allclose(portfolio.max_signal_to_risk(s, cov, 0.5, w_prev=prev, kappa=0.0), w0)


def test_rank_invariance_lemma():
    """The optimiser sees only ranks: a strictly increasing transform of all forecasts (common to all
    assets) leaves the signals and therefore the weights unchanged."""
    rng = np.random.default_rng(3)
    pred = pd.DataFrame(rng.normal(0, 0.01, (5, 6)))
    for f in (lambda x: np.exp(50 * x), lambda x: 3 * x + 1, lambda x: np.arctan(100 * x)):
        pd.testing.assert_frame_equal(signals.rank_signal(pred), signals.rank_signal(f(pred)))
    a = rng.normal(size=(200, 6)) * 0.01
    cov = np.cov(a.T) * 252
    s1 = signals.rank_signal(pred).iloc[-1].to_numpy()
    s2 = signals.rank_signal(np.exp(50 * pred)).iloc[-1].to_numpy()
    assert np.allclose(portfolio.max_signal_to_risk(s1, cov, 0.4), portfolio.max_signal_to_risk(s2, cov, 0.4))


def test_gate_parameters_fixed():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import end_to_end as e2e
    assert e2e.GATE_T == 1.645 and e2e.GATE_WINDOW == 52
    assert not e2e.Config("x", True, True, True).gate and e2e.Config("x", True, True, True).kappa == 0.0
