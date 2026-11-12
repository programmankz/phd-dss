"""Tests of the Shapley decomposition of module contributions (dss.evaluation) and of the
Module III weights without signals (portfolio.cluster_equal_weights)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dss import evaluation, portfolio  # noqa: E402


def test_coalitions_count_and_names():
    c = evaluation.coalitions()
    assert len(c) == 16 and c[0] == frozenset() and c[-1] == frozenset(evaluation.MODULES)
    assert evaluation.coalition_name(frozenset()) == "none"
    assert evaluation.coalition_name(frozenset({"IV", "I"})) == "I+IV"


def test_shapley_additive_game_returns_own_contributions():
    own = {"I": 0.3, "II": -0.1, "III": 0.0, "IV": 0.5}
    v = {s: 1.0 + sum(own[p] for p in s) for s in evaluation.coalitions()}
    phi = evaluation.shapley(v)
    assert np.allclose(phi.to_numpy(), [own[p] for p in evaluation.MODULES])
    assert abs(evaluation.efficiency_gap(v, phi)) < 1e-12


def test_shapley_interaction_is_split_equally():
    # value 1 only when I and II are both present: each gets 1/2, the others 0
    v = {s: float({"I", "II"} <= s) for s in evaluation.coalitions()}
    phi = evaluation.shapley(v)
    assert np.allclose(phi[["I", "II"]], 0.5) and np.allclose(phi[["III", "IV"]], 0.0)
    # three players, glove game: v = 1 iff {A} and one of {B, C}
    pl = ("A", "B", "C")
    v3 = {s: float("A" in s and ("B" in s or "C" in s)) for s in evaluation.coalitions(pl)}
    phi3 = evaluation.shapley(v3, pl)
    assert np.allclose(phi3.to_numpy(), [2 / 3, 1 / 6, 1 / 6])


def test_shapley_table_on_synthetic_returns():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2023-01-02", periods=300)
    base = rng.normal(0.0004, 0.005, len(days))
    cols = {}
    for s in evaluation.coalitions(("I", "II")):
        # module I adds a constant 2 b.p. a day, module II does nothing
        cols[evaluation.coalition_name(s, ("I", "II"))] = base + (0.0002 if "I" in s else 0.0)
    daily = pd.DataFrame(cols, index=days)
    rf = pd.Series(0.0, index=days)
    t = evaluation.shapley_table(daily, rf, {"w": ("2023-02-01", "2023-03-01")}, ("I", "II"), n_boot=200)
    sh = t[t.metric == "sharpe"].set_index("module")["phi"]
    assert sh["I"] > 0 and abs(sh["II"]) < 1e-12
    assert t.loc[t.module == "efficiency gap", "phi"].abs().max() < 1e-10
    lo, hi = t[(t.metric == "sharpe") & (t.module == "I")][["ci_low", "ci_high"]].iloc[0]
    assert lo <= sh["I"] <= hi
    w = t[t.metric == "return w"].set_index("module")["phi"]
    assert w["I"] > 0 and np.isnan(t[(t.metric == "return w") & (t.module == "I")]["ci_low"].iloc[0])


def test_max_drawdown_metric_matches_curve():
    r = np.array([0.1, -0.2, 0.05, -0.1, 0.3])
    curve = np.cumprod(1 + r)
    expected = (curve / np.maximum.accumulate(curve) - 1).min()
    assert np.isclose(evaluation.max_drawdown_metric(r), expected)


def test_cluster_equal_weights_respects_limits():
    w = portfolio.cluster_equal_weights([0, 1, 2, 3], 0.3, 0.5)
    assert np.allclose(w, 0.25)
    g = np.array([0, 0, 0, 0, 0, 1, 2])
    w = portfolio.cluster_equal_weights(g, 0.2, 0.5)
    wm, cap, relaxed = portfolio.feasible_limits(g, 0.2, 0.5)
    assert relaxed and np.isclose(w.sum(), 1.0)
    assert (w <= wm + 1e-9).all()
    assert all(w[g == k].sum() <= cap + 1e-9 for k in np.unique(g))


def test_config_cluster_flag_defaults_to_m4():
    import end_to_end as e2e
    assert e2e.Config("a", True, True, True).use_clusters
    assert not e2e.Config("b", False, False, False).use_clusters
    assert not e2e.Config("c", True, True, True, clusters=False).use_clusters
    assert e2e.Config("d", False, False, False, clusters=True).use_clusters
