"""Conditional currency rule: peer gap uses only past values, the rule is off by default and
switches the expert dollar share on only above the threshold r_usd + E[delta] - r_kzt > 0."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
from dss import config, fx, integration  # noqa: E402


def _series():
    cal = pd.date_range("2014-01-01", "2015-12-31")
    usd = pd.Series(180.0, index=cal)                                  # managed rate: flat
    peers = pd.DataFrame({"RUB": 35.0, "NOK": 6.0, "BRL": 2.3, "CLP": 550.0}, index=cal)
    peers.loc["2014-12-01":, "RUB"] = 70.0                             # the rouble halves
    return usd, peers


def test_peer_gap_positive_when_peers_weaken_and_tenge_is_flat():
    usd, peers = _series()
    assert fx.peer_gap(usd, peers, "2014-11-01") == 0.0
    g = fx.peer_gap(usd, peers, "2015-02-01")
    assert np.isclose(g, np.log(2) / 4)                                # one of four peers doubled


def test_peer_gap_ignores_values_after_t_and_respects_lag():
    usd, peers = _series()
    t = pd.Timestamp("2014-12-05")
    assert fx.peer_gap(usd, peers, t) == 0.0                           # jump of 1 Dec is inside the 7-day lag
    changed = peers.copy()
    changed.loc[t + pd.Timedelta(days=1):] = 1.0
    assert fx.peer_gap(usd, changed, "2015-02-01") != fx.peer_gap(usd, peers, "2015-02-01")
    assert fx.peer_gap(usd, changed, t) == fx.peer_gap(usd, peers, t)


def test_peer_gap_falls_when_tenge_follows():
    usd, peers = _series()
    usd.loc["2015-01-01":] *= 1.5
    assert fx.peer_gap(usd, peers, "2015-02-01") < fx.peer_gap(_series()[0], peers, "2015-02-01")


def test_conditional_hedge_is_expert_rule_above_threshold_and_zero_below():
    r = config.RULES
    assert integration.conditional_fx_hedge(0.5, 0.02, 0.10, 0.05, r) == 0.0          # m = -3 %
    assert integration.conditional_fx_hedge(0.5, 0.02, 0.10, 0.08, r) == 0.0          # m = 0: not above
    assert integration.conditional_fx_hedge(0.5, 0.02, 0.10, 0.20, r) == integration.fx_hedge(0.5, r)
    assert integration.conditional_fx_hedge(0.5, 0.02, 0.10, float("nan"), r) == 0.0


def test_default_config_keeps_expert_rule():
    assert e2e.Config("x", True, True, False).fx_rule == "expert"


def test_engine_conditional_rule_point_in_time():
    days = pd.bdate_range("2014-06-02", "2015-06-30")
    prices = pd.DataFrame({"A": np.linspace(100, 120, len(days)), "B": 50.0}, index=days)
    cal = pd.date_range("2008-01-01", "2015-07-10")
    fxr = pd.DataFrame({"USD": 180.0, "EUR": 240.0, "CNY": 29.0, "RUB": 5.0}, index=cal)
    rates = pd.DataFrame({"kzt": 0.055, "usd": 0.001}, index=cal)
    gdp = pd.DataFrame({y: [5.0, 3.0] for y in range(2000, 2015)}, index=["Kazakhstan", "Other"])
    peers = pd.DataFrame({"RUB": 35.0, "NOK": 6.0, "BRL": 2.3, "CLP": 550.0}, index=cal)
    peers.loc["2014-12-01":] *= 1.6
    inp = e2e.Inputs(prices, fxr["USD"].reindex(days), fxr, gdp, rates, "test", None, peers)
    eng = e2e.Engine(inp, "moderate", "2014-06-02", "2015-06-30", need_m4=False, log=lambda *a: None)
    cfg = e2e.Config("cond", False, True, False, fx_rule="conditional")
    before, after = eng.decide(pd.Timestamp("2014-11-03"), cfg), eng.decide(pd.Timestamp("2015-02-02"), cfg)
    assert before["h"] == 0.0 and after["h"] > 0.0
    assert np.isclose(after["peer_gap"], np.log(1.6))
    # replacing everything after the decision date does not change the decision
    peers2 = peers.copy()
    peers2.loc["2014-11-04":] = 999.0
    inp2 = e2e.Inputs(prices, fxr["USD"].reindex(days), fxr, gdp, rates, "test", None, peers2)
    eng2 = e2e.Engine(inp2, "moderate", "2014-06-02", "2015-06-30", need_m4=False, log=lambda *a: None)
    assert eng2.decide(pd.Timestamp("2014-11-03"), cfg)["h"] == before["h"]


def test_macro_module_runs_on_a_vintage_before_the_default_split():
    from dss import macro
    rng = np.random.default_rng(0)
    years = list(range(1980, 2007))                                    # panel ends before MACRO_TRAIN_END_YEAR
    gdp = pd.DataFrame(rng.normal(3, 4, (40, len(years))), index=["Kazakhstan"] + [f"C{i}" for i in range(39)],
                       columns=years)
    m = macro.run(gdp, dtw=False)
    assert 0.0 <= m.recession_prob <= 1.0
