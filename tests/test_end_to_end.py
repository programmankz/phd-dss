"""Tests of the point-in-time accounting of scripts/end_to_end.py: GDP
publication lag, weekly rebalancing dates, accrual of the defensive part,
transaction costs and the future-perturbation helper."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402


def _inputs(n=12):
    days = pd.bdate_range("2024-01-01", periods=n)
    prices = pd.DataFrame({"A": np.linspace(100, 111, n), "B": 50.0}, index=days)
    cal = pd.date_range(days[0] - pd.Timedelta(days=5), days[-1] + pd.Timedelta(days=5))
    fx = pd.DataFrame({"USD": np.linspace(450, 460, len(cal)), "EUR": 490.0, "CNY": 62.0, "RUB": 5.0}, index=cal)
    rates = pd.DataFrame({"kzt": 0.1460, "usd": 0.0365}, index=cal)   # 14.6 % and 3.65 % p.a.
    gdp = pd.DataFrame({2022: [3.0, 2.0], 2023: [4.0, 1.0]}, index=["Kazakhstan", "Other"])
    return e2e.Inputs(prices, fx["USD"].reindex(days), fx, gdp, rates, "test")


def test_published_year_uses_1_april():
    assert e2e.published_year(pd.Timestamp("2024-03-31")) == 2022
    assert e2e.published_year(pd.Timestamp("2024-04-01")) == 2023
    assert e2e.published_year(pd.Timestamp("2024-12-31")) == 2023


def test_weekly_dates_first_trading_day_of_week():
    days = pd.bdate_range("2024-01-01", "2024-01-31").drop(pd.Timestamp("2024-01-08"))   # Monday 8 Jan missing
    d = e2e.weekly_dates(days, "2024-01-03", "2024-01-31")
    assert list(d.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-09", "2024-01-15", "2024-01-22", "2024-01-29"]


def test_defensive_part_accrues_actual_365_and_fx_change():
    inp = _inputs()
    r = e2e.asset_returns(inp)
    mon = pd.Timestamp("2024-01-08")                  # Friday -> Monday: 3 calendar days
    assert np.isclose(r.loc[mon, e2e.KZT], 0.1460 * 3 / 365)
    tue = pd.Timestamp("2024-01-09")
    assert np.isclose(r.loc[tue, e2e.KZT], 0.1460 / 365)
    s = inp.usd
    assert np.isclose(r.loc[tue, e2e.USD], (1 + 0.0365 / 365) * s[tue] / s[tue - pd.Timedelta(days=1)] - 1)
    assert np.isclose(r.loc[tue, "A"], inp.prices["A"].pct_change()[tue])
    # lower attainable tenge yield
    assert np.isclose(e2e.asset_returns(inp, 0.02).loc[tue, e2e.KZT], 0.1260 / 365)


class _FixedEngine:
    """Engine stub: a fixed target every rebalancing date."""

    def __init__(self, days, weights):
        self.days, self.rebal = days, days[[0, 5]]
        self.start, self.end = days[0], days[-1]
        self.w = np.asarray(weights)

    def decide(self, t, cfg):
        return {"date": t, "d": 1.0, "h": 0.0, "weights": self.w.copy()}


def test_run_config_accounting_and_costs():
    inp = _inputs()
    rets = e2e.asset_returns(inp)
    cfg = e2e.Config("test", False, False, False)
    eng = _FixedEngine(inp.prices.index, [0.5, 0.5, 0.0, 0.0])
    s, log = e2e.run_config(eng, cfg, rets, tc_bps=10.0)
    # start fully in tenge -> turnover of the first trade is 1 (one-sided), costs 2 x 10 b.p.
    assert np.isclose(log["turnover"].iloc[0], 1.0)
    first = inp.prices.index[1]
    gross = 0.5 * rets.loc[first, "A"] + 0.5 * rets.loc[first, "B"]
    assert np.isclose(s[first], (1 - 2 * 10 / 1e4) * (1 + gross) - 1)
    # between rebalancing dates the weights drift with the prices (no costs)
    second = inp.prices.index[2]
    wa = 0.5 * (1 + rets.loc[first, "A"]) / (1 + gross)
    assert np.isclose(s[second], wa * rets.loc[second, "A"] + (1 - wa) * rets.loc[second, "B"])
    assert len(s) == len(inp.prices) - 1
    free, _ = e2e.run_config(eng, cfg, rets, tc_bps=0.0)
    assert (free >= s - 1e-12).all()


def test_perturbation_changes_only_the_future():
    inp = _inputs()
    pert = e2e.perturb_inputs(inp, "2024-01-10")
    before = inp.prices.index < "2024-01-10"
    pd.testing.assert_frame_equal(pert.prices[before], inp.prices[before])
    assert not np.allclose(pert.prices[~before], inp.prices[~before])
    assert (pert.fx_rates.loc[:"2024-01-09"] == inp.fx_rates.loc[:"2024-01-09"]).all().all()
    assert not np.allclose(pert.gdp[2023], inp.gdp[2023])
    pd.testing.assert_series_equal(pert.gdp[2022], inp.gdp[2022])
