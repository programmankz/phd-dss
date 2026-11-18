"""Investment attractiveness: targets, the KASE return window and the absence of look-ahead."""
import numpy as np
import pandas as pd
import pytest

from dss import attractiveness as att
from dss.data import loaders


def _index(start="1999-01-01", end="2026-06-30", seed=0):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days)))), index=days)


def _annual(n=24, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(4 + rng.normal(0, 3, n), index=pd.period_range("2001", periods=n, freq="Y"))


def test_targets_follow_the_definitions():
    g = pd.Series([2.0, -1.0, 0.5, 3.0], index=pd.period_range("2020", periods=4, freq="Y"))
    assert att.target(g, "sign").tolist() == [1.0, 0.0, 1.0, 1.0]
    accel = att.target(g, "accel")
    assert np.isnan(accel.iloc[0]) and accel.iloc[1:].tolist() == [0.0, 1.0, 1.0]


def test_kase_window_ends_before_the_period():
    idx = pd.Series([100.0, 110.0, 110.0, 121.0, 500.0],
                    index=pd.to_datetime(["2020-06-30", "2020-12-30", "2020-12-31", "2021-06-30", "2021-07-01"]))
    r = att.kase_return(idx, pd.PeriodIndex(["2021", "2021Q3"], freq="Q"), 6)
    assert r[pd.Period("2021Q1", "Q")] == pytest.approx(10.0)       # 30.06.2020 -> 31.12.2020
    assert r[pd.Period("2021Q3", "Q")] == pytest.approx(10.0)       # the value of 01.07.2021 is not used


def test_forecasts_do_not_use_the_future():
    g, idx = _annual(), _index()
    data = att.build_dataset("accel", "annual", g, idx)
    probs, means = att.walk_forward("accel", data, g, 10)
    g2, idx2 = g.copy(), idx.copy()
    cut = probs.index[3]
    g2[g2.index >= cut] = 99.0                                       # change the period itself and all later ones
    idx2[idx2.index >= cut.start_time] = 1.0
    data2 = att.build_dataset("accel", "annual", g2, idx2)
    probs2, means2 = att.walk_forward("accel", data2, g2, 10)
    pd.testing.assert_series_equal(probs.loc[cut], probs2.loc[cut])
    pd.testing.assert_series_equal(means.loc[cut], means2.loc[cut])


def test_run_returns_a_complete_result_on_synthetic_data():
    r = att.run("accel", "annual", growth=_annual(), index=_index())
    assert set(att.REG_MODELS) <= set(r.metrics.index) and att.BASE in r.metrics.index
    assert r.metrics["Точность"].between(0, 1).all() and r.probs.stack().between(0, 1).all()
    assert r.signal["available"] and r.signal["value"] in (0, 1)
    assert len(r.coefficients) == 3 and (r.coefficients["Ст. ошибка (HAC)"] > 0).all()


def test_an_informative_index_is_detected():
    rng = np.random.default_rng(3)
    idx = _index(seed=5)
    periods = pd.period_range("2001", periods=24, freq="Y")
    k = att.kase_return(idx, periods, 6)
    g = pd.Series(3 + 0.2 * k.to_numpy() + rng.normal(0, 0.5, len(k)), index=k.index)
    r = att.run("sign", "annual", growth=g, index=idx)
    assert r.coefficients.loc[att.FEATURE_LABELS["kase"], "p"] < 0.01
    assert r.metrics.loc["AR(1) + KASE", "RMSE роста, п.п."] < r.metrics.loc["AR(1)", "RMSE роста, п.п."]


def test_heteroskedasticity_tests_react_to_changing_variance():
    rng = np.random.default_rng(7)
    x = np.linspace(1, 5, 400)
    assert att.breusch_pagan(rng.normal(0, 1, 400) * x, x)[1] < 0.01
    assert att.breusch_pagan(rng.normal(0, 1, 400), x)[1] > 0.01


def test_parse_quarterly_gdp_sheet():
    sheet = pd.DataFrame([[None] * 6, ["title"] + [None] * 5, [None] * 6,
                          ["ОКЭД", "наименование ОКЭД", 2011, None, None, None],
                          [None, None, "I квартал", "II квартал", "III квартал", "IV квартал"],
                          [None, "Валовая добавленная стоимость", 105, 106, 107, 108],
                          [None, "Валовой внутренний продукт", 106.2, 116.3, 109.0, 99.4]])
    s = loaders.parse_stat_quarterly_gdp({"3": sheet})
    assert list(s.index.astype(str)) == ["2011Q1", "2011Q2", "2011Q3", "2011Q4"]
    assert s.iloc[0] == pytest.approx(6.2) and s.iloc[-1] == pytest.approx(-0.6)
