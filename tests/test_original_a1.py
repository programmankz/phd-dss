"""Authors' notebook of article [A1] (dss.original_a1)."""
import numpy as np
import pandas as pd
import pytest

from dss import original_a1 as oa
from dss.data.loaders import data_dir


def _panel(n=40, years=range(1980, 2025), seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(3, 4, (n, len(years))), index=[f"C{i:02d}" for i in range(n)], columns=list(years))


def test_same_year_features_contain_the_predicted_year():
    w = _panel()
    d, p = oa.same_year_dataset(w), oa.same_year_dataset(w, past_only=True)
    row = d[(d.country_name == "C00") & (d.year == 2000)].iloc[0]
    assert row["rolling_mean3"] == pytest.approx(w.loc["C00", [1998, 1999, 2000]].mean())    # as in the notebook
    row = p[(p.country_name == "C00") & (p.year == 2000)].iloc[0]
    assert row["rolling_mean3"] == pytest.approx(w.loc["C00", [1997, 1998, 1999]].mean())    # control without year t


def test_leak_explains_the_accuracy():
    w = _panel(60)
    leak, clean = oa.same_year_classification(w), oa.same_year_classification(w, past_only=True)
    # independent noise cannot be predicted from the past; with year t among the predictors it can
    assert leak["roc_auc"].min() > 0.9 and clean["roc_auc"].max() < 0.65


def test_ar3_on_differences_recovers_a_linear_trend():
    y = 2.0 + 0.5 * np.arange(30)
    coef = oa.ar_fit(y)
    np.testing.assert_allclose(oa.ar_forecast(y, coef), y[-1] + 0.5 * np.arange(1, 6), atol=1e-8)
    np.testing.assert_allclose(oa.prophet_style(y), y[-1] + 0.5 * np.arange(1, 6), atol=1e-6)
    np.testing.assert_allclose(oa.lstm_style(y), y[-1] + 0.5 * np.arange(1, 6), atol=1e-6)


def test_next_year_target_and_split():
    d = oa.next_year_dataset(_panel())
    r = d[(d.country_name == "C01") & (d.year == 2010)].iloc[0]
    nxt = d[(d.country_name == "C01") & (d.year == 2011)].iloc[0]
    assert r["target_reg"] == pytest.approx(nxt["gdp_growth"]) and d["year"].max() == 2023
    assert set(oa.NEXT_YEAR_FEATURES) <= set(d.columns) and len(oa.NEXT_YEAR_FEATURES) == 24


@pytest.mark.skipif(not (data_dir() / "world_gdp_data.csv").exists(), reason="the article's data file is absent")
def test_notebook_numbers_on_the_article_data():
    w = oa.load_article_data()
    kz = oa.country_forecasts(w)
    for name, values in oa.NOTEBOOK["kz"].items():
        np.testing.assert_allclose(kz[name], values, atol=0.006)
    st, sil, outliers = oa.clusters(w)
    assert st["cluster"].value_counts().to_dict() == oa.NOTEBOOK["clusters"]
    assert outliers == oa.NOTEBOOK["dbscan_outliers"] and sil.idxmax() == oa.NOTEBOOK["best_k"]
    acc = oa.same_year_classification(w)["accuracy"]
    assert acc["Gradient Boosting"] == pytest.approx(0.9793, abs=0.003)
