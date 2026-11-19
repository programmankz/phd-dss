"""Authors' pipeline of article [A4] (dss.original_a4)."""
import numpy as np
import pandas as pd
import pytest

from dss import original_a4 as oa


def _bars(n=400, seed=0, drift=0.0005):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame({"Date": pd.bdate_range("2021-01-04", periods=n), "Open": close * 0.999, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close, "Volume": rng.integers(1_000, 5_000, n).astype(float)})


def test_features_follow_the_notebook():
    raw = _bars()
    f = oa.add_features(raw)
    assert list(f.columns[-7:]) == ["SMA_14", "Log_Ret", "RSI_14", "MACD", "ATR_14", "BB_Width", "OBV"]
    assert len(f) == len(raw) - 19            # the 20-day Bollinger window is the longest one
    c = raw["Close"]
    assert f["SMA_14"].iloc[-1] == pytest.approx(c.iloc[-14:].mean())
    assert f["BB_Width"].iloc[-1] == pytest.approx(4 * c.iloc[-20:].std() / c.iloc[-20:].mean())
    assert f["RSI_14"].between(0, 100).all()


def test_stage_a_uses_60_day_windows_and_a_chronological_split():
    f = oa.add_features(_bars())
    X, y = oa._sequences(f)
    assert X.shape[1:] == (oa.LOOKBACK, len(oa.FEATURES)) and len(X) == len(y) == len(f) - 1 - oa.LOOKBACK
    # as in the notebook: a window of rows 0..59 is paired with the return from day 60 to day 61
    assert y[0] == pytest.approx(f["Close"].iloc[oa.LOOKBACK + 1] / f["Close"].iloc[oa.LOOKBACK] - 1)
    table, stable = oa.stage_a_stock(f)
    assert set(table["Model"]) == set(oa.sklearn_models()) == set(stable)
    assert table["Combined_Score"].is_monotonic_decreasing
    r = table.iloc[0]
    assert r["Combined_Score"] == pytest.approx(0.6 * r["Dir_Acc"] + 0.4 * (r["R2"] + 1) * 50)


def test_portfolios_of_stage_b():
    processed = {f"S{k}": oa.add_features(_bars(seed=k, drift=0.0002 * k)) for k in range(5)}
    summary = pd.DataFrame({"Stock": list(processed), "Best_Model": "Ridge", "Predicted_Return": [0.003, -0.001, 0.002, 0.0, 0.001]})
    r = oa.run(processed, summary=summary)
    w = r.weights
    assert list(w.columns) == list(oa.STRATEGIES)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all().all() and (w <= 1 + 1e-9).all().all()
    # Signal Weighted is proportional to the rank of the forecast, Equal Weighted is 1/N
    np.testing.assert_allclose(w["Signal Weighted"], r.summary.set_index("Stock")["Rank_Score"] / 3.0)
    assert w["Signal Weighted"].idxmax() == "S0" and w["Equal Weighted"].nunique() == 1
    prices = oa.common_prices(processed, w.index)
    cov = prices.pct_change().dropna().cov().to_numpy()
    var = lambda c: w[c].to_numpy() @ cov @ w[c].to_numpy()   # noqa: E731
    assert var("Min Variance") <= min(var(c) for c in oa.STRATEGIES) + 1e-12


def test_backtest_holds_the_weights_over_the_whole_sample():
    processed = {f"S{k}": oa.add_features(_bars(seed=k)) for k in range(4)}
    summary = pd.DataFrame({"Stock": list(processed), "Best_Model": "Ridge", "Predicted_Return": [0.1, 0.2, 0.3, 0.4]})
    r = oa.run(processed, summary=summary)
    prices = oa.common_prices(processed, r.weights.index)
    daily = prices.pct_change().dropna() @ r.weights["Equal Weighted"]
    t = r.table.loc["Equal Weighted"]
    assert r.info["days"] == len(daily) == len(r.curves)
    assert t["total_return"] == pytest.approx((1 + daily).prod() - 1)
    assert t["sharpe"] == pytest.approx((t["annual_return"] - oa.RISK_FREE) / t["annual_vol"])
    # the second table uses the same weights on the last 20 % of the dates only
    assert r.info["test_start"] == prices.index[int(len(prices) * 0.8)]
    assert r.test_table.loc["Equal Weighted", "total_return"] == pytest.approx((1 + daily.loc[r.info["test_start"]:]).prod() - 1)


def test_kase_run_with_stored_stage_a():
    if not (oa.results_dir() / oa.STAGE_A_FILES[7]).exists():
        pytest.skip("stage A results are not stored")
    r = oa.run_kase("seven", 7)
    assert r.info["assets"] == 7 and not {"KSPI", "KMGZ"} & set(r.weights.index)
    assert r.info["start"].year == 2020 and set(r.table.index) == set(oa.STRATEGIES)
