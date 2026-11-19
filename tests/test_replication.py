"""Experiment of article [A4] repeated on another market (dss.replication)."""
import numpy as np
import pandas as pd

from dss import config, replication
from dss.data import demo


def _prices(n_days=700):
    return demo.equity_prices().iloc[-n_days:]


def test_protocol_of_the_article_is_used():
    px = _prices()
    r = replication.run("kase", prices=px, local_rate=0.16)
    i = r.info
    assert i["train_share"] == config.TRAIN_SHARE == 0.80
    assert i["models"] == len(replication.ARTICLE_MODELS) == 7
    # one chronological split: the test sample is the last 20 % of the feature dates
    assert i["test_start"] > px.index[int(len(px) * 0.75)]
    assert i["bt_start"] > i["test_start"] - pd.Timedelta(days=1)
    assert set(replication.METRICS) <= set(r.strategies.columns)
    assert {"Signal Weighted", "Max Signal-to-Risk", "Min Variance", "Equal Weighted"} <= set(r.strategies.index)


def test_local_rate_only_changes_the_sharpe_ratio():
    px = _prices()
    r = replication.run("kase", prices=px, local_rate=0.16)
    s = r.strategies
    # Sharpe at the article's 5 % and at the local rate differ by (0.16 - 0.05) / volatility
    np.testing.assert_allclose(s["sharpe"] - s["sharpe_local"], (0.16 - config.RISK_FREE) / s["annual_vol"], rtol=1e-9)


def test_backtest_uses_only_the_test_sample():
    px = _prices()
    r = replication.run("kase", prices=px, local_rate=0.16)
    assert r.curves.index[0] >= r.info["test_start"]
    assert r.info["bt_days"] == len(r.curves) < 0.25 * len(px)


def test_kase_sets():
    assert replication.KASE_SETS["nine"]["drop"] == ()
    assert set(replication.KASE_SETS["seven"]["drop"]) == {"KSPI", "KMGZ"}
