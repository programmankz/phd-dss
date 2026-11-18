"""Tests of the KZ-MacroFin dataset (dss.dataset): no look-ahead in the features, label definitions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import dataset  # noqa: E402
from dss.data import loaders  # noqa: E402


def _inputs():
    need = ("index_kase.csv", "brent.csv", "macro_monthly_kz.csv", "fx_commodity_panel.csv", "nbk_base_rate.csv")
    if not all((loaders.data_dir() / f).exists() for f in need):
        pytest.skip("KZ-MacroFin inputs are not in data/")
    return dataset.load_inputs()


def test_features_do_not_use_the_future():
    inp = _inputs()
    cut = pd.Timestamp("2016-06-30")
    full = dataset.build(inp).table
    trunc = dataset.build(inp, end=cut).table
    common = trunc.index[trunc.index <= cut - pd.Timedelta(days=7)]
    feats = [c for c in trunc.columns if not c.startswith(("y_", "fwd_"))]
    pd.testing.assert_frame_equal(full.loc[common, feats], trunc.loc[common, feats], check_freq=False,
                                  check_dtype=False)
    asof = [c for c in feats if c.endswith("_asof")]
    for c in asof:
        d = pd.to_datetime(full[c])
        assert (d.dropna() <= d.dropna().index).all(), c


def test_devaluation_detector():
    days = pd.bdate_range("2015-08-03", periods=30)
    usd = pd.Series(200.0, index=days)
    usd.iloc[10:] = 255.0                       # +27.5 % in a day (20.08.2015-like)
    usd.iloc[20:] = 255.0 * 1.03                # +3 % in a day: no episode
    assert list(dataset.devaluation_days(usd)) == list(days[10:15])   # 1-day jump and the 5-day window after it


def test_detection_lag():
    idx = pd.RangeIndex(10)
    y = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], index=idx, dtype=float)
    p = pd.Series([0, 0.2, 0.7, 0.9, 0, 0, 0.1, 0.2, 0, 0], index=idx)
    lag, det, n = dataset.detection_lag(y, p)
    assert (lag, det, n) == (1.0, 1, 2)
    assert np.isnan(dataset.detection_lag(y, p * 0)[0])
