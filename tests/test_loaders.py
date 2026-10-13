"""Tests of the data layer: user CSV files are validated and every malformed
file yields a readable message (file, cause, action) instead of a traceback.

Run with pytest (``pytest -q``) or without it (``python tests/run.py``).
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss.data import DataError, loaders  # noqa: E402


@contextmanager
def data_dir(files: dict):
    """Temporary DSS_DATA_DIR holding ``{file name: text}``."""
    old = os.environ.get("DSS_DATA_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items():
            Path(tmp, name).write_text(text, encoding="utf-8")
        os.environ["DSS_DATA_DIR"] = tmp
        try:
            yield Path(tmp)
        finally:
            if old is None:
                os.environ.pop("DSS_DATA_DIR", None)
            else:
                os.environ["DSS_DATA_DIR"] = old


def raises(fn, *fragments):
    """Assert ``fn`` raises DataError whose message contains every fragment."""
    try:
        fn()
    except DataError as exc:
        msg = str(exc)
        for f in fragments:
            assert f in msg, f"{f!r} not in {msg!r}"
        return msg
    raise AssertionError("DataError was not raised")


def prices_csv(n_days=320, n_assets=6, date_fmt="%Y-%m-%d") -> str:
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_assets)), axis=0))
    df = pd.DataFrame(px, index=idx.strftime(date_fmt), columns=[f"T{i}" for i in range(n_assets)])
    df.index.name = "Date"
    return df.to_csv()


def fx_csv(years=6) -> str:
    idx = pd.date_range("2015-01-01", periods=365 * years, freq="D")
    t = np.arange(len(idx))
    df = pd.DataFrame({"USD": 180 + 0.1 * t, "EUR": 200 + 0.1 * t, "CNY": 30 + 0.01 * t, "RUB": 5 + 0.001 * t},
                      index=idx)
    df.index.name = "Date"
    return df.to_csv()


def gdp_csv(countries=("Kazakhstan", "Chile"), years=range(2000, 2021)) -> str:
    df = pd.DataFrame(3.0, index=pd.Index(countries, name="country"), columns=list(years))
    return df.to_csv()


# ------------------------------------------------------------------ prices
def test_valid_price_csv_is_loaded_and_forward_filled():
    df = pd.read_csv(pd.io.common.StringIO(prices_csv()))
    df.loc[5, "T0"] = np.nan                              # a missing quote mid-series
    df = df.iloc[::-1]                                    # and rows in reverse order
    with data_dir({"prices_kase.csv": df.to_csv(index=False)}):
        ds = loaders.load_prices("kase")
    assert not ds.is_demo and ds.source == "CSV: prices_kase.csv"
    assert ds.frame.shape == (320, 6) and ds.frame.notna().all().all()
    assert ds.frame.index.is_monotonic_increasing
    assert ds.frame["T0"].iloc[5] == ds.frame["T0"].iloc[4]   # carried forward


def test_missing_date_column_names_file_and_action():
    text = prices_csv().replace("Date,", "Дата,", 1)
    with data_dir({"prices_kase.csv": text}):
        msg = raises(lambda: loaders.load_prices("kase"), "prices_kase.csv", "нет столбца Date",
                     "проверьте заголовок")
    assert "Traceback" not in msg


def test_unparseable_dates_are_reported():
    text = prices_csv().replace("2021-01-06", "вчера", 1)
    with data_dir({"prices_kase.csv": text}):
        raises(lambda: loaders.load_prices("kase"), "«вчера»", "ГГГГ-ММ-ДД")


def test_russian_dotted_dates_are_read_day_first():
    with data_dir({"prices_kase.csv": prices_csv(date_fmt="%d.%m.%Y")}):
        df = loaders.load_prices("kase").frame
    assert df.index[0] == pd.Timestamp("2021-01-04") and df.index[1] == pd.Timestamp("2021-01-05")


def test_duplicate_dates_are_rejected():
    lines = prices_csv().splitlines()
    text = "\n".join(lines[:3] + [lines[2]] + lines[3:]) + "\n"
    with data_dir({"prices_kase.csv": text}):
        raises(lambda: loaders.load_prices("kase"), "2021-01-05", "несколько раз")


def test_non_numeric_column_is_reported():
    df = pd.read_csv(pd.io.common.StringIO(prices_csv()))
    df["T3"] = "н/д"
    with data_dir({"prices_kase.csv": df.to_csv(index=False)}):
        raises(lambda: loaders.load_prices("kase"), "«T3»", "нет ни одного числа")


def test_non_positive_prices_are_rejected():
    df = pd.read_csv(pd.io.common.StringIO(prices_csv()))
    df.loc[10, "T2"] = -1.0
    with data_dir({"prices_kase.csv": df.to_csv(index=False)}):
        raises(lambda: loaders.load_prices("kase"), "«T2»", "положительными")


def test_too_few_assets_or_days():
    with data_dir({"prices_kase.csv": prices_csv(n_assets=3)}):
        raises(lambda: loaders.load_prices("kase"), "3 бумаг", f"не меньше {loaders.MIN_ASSETS}")
    with data_dir({"prices_kase.csv": prices_csv(n_days=100)}):
        raises(lambda: loaders.load_prices("kase"), "100 торговых дней", str(loaders.MIN_PRICE_DAYS))


def test_empty_file_is_reported():
    with data_dir({"prices_kase.csv": ""}):
        raises(lambda: loaders.load_prices("kase"), "prices_kase.csv", "файл пуст")
    with data_dir({"prices_kase.csv": "Date,A,B\n"}):
        raises(lambda: loaders.load_prices("kase"), "нет строк с данными")


# ------------------------------------------------------------------ FX
def test_fx_requires_all_four_currencies_and_five_years():
    with data_dir({"fx_rates.csv": fx_csv()}):
        ds = loaders.load_fx()
    assert list(ds.frame.columns) == ["USD", "EUR", "CNY", "RUB"] and not ds.is_demo
    with data_dir({"fx_rates.csv": fx_csv().replace(",RUB", ",RUR", 1)}):
        raises(lambda: loaders.load_fx(), "fx_rates.csv", "нет столбцов RUB")
    with data_dir({"fx_rates.csv": fx_csv(years=2)}):
        raises(lambda: loaders.load_fx(), "короче 5 лет", "fetch fx")


# ------------------------------------------------------------------ GDP
def test_gdp_panel_is_validated():
    with data_dir({"gdp_growth.csv": gdp_csv()}):
        ds = loaders.load_gdp()
    assert ds.frame.index.name == "country" and 2014 in ds.frame.columns and not ds.is_demo
    with data_dir({"gdp_growth.csv": gdp_csv(countries=("Chile", "Peru"))}):
        raises(lambda: loaders.load_gdp(), "нет строки «Kazakhstan»")
    with data_dir({"gdp_growth.csv": gdp_csv().replace(",2005,", ",y2005,", 1)}):
        raises(lambda: loaders.load_gdp(), "«y2005» не является годом")
    with data_dir({"gdp_growth.csv": gdp_csv(years=range(2000, 2014))}):
        raises(lambda: loaders.load_gdp(), "не покрывают", "2015")
    with data_dir({"gdp_growth.csv": "country\nKazakhstan\n"}):
        raises(lambda: loaders.load_gdp(), "(нет)")


# ------------------------------------------------------------------ fundamentals
def test_fundamentals_require_ticker_column():
    with data_dir({"fundamentals_kase.csv": "ticker,sector,PB,PE\nHSBK,Банки,1.1,5\nKSPI,Финтех,9,x\n"}):
        df = loaders.load_fundamentals("kase").frame
    assert list(df.index) == ["HSBK", "KSPI"] and np.isnan(df.loc["KSPI", "PE"])
    with data_dir({"fundamentals_kase.csv": "Тикер,PB\nHSBK,1.1\n"}):
        raises(lambda: loaders.load_fundamentals("kase"), "нет столбца ticker")


def test_no_user_files_means_demo():
    with data_dir({}):
        assert loaders.load_prices("kase").is_demo and loaders.load_fx().is_demo and loaders.load_gdp().is_demo


# ------------------------------------------------------------------ live sources
def test_gdp_fetch_falls_back_to_world_bank_when_imf_refuses():
    import json
    import urllib.error

    countries = [{"id": "KAZ", "name": "Kazakhstan", "region": {"id": "ECS"}},
                 {"id": "WLD", "name": "World", "region": {"id": "NA"}}]
    obs = [{"countryiso3code": iso, "date": str(y), "value": v}
           for iso in ("KAZ", "WLD") for y, v in ((2014, 4.2), (2015, 1.2), (2016, None))]

    def fake_get(url, timeout=30):
        if "imf.org" in url:
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        return json.dumps([{}, countries if "/indicator/" not in url else obs]).encode()

    real_get, loaders._get = loaders._get, fake_get
    try:
        df, source = loaders.fetch_gdp()
    finally:
        loaders._get = real_get
    assert "World Bank" in source
    assert list(df.index) == ["Kazakhstan"]            # aggregate "World" dropped
    assert df.loc["Kazakhstan", 2015] == 1.2 and 2016 not in df.columns


def test_demerger_is_back_adjusted_not_a_loss():
    idx = pd.bdate_range("2025-10-10", periods=4)            # ex-date is the 3rd day
    px = pd.DataFrame({"T": [100.0, 102.0, 60.0, 61.2], "U": [10.0, 11.0, 12.0, 13.0]}, index=idx)
    adj = loaders.adjust_demergers(px, {"T": idx[2], "MISSING": idx[2]})
    r = adj.pct_change().iloc[1:]
    assert abs(r.loc[idx[2], "T"]) < 1e-12                    # no -41% jump on the ex-date
    assert np.allclose(r["T"].values[[0, 2]], [0.02, 0.02])   # other returns unchanged
    assert adj["U"].equals(px["U"]) and adj.loc[idx[2]:, "T"].equals(px.loc[idx[2]:, "T"])


def test_kase_fetch_reads_chart_feed_and_skips_unknown_tickers():
    import json
    import urllib.error

    def fake_get(url, timeout=30):
        if "symbol=NOPE" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if "symbol=EMPTY" in url:
            return json.dumps({"s": "no_data"}).encode()
        day = 86400 * 19000                                   # 2022-01-08 .. 2022-01-10
        t = [day, day + 86400 * 2] if "symbol=A" in url else [day + 86400, day + 86400 * 2]
        return json.dumps({"s": "ok", "t": t, "c": [10.0, 11.0]}).encode()

    real_get, loaders._get = loaders._get, fake_get
    try:
        px = loaders.fetch_kase(["A", "B", "NOPE", "EMPTY"])
    finally:
        loaders._get = real_get
    assert list(px.columns) == ["A", "B"] and px.index.name == "Date"
    assert px["A"].tolist() == [10.0, 10.0, 11.0]            # gap day forward-filled
    assert np.isnan(px["B"].iloc[0])                          # no back-fill before the first trade


# ------------------------------------------------------------------ corporate actions
def test_corporate_actions_file_adjusts_any_universe():
    px = prices_csv(n_days=320, n_assets=6)
    df = pd.read_csv(pd.io.common.StringIO(px), index_col=0)
    ex = df.index[200]
    df.loc[ex:, "T0"] *= 0.5                                   # a 2:1 split without adjustment
    df.loc[ex:, "T1"] *= 0.6                                   # a spin-off
    actions = (f"ticker,ex_date,method,factor,note\nT0,{ex},factor,0.5,split 2:1\nT1,{ex},ratio,,spin-off\n"
               f"OTHER.NS,{ex},ratio,,ticker of another universe\nT2,1990-01-01,factor,2,before the data\n")
    with data_dir({"prices_nse.csv": df.to_csv(), "corporate_actions.csv": actions}):
        ds = loaders.load_prices("nse")
    r = ds.frame.pct_change()
    assert abs(r.loc[ex, "T0"]) < 0.05 and abs(r.loc[ex, "T1"]) < 1e-12   # no fake -50% / -40% day
    assert ds.source.startswith("CSV: prices_nse.csv") and "корпоративные события: 2" in ds.source
    raw = pd.read_csv(pd.io.common.StringIO(px), index_col=0)
    assert np.allclose(ds.frame["T2"].to_numpy(), raw["T2"].to_numpy())  # out-of-range event skipped
    again, n = loaders.apply_corporate_actions(ds.frame, loaders.validate_corporate_actions(
        pd.read_csv(pd.io.common.StringIO(actions)).iloc[[1]]))
    assert n == 1 and np.allclose(again["T1"], ds.frame["T1"])           # ratio method is idempotent


def test_corporate_actions_are_validated():
    ok = "ticker,ex_date,method,factor,note\n"
    cases = [("ticker,date,method\nT0,2021-05-03,ratio\n", "нет столбцов ex_date"),
             (ok + "T0,03.05.2021,ratio,,\n", "ex_date не распознана"),
             (ok + "T0,2021-05-03,split,2,\n", "метод «split» не поддерживается"),
             (ok + "T0,2021-05-03,factor,,\n", "требует положительного множителя"),
             (ok + "T0,2021-05-03,ratio,,\nT0,2021-05-03,factor,2,\n", "указано несколько раз")]
    for text, fragment in cases:
        with data_dir({"prices_kase.csv": prices_csv(), "corporate_actions.csv": text}):
            raises(lambda: loaders.load_prices("kase"), "corporate_actions.csv", fragment)
