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
