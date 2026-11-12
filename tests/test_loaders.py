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


def test_dividend_is_removed_from_the_price_gap_but_the_market_move_is_kept():
    px = prices_csv(n_days=320, n_assets=6)
    df = pd.read_csv(pd.io.common.StringIO(px), index_col=0)
    ex = df.index[200]
    last = df["T0"].iloc[199]
    df.loc[ex:, "T0"] -= 0.4 * last                            # a cash dividend of 40 % of the price, unadjusted
    actions = f"ticker,ex_date,method,factor,note\nT0,{ex},dividend,{0.4 * last},dividend\n"
    with data_dir({"prices_kase.csv": df.to_csv(), "corporate_actions.csv": actions}):
        ds = loaders.load_prices("kase")
    raw = pd.read_csv(pd.io.common.StringIO(px), index_col=0)
    r_adj, r_raw = ds.frame["T0"].pct_change(), raw["T0"].pct_change()
    assert np.isclose(r_adj.loc[ex], r_raw.loc[ex], atol=0.02)            # the ex-date keeps its market return
    assert np.allclose(r_adj.iloc[201:], df["T0"].pct_change().iloc[201:])  # later returns unchanged
    assert np.isclose(ds.frame["T0"].iloc[199], 0.6 * last)                 # earlier prices scaled by 1 - D/P
    too_big = f"ticker,ex_date,method,factor,note\nT0,{ex},dividend,{2 * last},typo\n"
    with data_dir({"prices_kase.csv": df.to_csv(), "corporate_actions.csv": too_big}):
        raises(lambda: loaders.load_prices("kase"), "corporate_actions.csv", "не меньше цены")


def test_corporate_actions_are_validated():
    ok = "ticker,ex_date,method,factor,note\n"
    cases = [("ticker,date,method\nT0,2021-05-03,ratio\n", "нет столбцов ex_date"),
             (ok + "T0,03.05.2021,ratio,,\n", "ex_date не распознана"),
             (ok + "T0,2021-05-03,split,2,\n", "метод «split» не поддерживается"),
             (ok + "T0,2021-05-03,factor,,\n", "метод factor требует положительного"),
             (ok + "T0,2021-05-03,ratio,,\nT0,2021-05-03,factor,2,\n", "указано несколько раз")]
    for text, fragment in cases:
        with data_dir({"prices_kase.csv": prices_csv(), "corporate_actions.csv": text}):
            raises(lambda: loaders.load_prices("kase"), "corporate_actions.csv", fragment)


# ------------------------------------------------------------------ KASE dividends and volumes
def test_kase_ex_date_register_at_midnight_t2_and_t0():
    cal = pd.bdate_range("2015-11-02", "2015-11-30").append(pd.bdate_range("2023-05-22", "2023-06-09"))
    # T+2, record Wed 2023-05-31: settle by Tue 30 -> deal Fri 26 -> ex-date Mon 29 (HSBK 2023)
    assert loaders.kase_ex_date(pd.Timestamp("2023-05-31"), cal) == pd.Timestamp("2023-05-29")
    # record on a Sunday: the last trading day before it is Fri 2 June -> ex-date Thu 1 June
    assert loaders.kase_ex_date(pd.Timestamp("2023-06-04"), cal) == pd.Timestamp("2023-06-01")
    # T+0 before the switch: ex-date is the first trading day on or after the record date
    assert loaders.kase_ex_date(pd.Timestamp("2015-11-11"), cal, "2015-12-09") == pd.Timestamp("2015-11-11")
    assert loaders.kase_ex_date(pd.Timestamp("2015-11-15"), cal, "2015-12-09") == pd.Timestamp("2015-11-16")


def test_kase_dividends_are_parsed_and_empty_entries_dropped():
    import json
    records = [{"is_paid": True, "protocol_date": "2025-05-15", "fixation_date": "2025-05-25",
                "period_start_date": "2024-01-01", "period_end_date": "2024-12-31",
                "payment_start_date": "2025-05-25", "per_share": 19923.74, "total_sum": None},
               {"is_paid": False, "protocol_date": None, "fixation_date": None, "per_share": None},
               {"is_paid": True, "fixation_date": "1996-12-31", "per_share": 0.0, "total_sum": 7.8e7}]

    real_get, loaders._get = loaders._get, lambda url, timeout=30: json.dumps(records).encode()
    try:
        with data_dir({}) as tmp:
            df = loaders.fetch_kase_dividends(["KZTK"])
            saved = pd.read_csv(tmp / "dividends_kase.csv")
    finally:
        loaders._get = real_get
    assert len(df) == 1 and df.loc[0, "record_date"] == "2025-05-25" and df.loc[0, "per_share"] == 19923.74
    assert list(saved.columns[:3]) == ["ticker", "record_date", "per_share"]


def test_kase_history_keeps_turnover_and_drops_todays_bar():
    import json
    day = 86400 * 19000
    today = int(pd.Timestamp.today().normalize().timestamp())

    def fake_get(url, timeout=30):
        return json.dumps({"s": "ok", "t": [day, day + 86400 * 2, today], "c": [10.0, 11.0, 12.0],
                           "v": [1e6, 2e6, 5.0]}).encode()

    real_get, loaders._get = loaders._get, fake_get
    try:
        out = loaders.fetch_kase_history(["A"], start="2022-01-01")
    finally:
        loaders._get = real_get
    assert out["close"]["A"].tolist() == [10.0, 11.0]
    assert out["volume"]["A"].tolist() == [1e6, 2e6]


def test_kase_ex_date_for_shares_that_never_switched_to_t2():
    cal = pd.bdate_range("2025-05-19", "2025-05-30")
    # KZTKp: record Sun 25.05.2025, T+0 -> first trading day on or after it, Mon 26.05
    assert loaders.kase_ex_date(pd.Timestamp("2025-05-25"), cal, None) == pd.Timestamp("2025-05-26")
    assert loaders.kase_ex_date(pd.Timestamp("2025-05-25"), cal) == pd.Timestamp("2025-05-22")


def test_kase_history_fill_modes():
    import json
    day = 86400 * 19000

    def fake_get(url, timeout=30):
        t = [day, day + 86400 * 2] if "symbol=A" in url else [day, day + 86400, day + 86400 * 3]
        return json.dumps({"s": "ok", "t": t, "c": [10.0 + i for i in range(len(t))], "v": [1.0] * len(t)}).encode()

    real_get, loaders._get = loaders._get, fake_get
    try:
        inside = loaders.fetch_kase_history(["A", "B"], start="2022-01-01", fill="inside")["close"]
        none = loaders.fetch_kase_history(["A", "B"], start="2022-01-01", fill="none")["close"]
    finally:
        loaders._get = real_get
    assert inside["A"].tolist()[:3] == [10.0, 10.0, 11.0] and np.isnan(inside["A"].iloc[3])  # delisted: no tail
    assert np.isnan(none["A"].iloc[1]) and none["B"].notna().sum() == 3


def test_kase_archive_sheet_is_parsed_day_first_and_sorted():
    raw = pd.DataFrame([["Индекс KASE 2008-01-01 - 2008-01-09", None, None],
                        ["Дата", "Закрытие", "Объем, млн KZT"],
                        ["09.01.08", "2624,95", 194.55],
                        ["03.01.2008", 2640.15, 141.19],
                        ["04.01.08", 2600.12, "1 136,2"]])
    df = loaders.parse_kase_archive(raw)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2008-01-03", "2008-01-04", "2008-01-09"]
    assert df["Закрытие"].tolist() == [2640.15, 2600.12, 2624.95] and df["Объем, млн KZT"].iloc[1] == 1136.2


def test_kase_bond_passport_is_read_by_label():
    j = {"848": {"label": "ISIN", "value": "XS1120709826"},
         "1": {"label": "Валюта выпуска и обслуживания", "value": "USD"},
         "2": {"label": "Номинальная стоимость в валюте выпуска",
               "value": "1,00 (минимальный размер долга в номинальном выражении)"},
         "3": {"label": "Текущая купонная ставка, % годовых", "value": "4,875"},
         "4": {"label": "Дата начала обращения", "value": "14.10.14"},
         "5": {"label": "Срок обращения, дней", "value": "10 800"},
         "market_makers": {"label": "Маркет-мейкеры", "value": "отсутствует"}}
    row = loaders.parse_kase_characteristics(j, "KZ_06_4410")
    assert row["isin"] == "XS1120709826" and row["currency"] == "USD" and row["nominal"] == 1.0
    assert row["coupon_pct"] == 4.875 and row["start_date"] == "2014-10-14" and row["maturity"] == "2044-10-14"
    j["6"] = {"label": "Дата погашения", "value": "25.03.15"}
    assert loaders.parse_kase_characteristics(j, "X")["maturity"] == "2015-03-25"


# ------------------------------------------------------------------ rates and macro data
def test_nbk_base_rate_dates_broken_by_markup():
    html = ("<td>26.04<span> .2022</span></td><td>&nbsp;14.00</td><td>13.00 - 15.00</td><td>25.01.2022</td><td>10.25</td>"
            "<td>07.09.2026</td><td>16 <b>.</b> 25</td><td>15 . 25 - 17 . 25</td>")
    df = loaders.parse_nbk_base_rate(html)
    assert df["effective_from"].dt.strftime("%Y-%m-%d").tolist() == ["2022-04-26", "2022-01-25", "2026-09-07"]
    assert df["rate"].tolist() == [14.0, 10.25, 16.25]


def test_nbk_refinancing_table_is_parsed():
    html = ("<tr><td>01.12.2007</td><td>№137 от 30.11.2007</td><td>11</td></tr>"
            "<tr><td>05.02.2009</td><td>№ 8 от 04.02.2009</td><td>9,5</td></tr>"
            "<tr><td>01.06.1995</td><td>№-, 31.05.95</td><td>75</td><td>69</td></tr>"
            "<tr><td>01.04.2017*</td><td>№30 от 24.02.2017</td><td>11.0</td></tr>")
    df = loaders.parse_nbk_refinancing(html)
    assert df["effective_from"].dt.strftime("%Y-%m-%d").tolist() == ["1995-06-01", "2007-12-01", "2009-02-05",
                                                                    "2017-04-01"]
    assert df["rate"].tolist() == [75.0, 11.0, 9.5, 11.0]


def test_fred_series_is_read_and_retried():
    import urllib.error
    calls = []

    def fake_get(url, timeout=30, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        return b"observation_date,DCOILBRENTEU\n1999-12-31,25.0\n2000-01-04,23.95\n2000-01-05,.\n2000-01-06,23.72\n"

    real_get, real_sleep = loaders._get, loaders.time.sleep
    loaders._get, loaders.time.sleep = fake_get, lambda s: None
    try:
        s = loaders.fetch_fred("DCOILBRENTEU", "2000-01-01")
    finally:
        loaders._get, loaders.time.sleep = real_get, real_sleep
    assert len(calls) == 2 and s.tolist() == [23.95, 23.72]


def test_stat_cpi_growth_and_index_conventions():
    old = pd.DataFrame([["Индекс потребительских цен"] + [None] * 12,
                        ["к соответствующему месяцу предыдущего года"] + [None] * 12,
                        ["Товары и услуги", 18.7, 18.8] + [None] * 10])
    new = pd.DataFrame([["к предыдущему месяцу"] + [None] * 12, ["Товары и услуги", 101.0] + [None] * 11,
                        ["к соответствующему месяцу предыдущего года"] + [None] * 12,
                        ["Товары и услуги", 112.2] + [None] * 11])
    s = loaders.parse_stat_cpi({"Метаданные": pd.DataFrame(), "monthly 2008": old, "monthly 2026": new})
    assert s.to_dict() == {pd.Period("2008-01", "M"): 18.7, pd.Period("2008-02", "M"): 18.8,
                           pd.Period("2026-01", "M"): 12.2}


def test_stat_kei_months_follow_year_rows():
    df = pd.DataFrame([[None, "Период", "за месяц", "с начала года", "ИФО, %"],
                       [None, None, "2021 г.", None, None], [None, "Январь", 4651.7, 4651.7, 94.2],
                       [None, "Февраль", 4936.2, 9587.8, 96], [None, "2022г.", None, None, None],
                       [None, "Январь", 5888.6, 5888.6, 103]])
    s = loaders.parse_stat_kei({"2021-2026": df, "Метаданные": pd.DataFrame()})
    assert s.to_dict() == {pd.Period("2021-01", "M"): 94.2, pd.Period("2021-02", "M"): 96.0,
                           pd.Period("2022-01", "M"): 103.0}


def test_cbar_bulletin_converts_old_manat():
    xml = ('<ValCurs Date="03.01.2000"><ValType Type="Xarici valyutalar"><Valute Code="EUR"><Nominal>1</Nominal>'
           '<Value>4400</Value></Valute><Valute Code="USD"><Nominal>1</Nominal><Value>4368</Value></Valute>'
           '</ValType></ValCurs>').encode()
    d, v = loaders.parse_cbar(xml)
    assert d == pd.Timestamp("2000-01-03") and abs(v - 0.8736) < 1e-12
    d, v = loaders.parse_cbar(xml.replace(b"03.01.2000", b"04.01.2016").replace(b"4368", b"1.5594"))
    assert v == 1.5594


def test_taldau_kei_keeps_the_republic_row_in_bn_kzt():
    records = [{"terms": [1], "termNames": ["Г.АЛМАТЫ"], "periods": [{"date": "31.01.2021", "value": "5"}]},
               {"terms": [2], "termNames": ["РЕСПУБЛИКА КАЗАХСТАН"],
                "periods": [{"name": "Февраль 2021 г.", "date": "28.02.2021", "value": "4936200000000"},
                            {"name": "Январь 2021 г.", "date": "31.01.2021", "value": "4651700000000"}]}]
    s = loaders.parse_taldau_kei(records)
    assert s.to_dict() == {pd.Period("2021-01", "M"): 4651.7, pd.Period("2021-02", "M"): 4936.2}


def test_clp_years_are_merged_and_failed_years_skipped():
    import json
    import urllib.error

    def fake_get(url, timeout=30, **kw):
        if url.endswith("/2001"):
            raise urllib.error.URLError("timeout")
        return json.dumps({"codigo": "dolar", "serie": [{"fecha": "2000-01-04T03:00:00.000Z", "valor": 520.1},
                                                         {"fecha": "2000-01-03T03:00:00.000Z", "valor": 518.0}]}).encode()

    real_get, loaders._get = loaders._get, fake_get
    try:
        s = loaders.fetch_clp(2000, 2001)
    finally:
        loaders._get = real_get
    assert list(s.index.strftime("%Y-%m-%d")) == ["2000-01-03", "2000-01-04"] and s.iloc[0] == 518.0
