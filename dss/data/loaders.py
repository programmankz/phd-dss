"""Data access layer.

Resolution order for every dataset (``source="auto"``):

1. a user CSV in the data directory (``DSS_DATA_DIR`` or ``./data``),
2. the synthetic DEMO generator (flagged with ``is_demo=True``).

``python -m dss.cli fetch`` downloads the real series from the original
sources used in the articles and writes them to the data directory:

* GDP growth  - IMF World Economic Outlook DataMapper API, indicator NGDP_RPCH  [A1]
                (falls back to World Bank WDI NY.GDP.MKTP.KD.ZG when IMF refuses)
* FX rates    - National Bank of the Republic of Kazakhstan, daily official rates [A2]
* NSE prices  - Yahoo Finance via ``yfinance`` (optional dependency)             [A4]
* KASE prices - CSV exported from kase.kz (no public API; see README)             [A3]

Expected CSV layouts
--------------------
gdp_growth.csv    : country,1980,1981,...,2024           (wide, % change)
fx_rates.csv      : Date,USD,EUR,CNY,RUB                 (KZT per unit)
prices_<u>.csv    : Date,<TICKER1>,<TICKER2>,...         (close prices)
fundamentals_<u>.csv : ticker,sector,PB,PE

A malformed user CSV raises :class:`DataError` with a message that names the
file, the cause and the action; :func:`dss.pipeline.run` turns it into a
warning and falls back to the demo set for that dataset only.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config
from . import demo

NSE_TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "BPCL.NS",
    "BRITANNIA.NS", "ITC.NS", "NESTLEIND.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS",
]  # [A4] Table II
NSE_SECTORS = dict(zip(NSE_TICKERS, ["Финансы"] * 4 + ["IT"] * 4 + ["Энергетика"] * 4 +
                       ["Потребительский"] * 4 + ["Авто"] * 4))


def data_dir() -> Path:
    return Path(os.environ.get("DSS_DATA_DIR", Path.cwd() / "data"))


@dataclass
class Dataset:
    frame: pd.DataFrame
    source: str
    is_demo: bool


# --------------------------------------------------------------------------
# Validation of user CSV files
# --------------------------------------------------------------------------
MIN_ASSETS = 6            # k-selection of Module III needs n / 3 >= 2
MIN_PRICE_DAYS = 300      # 252-day covariance window + walk-forward of Module IV
MIN_FX_YEARS = 5          # expected depreciation blends 1- and 5-year drifts


class DataError(ValueError):
    """A user data file is malformed. The message (in Russian) names the file,
    the cause and the action, so it can be shown to the user as is."""


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DataError(f"Файл {path.name}: файл пуст — заполните его или удалите, "
                        "чтобы использовать демо-набор") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"Файл {path.name}: не удалось прочитать как CSV ({exc}) — сохраните файл "
                        "в формате CSV (UTF-8, разделитель «,»)") from None
    if df.empty:
        raise DataError(f"Файл {path.name}: нет строк с данными — заполните файл или удалите его")
    return df


def _dated_frame(path: Path, layout: str) -> pd.DataFrame:
    """CSV with a ``Date`` column -> numeric frame indexed by date, sorted."""
    df = _read_csv(path)
    if "Date" not in df.columns:
        raise DataError(f"Файл {path.name}: нет столбца Date — проверьте заголовок "
                        f"(ожидается: {layout})")
    raw = df["Date"].astype(str).str.strip()
    # kase.kz and Russian-locale Excel export DD.MM.YYYY, which pandas would read month-first
    dotted = raw.str.fullmatch(r"\d{2}\.\d{2}\.\d{4}").all()
    dates = pd.to_datetime(raw, format="%d.%m.%Y" if dotted else None, errors="coerce")
    if dates.isna().any():
        bad = df.loc[dates.isna(), "Date"].iloc[0]
        raise DataError(f"Файл {path.name}: значение «{bad}» в столбце Date не распознано как дата "
                        f"({int(dates.isna().sum())} шт.) — используйте формат ГГГГ-ММ-ДД")
    if dates.duplicated().any():
        raise DataError(f"Файл {path.name}: дата {dates[dates.duplicated()].iloc[0]:%Y-%m-%d} "
                        "встречается несколько раз — удалите повторяющиеся строки")
    df = df.drop(columns="Date").set_index(pd.DatetimeIndex(dates, name="Date")).sort_index()
    num = df.apply(pd.to_numeric, errors="coerce")
    empty = [c for c in df.columns if num[c].isna().all()]
    if empty:
        raise DataError(f"Файл {path.name}: в столбце «{empty[0]}» нет ни одного числа — проверьте "
                        "разделитель дробной части (нужна точка) и формат файла")
    if (num <= 0).any().any():
        col = num.columns[(num <= 0).any()][0]
        raise DataError(f"Файл {path.name}: в столбце «{col}» есть нулевые или отрицательные значения — "
                        "цены и курсы должны быть положительными")
    return num


def validate_gdp(df: pd.DataFrame, name: str = "gdp_growth.csv") -> pd.DataFrame:
    df = df.set_index(df.columns[0])
    years = pd.to_numeric(pd.Series(df.columns), errors="coerce")
    if years.isna().any() or (years % 1 != 0).any():
        bad = df.columns[years.isna() | (years % 1 != 0)][0]
        raise DataError(f"Файл {name}: столбец «{bad}» не является годом — заголовок должен иметь вид "
                        "country,1980,1981,…,2024")
    df.columns = years.astype(int).tolist()
    df.index = df.index.astype(str).str.strip()
    df.index.name = "country"
    if config.TARGET_COUNTRY not in df.index:
        raise DataError(f"Файл {name}: нет строки «{config.TARGET_COUNTRY}» — Модуль I строит прогноз для "
                        "Казахстана; проверьте название страны в первом столбце")
    if df.index.duplicated().any():
        raise DataError(f"Файл {name}: страна «{df.index[df.index.duplicated()][0]}» встречается "
                        "несколько раз — оставьте одну строку на страну")
    split = config.MACRO_TRAIN_END_YEAR
    if not len(df.columns) or min(df.columns) > split - 5 or max(df.columns) <= split:
        span = f"{min(df.columns)}–{max(df.columns)}" if len(df.columns) else "нет"
        raise DataError(f"Файл {name}: годы ({span}) не покрывают обучающую (≤ {split}) и тестовую "
                        f"(> {split}) выборки — нужен ряд хотя бы с {split - 5} по {split + 1} г.")
    return df.apply(pd.to_numeric, errors="coerce")


def validate_fx(df: pd.DataFrame, name: str = "fx_rates.csv") -> pd.DataFrame:
    missing = [c for c in config.CURRENCIES if c not in df.columns]
    if missing:
        raise DataError(f"Файл {name}: нет столбцов {', '.join(missing)} — проверьте заголовок "
                        "(ожидается: Date,USD,EUR,CNY,RUB)")
    df = df[config.CURRENCIES].dropna()
    if len(df) < 2 or df.index[-1] - df.index[0] < pd.Timedelta(days=365 * MIN_FX_YEARS):
        raise DataError(f"Файл {name}: история курсов короче {MIN_FX_YEARS} лет — Модулю II нужен "
                        "ряд не короче 5 лет; загрузите его командой python -m dss.cli fetch fx")
    return df


def validate_prices(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.ffill().dropna(axis=1, how="all").dropna()
    if df.shape[1] < MIN_ASSETS:
        raise DataError(f"Файл {name}: {df.shape[1]} бумаг(и) с ценами — для кластеризации и лимитов "
                        f"нужно не меньше {MIN_ASSETS}; добавьте столбцы с тикерами")
    if len(df) < MIN_PRICE_DAYS:
        raise DataError(f"Файл {name}: {len(df)} торговых дней с ценами по всем бумагам — нужно "
                        f"не меньше {MIN_PRICE_DAYS}; удлините ряд или уберите бумаги с короткой историей")
    return df


def validate_fundamentals(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "ticker" not in df.columns:
        raise DataError(f"Файл {name}: нет столбца ticker — проверьте заголовок "
                        "(ожидается: ticker,sector,PB,PE)")
    if df["ticker"].duplicated().any():
        raise DataError(f"Файл {name}: тикер «{df.ticker[df.ticker.duplicated()].iloc[0]}» встречается "
                        "несколько раз — оставьте одну строку на бумагу")
    df = df.set_index("ticker")
    for c in ("PB", "PE"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Auto-resolving loaders
# --------------------------------------------------------------------------
def load_gdp(source: str = "auto") -> Dataset:
    path = data_dir() / "gdp_growth.csv"
    if source in ("auto", "csv") and path.exists():
        return Dataset(validate_gdp(_read_csv(path), path.name), f"CSV: {path.name}", False)
    if source == "live":
        df, src = fetch_gdp()
        return Dataset(df, src, False)
    return Dataset(demo.gdp_panel(), "DEMO: синтетическая панель", True)


def load_fx(source: str = "auto") -> Dataset:
    path = data_dir() / "fx_rates.csv"
    if source in ("auto", "csv") and path.exists():
        df = _dated_frame(path, "Date,USD,EUR,CNY,RUB")
        return Dataset(validate_fx(df, path.name), f"CSV: {path.name}", False)
    return Dataset(demo.fx_rates(), "DEMO: синтетические курсы", True)


def load_prices(universe: str = "kase", source: str = "auto") -> Dataset:
    path = data_dir() / f"prices_{universe}.csv"
    if source in ("auto", "csv") and path.exists():
        df = _dated_frame(path, "Date,<тикер1>,<тикер2>,…")
        return Dataset(validate_prices(df, path.name), f"CSV: {path.name}", False)
    if source == "live" and universe == "nse":
        return Dataset(fetch_yahoo(NSE_TICKERS), "Yahoo Finance (NSE)", False)
    return Dataset(demo.equity_prices(), "DEMO: синтетические цены (тикеры KASE)", True)


def load_fundamentals(universe: str = "kase", tickers=None, source: str = "auto") -> Dataset:
    path = data_dir() / f"fundamentals_{universe}.csv"
    if source in ("auto", "csv") and path.exists():
        df = validate_fundamentals(_read_csv(path), path.name)
        return Dataset(df, f"CSV: {path.name}", False)
    if universe == "nse" and tickers is not None:
        df = pd.DataFrame({"sector": [NSE_SECTORS.get(t, "—") for t in tickers]}, index=tickers)
        df.index.name = "ticker"
        return Dataset(df, "Секторы [A4] Table II", False)
    if tickers is not None and set(tickers) <= set(demo.KASE_DEMO):
        return Dataset(demo.fundamentals(), "DEMO: синтетические P/B, P/E", True)
    return Dataset(pd.DataFrame(index=pd.Index(tickers or [], name="ticker")), "нет", False)


# --------------------------------------------------------------------------
# Live sources
# --------------------------------------------------------------------------
def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "phd-dss/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_imf_gdp(last_year: int = 2024) -> pd.DataFrame:
    """Real GDP growth (%) for all IMF member countries, 1980-last_year."""
    base = "https://www.imf.org/external/datamapper/api/v1"
    countries = json.loads(_get(f"{base}/countries"))["countries"]
    values = json.loads(_get(f"{base}/NGDP_RPCH"))["values"]["NGDP_RPCH"]
    rows = {}
    for iso, series in values.items():
        if iso not in countries:          # drops regional aggregates
            continue
        label = countries[iso].get("label") or iso
        rows[label] = {int(y): v for y, v in series.items() if int(y) <= last_year}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)
    df.index.name = "country"
    return df


def fetch_worldbank_gdp(first_year: int = 1980, last_year: int = 2024) -> pd.DataFrame:
    """Real GDP growth (%) from World Bank WDI, indicator NY.GDP.MKTP.KD.ZG, same layout."""
    base = "https://api.worldbank.org/v2"
    meta = json.loads(_get(f"{base}/country?format=json&per_page=1000"))[1]
    countries = {c["id"]: c["name"] for c in meta if c["region"]["id"] != "NA"}  # drops aggregates
    url = (f"{base}/country/all/indicator/NY.GDP.MKTP.KD.ZG"
           f"?format=json&per_page=20000&date={first_year}:{last_year}")
    rows: dict = {}
    for obs in json.loads(_get(url, timeout=120))[1]:
        iso = obs["countryiso3code"]
        if iso in countries and obs["value"] is not None:
            rows.setdefault(countries[iso], {})[int(obs["date"])] = obs["value"]
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)
    df.index.name = "country"
    return df


def fetch_gdp(last_year: int = 2024) -> tuple[pd.DataFrame, str]:
    """IMF WEO [A1] first; the IMF API rejects scripted requests (HTTP 403), so fall back to World Bank."""
    try:
        return fetch_imf_gdp(last_year), "IMF WEO (NGDP_RPCH)"
    except (OSError, ValueError, KeyError) as exc:  # HTTPError/URLError are OSError
        print(f"  IMF DataMapper недоступен ({exc}); загружаю из World Bank WDI")
        return fetch_worldbank_gdp(last_year=last_year), "World Bank WDI (NY.GDP.MKTP.KD.ZG)"


def fetch_nbk_day(date: pd.Timestamp) -> dict:
    """Official NBK rates for one day (KZT per 1 unit)."""
    url = f"https://nationalbank.kz/rss/get_rates.cfm?fdate={date:%d.%m.%Y}"
    root = ET.fromstring(_get(url))
    out = {}
    for item in root.iter("item"):
        code = (item.findtext("title") or "").strip()
        if code in ("USD", "EUR", "CNY", "RUB"):
            quant = float(item.findtext("quant") or 1)
            out[code] = float(item.findtext("description")) / quant
    return out


def fetch_nbk(start: str = "2000-01-01", end: str | None = None,
              cache: Path | None = None, pause: float = 0.05) -> pd.DataFrame:
    """Download daily NBK rates incrementally, resuming from ``cache``."""
    cache = cache or data_dir() / "fx_rates.csv"
    have = pd.DataFrame()
    if cache.exists():
        have = pd.read_csv(cache, parse_dates=["Date"]).set_index("Date")
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    first = have.index.max() + pd.Timedelta(days=1) if len(have) else pd.Timestamp(start)
    new = {}
    for d in pd.date_range(first, end, freq="D"):
        try:
            rates = fetch_nbk_day(d)
            if rates:
                new[d] = rates
        except Exception as exc:  # network hiccup: keep what we have
            print(f"  {d:%Y-%m-%d}: {exc}")
        time.sleep(pause)
        if len(new) and len(new) % 250 == 0:
            print(f"  ... {d:%Y-%m-%d}")
    df = pd.concat([have, pd.DataFrame.from_dict(new, orient="index")]).sort_index()
    df.index.name = "Date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def fetch_yahoo(tickers, start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pip install yfinance") from exc
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px.ffill().dropna(how="all")
