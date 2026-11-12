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
* KASE prices - kase.kz chart feed (TradingView UDF, /tv-charts/securities)      [A3]
* NBK base rate - nationalbank.kz decision schedule pages (one table per year)
* US 3-month T-bill rate - FRED series TB3MS (CSV download)

Expected CSV layouts
--------------------
gdp_growth.csv    : country,1980,1981,...,2024           (wide, % change)
fx_rates.csv      : Date,USD,EUR,CNY,RUB                 (KZT per unit)
prices_<u>.csv    : Date,<TICKER1>,<TICKER2>,...         (close prices)
fundamentals_<u>.csv : ticker,sector,PB,PE
corporate_actions.csv : ticker,ex_date,method,factor,note  (optional, applied to every price file;
                        method ratio | factor | dividend, see validate_corporate_actions)
nbk_base_rate.csv : effective_from,rate   (NBK base rate, % p.a., by effective date)
tb3ms.csv         : Date,rate             (3-month US T-bill rate, % p.a., monthly, FRED TB3MS)

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
    "TMPV.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS",
]  # [A4] Table II; TATAMOTORS.NS was renamed TMPV.NS after the Oct 2025 demerger (history kept)
# Liquid KASE shares with >= 3 years of history (BAST trades too rarely, AIRA listed in 2024).
KASE_SECTORS = {
    "HSBK": "Финансы", "KSPI": "Финансы", "CCBN": "Финансы", "KZTK": "Телеком", "KCEL": "Телеком",
    "KEGC": "Энергетика", "KZTO": "Энергетика", "KMGZ": "Нефть и газ", "KZAP": "Уран",
}
KASE_TICKERS = list(KASE_SECTORS)
# Demergers Yahoo does not adjust for: {ticker: first ex-date}. TMCV.NS (commercial vehicles)
# was spun off from Tata Motors on 2025-10-14; the raw close drops ~40% that day. Built-in
# default applied by `fetch nse`; further events go to data/corporate_actions.csv.
NSE_DEMERGERS = {"TMPV.NS": "2025-10-14"}
ACTION_METHODS = ("ratio", "factor", "dividend")
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


def validate_corporate_actions(df: pd.DataFrame, name: str = "corporate_actions.csv") -> pd.DataFrame:
    """ticker,ex_date,method,factor,note -> typed frame; ``method`` is "ratio" (back-adjust by the
    observed close ratio around the ex-date), "factor" (multiply earlier prices by ``factor``) or
    "dividend" (``factor`` is the cash amount per share: earlier prices are multiplied by
    1 - amount / last close before the ex-date, the usual total-return adjustment)."""
    missing = [c for c in ("ticker", "ex_date", "method") if c not in df.columns]
    if missing:
        raise DataError(f"Файл {name}: нет столбцов {', '.join(missing)} — проверьте заголовок "
                        "(ожидается: ticker,ex_date,method,factor,note)")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    dates = pd.to_datetime(df["ex_date"].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        bad = df.loc[dates.isna(), "ex_date"].iloc[0]
        raise DataError(f"Файл {name}: дата «{bad}» в столбце ex_date не распознана — используйте формат ГГГГ-ММ-ДД")
    df["ex_date"] = dates
    bad = df.loc[~df["method"].isin(ACTION_METHODS), "method"]
    if len(bad):
        raise DataError(f"Файл {name}: метод «{bad.iloc[0]}» не поддерживается — укажите ratio (по отношению цен "
                        "до и после события), factor (явный множитель) или dividend (выплата на акцию)")
    factor = pd.to_numeric(df["factor"], errors="coerce") if "factor" in df else pd.Series(float("nan"), df.index)
    need = df["method"].isin(("factor", "dividend")) & ~(factor > 0)
    if need.any():
        t = df.loc[need].iloc[0]
        raise DataError(f"Файл {name}: для {t.ticker} ({t.ex_date:%Y-%m-%d}) метод {t.method} требует положительного "
                        "числа в столбце factor (множитель или выплата на акцию; точка как разделитель дробной части)")
    df["factor"] = factor
    dup = df.duplicated(["ticker", "ex_date"])
    if dup.any():
        t = df.loc[dup].iloc[0]
        raise DataError(f"Файл {name}: событие {t.ticker} {t.ex_date:%Y-%m-%d} указано несколько раз — "
                        "оставьте одну строку")
    return df.sort_values("ex_date").reset_index(drop=True)


def load_corporate_actions() -> pd.DataFrame | None:
    """Optional data/corporate_actions.csv, validated; None if there is no file."""
    path = data_dir() / "corporate_actions.csv"
    if not path.exists():
        return None
    return validate_corporate_actions(_read_csv(path), path.name)


def validate_rates(nbk: pd.DataFrame, tb: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Typed step series of the two reference rates (fractions, indexed by date)."""
    for df, cols, name in ((nbk, ("effective_from", "rate"), "nbk_base_rate.csv"), (tb, ("Date", "rate"), "tb3ms.csv")):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DataError(f"Файл {name}: нет столбцов {', '.join(missing)} — проверьте заголовок "
                            f"(ожидается: {','.join(cols)})")
    out = []
    for df, col, name in ((nbk, "effective_from", "nbk_base_rate.csv"), (tb, "Date", "tb3ms.csv")):
        dates = pd.to_datetime(df[col].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
        rate = pd.to_numeric(df["rate"], errors="coerce")
        if dates.isna().any() or rate.isna().any() or (rate < 0).any() or (rate > 100).any():
            raise DataError(f"Файл {name}: даты должны быть в формате ГГГГ-ММ-ДД, ставка — числом в процентах "
                            "годовых от 0 до 100")
        s = pd.Series(rate.to_numpy() / 100.0, index=dates).sort_index()
        if s.index.duplicated().any():
            raise DataError(f"Файл {name}: дата {s.index[s.index.duplicated()][0]:%Y-%m-%d} указана несколько раз")
        out.append(s)
    return out[0], out[1]


def load_rates() -> tuple[pd.DataFrame, str] | tuple[None, str]:
    """Daily (calendar) frame ``kzt``, ``usd`` of the reference rates as fractions, forward-filled
    from the decision / observation dates; (None, reason) when one of the files is missing."""
    nbk_path, tb_path = data_dir() / "nbk_base_rate.csv", data_dir() / "tb3ms.csv"
    if not nbk_path.exists() or not tb_path.exists():
        return None, "нет файлов nbk_base_rate.csv и tb3ms.csv (python -m dss.cli fetch rates)"
    nbk, tb = validate_rates(_read_csv(nbk_path), _read_csv(tb_path))
    days = pd.date_range(min(nbk.index.min(), tb.index.min()), max(nbk.index.max(), tb.index.max()) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=400), freq="D")
    df = pd.DataFrame({"kzt": nbk.reindex(days).ffill(), "usd": tb.reindex(days).ffill()})
    df.index.name = "Date"
    return df, (f"CSV: nbk_base_rate.csv ({len(nbk)} решений до {nbk.index.max():%Y-%m-%d}), "
                f"tb3ms.csv (до {tb.index.max():%Y-%m})")


def mean_base_rate(start: str = "2019-01-01", end: str | None = None) -> float | None:
    """Average NBK base rate over [start, end] from data/nbk_base_rate.csv, or None without the file."""
    df, _ = load_rates()
    if df is None:
        return None
    s = df["kzt"].loc[start:end or df.index.max()]
    return float(s.mean()) if len(s) else None


def apply_corporate_actions(px: pd.DataFrame, actions: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Back-adjust prices before each ex-date; returns (frame, number of events applied).
    Events for tickers not in ``px`` or with ex-dates outside its range are skipped. The ratio
    method is idempotent: on already adjusted prices the ratio is 1. The dividend method keeps
    the market move of the ex-date: only the cash amount is removed from the price gap."""
    px, n = px.copy(), 0
    if actions is None:
        return px, n
    for a in actions.itertuples():
        if a.ticker not in px:
            continue
        s = px[a.ticker]
        before, after = s[s.index < a.ex_date].dropna(), s[s.index >= a.ex_date].dropna()
        if before.empty or after.empty:
            continue
        if a.method == "ratio":
            k = after.iloc[0] / before.iloc[-1]
        elif a.method == "dividend":
            if a.factor >= before.iloc[-1]:
                raise DataError(f"Файл corporate_actions.csv: выплата {a.factor:g} по {a.ticker} не меньше цены "
                                f"{before.iloc[-1]:g} перед {a.ex_date:%Y-%m-%d} — проверьте сумму и дату")
            k = 1 - a.factor / before.iloc[-1]
        else:
            k = a.factor
        px.loc[px.index < a.ex_date, a.ticker] = s[s.index < a.ex_date] * k
        n += 1
    return px, n


def raw_prices(universe: str = "kase") -> tuple[pd.DataFrame, str]:
    """Prices from data/prices_<universe>.csv with corporate actions applied, before the
    common-history filter of :func:`validate_prices`; returns (frame, source label)."""
    path = data_dir() / f"prices_{universe}.csv"
    df, n = apply_corporate_actions(_dated_frame(path, "Date,<тикер1>,<тикер2>,…"), load_corporate_actions())
    return df, f"CSV: {path.name}" + (f"; корпоративные события: {n} (corporate_actions.csv)" if n else "")


def load_prices(universe: str = "kase", source: str = "auto") -> Dataset:
    path = data_dir() / f"prices_{universe}.csv"
    if source in ("auto", "csv") and path.exists():
        df, label = raw_prices(universe)
        return Dataset(validate_prices(df, path.name), label, False)
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
def _get(url: str, timeout: int = 30, agent: str = "phd-dss/1.0") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": agent})
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


def fetch_kase_history(tickers=None, start: str = "2019-01-01", end: str | None = None,
                       fill: str = "all") -> dict[str, pd.DataFrame]:
    """Daily bars from the kase.kz chart feed (TradingView UDF): {"close": KZT, "volume": turnover
    in KZT (field ``v``; equals ``volume_kzt`` of /api/instruments/trade-info)}. Closes are
    unadjusted for dividends and forward-filled over days without trades; volumes are left NaN
    on those days. Today's unfinished intraday bar is dropped. ``fill="inside"`` forward-fills
    only between a security's first and last bar (delisted shares stay empty afterwards);
    ``fill="none"`` keeps NaN on days without trades (bonds)."""
    today = pd.Timestamp.today().normalize()
    t0 = int(pd.Timestamp(start).timestamp())
    t1 = int((pd.Timestamp(end) if end else today + pd.Timedelta(days=1)).timestamp())
    close, volume = {}, {}
    for t in tickers or KASE_TICKERS:
        url = f"https://kase.kz/tv-charts/securities/history?symbol={t}&resolution=D&from={t0}&to={t1}"
        try:
            j = json.loads(_get(url))
        except (OSError, ValueError) as exc:  # HTTP errors; HTML error pages are not JSON
            print(f"  {t}: {exc} — пропущен")
            continue
        if j.get("s") != "ok" or not j.get("t"):
            print(f"  {t}: нет данных на kase.kz — пропущен")
            continue
        idx = pd.to_datetime(j["t"], unit="s").normalize()
        keep = (idx < today) & ~idx.duplicated(keep="last")   # one bar per day (rare duplicate stamps)
        close[t] = pd.Series(j["c"], index=idx, dtype=float)[keep]
        if "v" in j:
            volume[t] = pd.Series(j["v"], index=idx, dtype=float)[keep]
    if not close:
        raise RuntimeError("kase.kz не вернул цен ни по одному тикеру")
    px = pd.DataFrame(close).sort_index()
    px.index.name = "Date"
    vol = pd.DataFrame(volume).reindex(px.index)
    if fill == "all":
        px = px.ffill()
    elif fill == "inside":
        px = px.ffill(limit_area="inside")
    return {"close": px, "volume": vol}


def fetch_kase(tickers=None, start: str = "2019-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily closes (KZT) from the kase.kz chart feed. Unadjusted for dividends, as on KASE."""
    return fetch_kase_history(tickers, start, end)["close"]


# Liquid KASE shares of 2008-2015 that are delisted or outside the main set (data/prices_kase_extra.csv):
# >= ~500 trading days in 2008-2015 in the chart feed.
KASE_EXTRA_TICKERS = ["GB_KZMS", "RDGZ", "RDGZp", "KKGB", "KKGBp", "BTAS", "KZTKp", "GB_ENRC"]
KASE_INDEX_URL = "https://kase.kz/api/indicators/index-kase/archive-xls?start_date={start}&end_date={end}&language=ru"
# Bond indices published by KASE: archive path -> columns kept (GSec indices KZGB since 03.2022,
# corporate bond indices KASE_BMY / KASE_BMC since 09.2017).
KASE_BOND_INDICES = {"kzgb/archive-xls/Y": None, "kzgb/archive-xls/CP": None, "kzgb/archive-xls/DP": None,
                     "kase-b/KASE_BMY/archive-xls": None, "kase-b/KASE_BMC/archive-xls": None}
# Bonds with the longest daily history in the chart feed (KASE's secondary bond market is thin:
# a few hundred trading days per issue at best). MEUKAM (KZT GSec), sovereign USD Eurobonds,
# KZT and USD corporates.
KASE_BONDS = ["MUM120_0001", "MUM180_0001", "MUM240_0001", "MUM120_0017", "MUM180_0011", "MUM168_0003",
              "KZ_05_2410", "KZ_06_4410", "HSBKb10", "KKGBb4", "KEGCb3", "KMGZe11"]
# passport labels of /api/instruments/{gsecs|bonds}/<code>/characteristics (keys differ by issue)
KASE_BOND_FIELDS = {"ISIN": "isin", "Валюта выпуска и обслуживания": "currency",
                    "Номинальная стоимость в валюте выпуска": "nominal", "Текущая купонная ставка, % годовых": "coupon_pct",
                    "Вид купонной ставки": "coupon_type", "Дата начала обращения": "start_date",
                    "Дата погашения": "maturity_date", "Период погашения": "redemption", "Срок обращения, дней": "term_days",
                    "Наименование облигации": "kind", "Расчетный базис (дней в месяце / дней в году)": "day_count"}


def _excel(data: bytes) -> pd.DataFrame:
    import io
    try:
        return pd.read_excel(io.BytesIO(data), header=None)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("для архивов KASE в формате xlsx нужен openpyxl: pip install openpyxl") from exc


def parse_kase_archive(raw: pd.DataFrame) -> pd.DataFrame:
    """KASE indicator archive sheet (row 0: title, row 1: header, then one row per day, dates
    DD.MM.YY or DD.MM.YYYY, any order) -> numeric frame indexed by Date, ascending."""
    head = [str(c).strip() for c in raw.iloc[1]]
    df = raw.iloc[2:].copy()
    df.columns = head
    s = df[head[0]].astype(str).str.strip()
    dates = pd.to_datetime(s, format="%d.%m.%y", errors="coerce").fillna(
        pd.to_datetime(s, format="%d.%m.%Y", errors="coerce"))
    df = df.drop(columns=head[0]).set_index(pd.DatetimeIndex(dates, name="Date"))
    df = df[df.index.notna()].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(",", ".").str.replace(" ", ""),
                                                            errors="coerce"))
    return df[~df.index.duplicated()].sort_index()


INDEX_COLUMNS = {"Открытие": "open", "Максимум": "high", "Минимум": "low", "Закрытие": "close",
                 "Объем, млн KZT": "volume_mln_kzt", "Объем, тыс USD": "volume_thousand_usd"}


def fetch_kase_index(start: str = "2008-01-01", end: str | None = None, cache: Path | None = None) -> pd.DataFrame:
    """Daily KASE index (open/high/low/close, turnover) from the exchange's archive download;
    writes data/index_kase.csv."""
    end = end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = parse_kase_archive(_excel(_get(KASE_INDEX_URL.format(start=start, end=end), timeout=120)))
    df = df.rename(columns=INDEX_COLUMNS)[list(INDEX_COLUMNS.values())]
    (cache or data_dir() / "index_kase.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache or data_dir() / "index_kase.csv")
    return df


def fetch_kase_bond_indices(start: str = "2008-01-01", end: str | None = None, cache: Path | None = None) -> pd.DataFrame:
    """KASE bond indices (KZGB_* GSec yield / clean / dirty price, KASE_BMY yield and KASE_BMC
    clean price of corporates); writes data/bond_indices_kase.csv."""
    end = end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for path in KASE_BOND_INDICES:
        url = f"https://kase.kz/api/indicators/{path}?start_date={start}&end_date={end}&language=ru"
        frames.append(parse_kase_archive(_excel(_get(url, timeout=120))))
    df = pd.concat(frames, axis=1).sort_index()
    df.to_csv(cache or data_dir() / "bond_indices_kase.csv")
    return df


def parse_kase_characteristics(j: dict, code: str) -> dict:
    """Bond passport from /api/instruments/{gsecs|bonds}/<code>/characteristics (matched by label)."""
    import re
    by_label = {str(v.get("label", "")).strip(): v.get("value") for v in j.values() if isinstance(v, dict)}
    out = {"code": code}
    for label, name in KASE_BOND_FIELDS.items():
        v = re.sub(r"<[^>]+>|&[a-z]+;", " ", str(by_label.get(label) or "")).strip()
        out[name] = re.sub(r"\s+", " ", v) or None
    for name in ("nominal", "coupon_pct", "term_days"):
        num = re.match(r"[\d\s\xa0]+(?:,\d+)?", out[name] or "")       # "1 000,00", "1,00 (minimum ...)"
        out[name] = float(re.sub(r"[\s\xa0]", "", num.group(0)).replace(",", ".")) if num else None
    start = pd.to_datetime(out["start_date"], format="%d.%m.%y", errors="coerce") if out["start_date"] else pd.NaT
    out["start_date"] = None if pd.isna(start) else f"{start:%Y-%m-%d}"
    mat = pd.to_datetime((out.pop("maturity_date") or out["redemption"] or "")[:8], format="%d.%m.%y", errors="coerce")
    if pd.isna(mat) and out["term_days"] and not pd.isna(start):   # 30/360: years = days / 360
        mat = start + pd.DateOffset(months=round(out["term_days"] / 30))
    out["maturity"] = None if pd.isna(mat) else f"{mat:%Y-%m-%d}"
    return out


def fetch_kase_bonds(codes=None, start: str = "2008-01-01", cache: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily dirty prices of KASE bonds (chart feed, currency of the bond per one bond; NaN on
    days without trades) and their passports; writes data/bonds_kase.csv and bonds_kase_meta.csv."""
    codes = codes or KASE_BONDS
    px = fetch_kase_history(codes, start=start, fill="none")["close"]
    meta = []
    for c in px.columns:
        j = None
        for kind in ("gsecs", "bonds"):
            try:
                j = json.loads(_get(f"https://kase.kz/api/instruments/{kind}/{c}/characteristics"))
                if any(isinstance(v, dict) and v.get("label") == "ISIN" for v in j.values()) or "detail" not in j:
                    break
            except (OSError, ValueError):
                j = None
        row = parse_kase_characteristics(j or {}, c)
        row |= {"first_trade": f"{px[c].first_valid_index():%Y-%m-%d}", "last_trade": f"{px[c].last_valid_index():%Y-%m-%d}",
                "trading_days": int(px[c].notna().sum())}
        meta.append(row)
    out = data_dir() if cache is None else cache
    px.dropna(how="all").to_csv(out / "bonds_kase.csv")
    meta = pd.DataFrame(meta)
    meta.to_csv(out / "bonds_kase_meta.csv", index=False)
    return px, meta


# Dates from which KASE share deals settle T+2 only (Board decisions, KASE news 1279747,
# 1280974, 1282194). Before them deals settled T+0. Shares listed later trade T+2 from listing.
KASE_T2_FROM = {"KCEL": "2015-11-25", "RDGZ": "2015-11-25", "HSBK": "2015-12-09", "KZTK": "2015-12-09",
                "CCBN": "2015-12-09", "GB_KZMS": "2015-12-09", "KEGC": "2015-12-23", "KZTO": "2015-12-23",
                # never switched: the instrument passport on kase.kz gives "Схема расчетов: T+0"
                "KZTKp": None, "RDGZp": None, "KKGB": None, "KKGBp": None}
KASE_T2_DEFAULT = "2015-12-23"
KASE_DIVIDENDS_URL = "https://kase.kz/api/instruments/shares/{ticker}/dividends"


def kase_ex_date(record: pd.Timestamp, calendar: pd.DatetimeIndex, t2_from: str | None = KASE_T2_DEFAULT) -> pd.Timestamp:
    """First trading day without the right to the dividend. The shareholder register is fixed
    as of 00:00 of the record date R, so a deal must settle on a trading day strictly before R:
    with T+0 the ex-date is the first trading day >= R, with T+2 it is one trading day before
    the last trading day < R (the deal of the day before that settles on it). ``t2_from=None``:
    the share always settles T+0."""
    record = pd.Timestamp(record)
    last = int(calendar.searchsorted(record, side="left")) - 1      # last trading day < R
    if last < 1:
        raise ValueError(f"record date {record:%Y-%m-%d} is before the calendar")
    t2 = t2_from is not None and calendar[last - 1] >= pd.Timestamp(t2_from)
    if t2:
        return calendar[last - 1]
    if last + 1 >= len(calendar):
        raise ValueError(f"record date {record:%Y-%m-%d} is after the calendar")
    return calendar[last + 1]


def parse_kase_dividends(records: list, ticker: str) -> pd.DataFrame:
    """Rows of /api/instruments/shares/<ticker>/dividends -> ticker, record_date, per_share, ...;
    entries without a record date or a positive amount per share are dropped."""
    cols = ["ticker", "record_date", "per_share", "protocol_date", "payment_start_date",
            "period_start", "period_end", "is_paid"]
    rows = [{"ticker": ticker, "record_date": r.get("fixation_date"), "per_share": r.get("per_share"),
             "protocol_date": r.get("protocol_date"), "payment_start_date": r.get("payment_start_date"),
             "period_start": r.get("period_start_date"), "period_end": r.get("period_end_date"),
             "is_paid": r.get("is_paid")} for r in records]
    df = pd.DataFrame(rows, columns=cols)
    df["per_share"] = pd.to_numeric(df["per_share"], errors="coerce")
    df = df[df["record_date"].notna() & (df["per_share"] > 0)]
    return df.sort_values("record_date").reset_index(drop=True)


def fetch_kase_dividends(tickers=None, cache: Path | None = None) -> pd.DataFrame:
    """Dividend history of KASE shares (record date and amount per share, as published by
    the exchange); writes data/dividends_kase.csv."""
    frames = []
    for t in tickers or KASE_TICKERS:
        try:
            frames.append(parse_kase_dividends(json.loads(_get(KASE_DIVIDENDS_URL.format(ticker=t))), t))
        except (OSError, ValueError) as exc:
            print(f"  {t}: {exc} — пропущен")
    if not frames:
        raise RuntimeError("kase.kz не вернул дивидендов ни по одному тикеру")
    df = pd.concat(frames, ignore_index=True)
    cache = cache or data_dir() / "dividends_kase.csv"
    df.to_csv(cache, index=False)
    return df


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


NBK_BASE_RATE_URL = "https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke/rubrics/1544"


def parse_nbk_base_rate(html: str) -> pd.DataFrame:
    """Rows ``effective_from,rate`` from one yearly page of the NBK decision schedule: the table
    lists "DD.MM.YYYY" and the rate in %, one decision per row; rows without a rate are skipped."""
    import html as html_lib
    import re
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", html))           # "&nbsp;14.00" -> " 14.00"
    # the pages sometimes break a date with markup: "26.04<span>.2022</span>"
    # ... and a rate as well: "16 . 25"
    rows = re.findall(r"(\d{2})\s*\.\s*(\d{2})\s*\.\s*(\d{4})\*?\s+(\d{1,2})\s*[.,]\s*(\d{1,2})\b", text)
    df = pd.DataFrame([(f"{d}.{m}.{y}", f"{a}.{b}") for d, m, y, a, b in rows], columns=["effective_from", "rate"])
    df["effective_from"] = pd.to_datetime(df["effective_from"], format="%d.%m.%Y")
    df["rate"] = df["rate"].str.replace(",", ".").astype(float)
    return df


def fetch_nbk_base_rate(cache: Path | None = None, start: str = "2008-01-01") -> pd.DataFrame:
    """NBK policy rate by effective date; writes data/nbk_base_rate.csv. From 02.09.2015 the base
    rate (decision schedule, all yearly pages linked from the navigation); before it the official
    refinancing rate (NBK table of official rates since 1992), starting with the rate in force
    on ``start``. Only the dates on which the rate changed are kept."""
    import re
    first = _get(NBK_BASE_RATE_URL).decode("utf8", "replace")
    links = sorted(set(re.findall(r"/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke/rubrics/(\d+)", first)))
    frames = [parse_nbk_base_rate(first)]
    for rid in links:
        url = NBK_BASE_RATE_URL.rsplit("/", 1)[0] + "/" + rid
        if url != NBK_BASE_RATE_URL:
            frames.append(parse_nbk_base_rate(_get(url).decode("utf8", "replace")))
    base = pd.concat(frames)
    refi = parse_nbk_refinancing(_get(NBK_REFINANCING_URL).decode("utf8", "replace"))
    refi = refi[refi["effective_from"] < BASE_RATE_FROM]
    in_force = refi[refi["effective_from"] <= pd.Timestamp(start)].tail(1)
    refi = pd.concat([in_force, refi[refi["effective_from"] > pd.Timestamp(start)]])
    df = pd.concat([refi, base[base["effective_from"] >= BASE_RATE_FROM]])
    df = df.drop_duplicates("effective_from").sort_values("effective_from")
    df = df[df["rate"].ne(df["rate"].shift())]           # keep changes only
    cache = cache or data_dir() / "nbk_base_rate.csv"
    df.assign(effective_from=df["effective_from"].dt.strftime("%Y-%m-%d")).to_csv(cache, index=False)
    return df


NBK_REFINANCING_URL = "https://nationalbank.kz/ru/page/stavka-refinansirovaniya-nbk"
BASE_RATE_FROM = pd.Timestamp("2015-09-02")      # first base rate decision


def parse_nbk_refinancing(html: str) -> pd.DataFrame:
    """Rows ``effective_from,rate`` of the NBK table of the official refinancing rate:
    "DD.MM.YYYY №<resolution> от <date> <rate>"; before 1996 the table has several rate
    columns, the first one is kept."""
    import re
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    rows = re.findall(r"(\d{2}\.\d{2}\.\d{4})\*?\s+№\s*[\d-]*\s*,?\s*(?:от\s*)?[\d.]*\s+(\d{1,3}(?:[.,]\d+)?)\b", text)
    df = pd.DataFrame(rows, columns=["effective_from", "rate"])
    df["effective_from"] = pd.to_datetime(df["effective_from"], format="%d.%m.%Y")
    df["rate"] = df["rate"].str.replace(",", ".").astype(float)
    return df.drop_duplicates("effective_from").sort_values("effective_from").reset_index(drop=True)


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_AGENT = "curl/8.4.0"   # the FRED edge stalls requests with unknown user agents such as "phd-dss/1.0"


def fetch_fred(series: str, start: str = "2000-01-01", tries: int = 4) -> pd.Series:
    """One FRED series as a float Series indexed by date ("." missing values dropped). The FRED
    edge drops some scripted connections, so the request is repeated a few times."""
    import io
    for attempt in range(tries):
        try:
            raw = pd.read_csv(io.BytesIO(_get(FRED_URL.format(series=series), timeout=60, agent=FRED_AGENT)))
            break
        except (OSError, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]), name="Date"), name=series)
    return s[s.index >= pd.Timestamp(start)].dropna()


def fetch_tb3ms(cache: Path | None = None, start: str = "2008-01-01") -> pd.DataFrame:
    """Download the monthly 3-month T-bill rate (FRED TB3MS); writes data/tb3ms.csv."""
    s = fetch_fred("TB3MS", start)
    raw = pd.DataFrame({"Date": s.index.strftime("%Y-%m-%d"), "rate": s.to_numpy()})
    cache = cache or data_dir() / "tb3ms.csv"
    raw.to_csv(cache, index=False)
    return raw


def fetch_brent(cache: Path | None = None, start: str = "2000-01-01") -> pd.DataFrame:
    """Daily Brent spot price, USD per barrel (FRED DCOILBRENTEU, EIA); writes data/brent.csv."""
    df = fetch_fred("DCOILBRENTEU", start).rename("brent").to_frame()
    df.to_csv(cache or data_dir() / "brent.csv")
    return df


# ---------------------------------------------------------------- monthly macro data of Kazakhstan
STAT_KEI_URL = "https://stat.gov.kz/api/iblock/element/5941/file/ru/"        # KEI, Republic, 2021-
STAT_CPI_URL = "https://stat.gov.kz/api/iblock/element/1550/file/ru/"        # CPI by 12 divisions, 2008-
TALDAU_KEI_URL = "https://taldau.stat.gov.kz/ru/Api/GetIndexData/4023005?period=4&dics=67"
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
             "октябрь", "ноябрь", "декабрь"]


def parse_stat_kei(sheets: dict) -> pd.Series:
    """KEI table of stat.gov.kz (rows "<year> г." followed by month rows; the last column is the
    volume index, period of the year to date vs the same period of the previous year, %)."""
    import re
    out = {}
    for name, df in sheets.items():
        if not re.search(r"\d{4}", str(name)):
            continue
        year = None
        for _, row in df.iterrows():
            cells = [str(v).strip() for v in row.tolist()]
            y = next((re.match(r"(\d{4})\s*г", c) for c in cells if re.match(r"\d{4}\s*г", c)), None)  # "2021 г.", "2025г."
            if y:
                year = int(y.group(1))
                continue
            month = next((MONTHS_RU.index(c.lower()) + 1 for c in cells if c.lower() in MONTHS_RU), None)
            if year and month:
                val = pd.to_numeric(row.iloc[-1], errors="coerce")
                if pd.notna(val):
                    out[pd.Period(f"{year}-{month:02d}", "M")] = float(val)
    return pd.Series(out, name="kei").sort_index()


def parse_stat_cpi(sheets: dict) -> pd.Series:
    """Year-on-year CPI ("Товары и услуги" under "к соответствующему месяцу предыдущего года")
    from the yearly sheets "monthly YYYY" of stat.gov.kz. Older sheets give the growth in %,
    newer ones the index (previous year = 100); both are returned as growth in %."""
    import re
    out = {}
    for name, df in sheets.items():
        m = re.search(r"(\d{4})", str(name))
        if not m:
            continue
        year, block = int(m.group(1)), None
        for _, row in df.iterrows():
            label = str(row.iloc[0]).strip()
            if label.lower().startswith("к "):
                block = label.lower()
            elif label == "Товары и услуги" and block == "к соответствующему месяцу предыдущего года":
                for k, v in enumerate(row.iloc[1:13], start=1):
                    v = pd.to_numeric(v, errors="coerce")
                    if pd.notna(v):
                        out[pd.Period(f"{year}-{k:02d}", "M")] = float(v) - 100 if v > 50 else float(v)
    return pd.Series(out, name="cpi_yoy").sort_index().round(2)


def parse_taldau_kei(records: list) -> pd.Series:
    """Republic-level nominal KEI (bn KZT per month) from the taldau.stat.gov.kz API."""
    rk = [r for r in records if any("РЕСПУБЛИКА" in str(n) for n in r.get("termNames", []))]
    out = {}
    for r in rk[:1]:
        for p in r["periods"]:
            d = pd.to_datetime(p["date"], format="%d.%m.%Y")
            out[d.to_period("M")] = float(p["value"]) / 1e9
    return pd.Series(out, name="kei_nominal_bn_kzt").sort_index().round(1)


def fetch_macro_kz(cache: Path | None = None, start: str = "2008-01") -> pd.DataFrame:
    """Monthly macro data of Kazakhstan; writes data/macro_monthly_kz.csv with columns
    month, kei, kei_nominal_bn_kzt, cpi_yoy, reserves_usd_bn and assumed publication dates
    (``published_*``, see data/README.md). International reserves are not available as a
    machine-readable series on nationalbank.kz and stay empty."""
    kei = parse_stat_kei(_excel_sheets(_get(STAT_KEI_URL, timeout=120)))
    cpi = parse_stat_cpi(_excel_sheets(_get(STAT_CPI_URL, timeout=120)))
    try:
        nominal = parse_taldau_kei(json.loads(_get(TALDAU_KEI_URL, timeout=180)))
    except (OSError, ValueError) as exc:
        print(f"  taldau.stat.gov.kz: {exc} — номинальный КЭИ пропущен")
        nominal = pd.Series(dtype=float, name="kei_nominal_bn_kzt")
    df = pd.concat([kei, nominal, cpi], axis=1)
    df = df[df.index >= pd.Period(start, "M")].sort_index()
    df["reserves_usd_bn"] = float("nan")
    m = df.index.to_timestamp(how="start")
    nxt = m + pd.offsets.MonthBegin(1)
    df["published_kei"] = (nxt + pd.Timedelta(days=14)).strftime("%Y-%m-%d")      # 15th of M+1
    df["published_cpi"] = (nxt + pd.offsets.BDay(0)).strftime("%Y-%m-%d")       # 1st working day of M+1
    df.loc[df["kei"].isna() & df["kei_nominal_bn_kzt"].isna(), "published_kei"] = None
    df.loc[df["cpi_yoy"].isna(), "published_cpi"] = None
    df.index = df.index.strftime("%Y-%m")
    df.index.name = "month"
    df.to_csv(cache or data_dir() / "macro_monthly_kz.csv")
    return df


def _excel_sheets(data: bytes) -> dict:
    import io
    try:
        return pd.read_excel(io.BytesIO(data), header=None, sheet_name=None)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("для таблиц stat.gov.kz в формате xlsx нужен openpyxl: pip install openpyxl") from exc


# ---------------------------------------------------------------- commodity currencies vs USD
CBAR_URL = "https://www.cbar.az/currencies/{date:%d.%m.%Y}.xml"
CLP_URL = "https://mindicador.cl/api/dolar/{year}"
AZM_PER_AZN = 5000.0      # redenomination of 01.01.2006


def parse_cbar(xml: bytes) -> tuple[pd.Timestamp, float] | None:
    """(rate date, AZN per USD) from one daily bulletin of the Central Bank of Azerbaijan;
    values before 2006 are in old manat (AZM) and are converted at 5000 AZM = 1 AZN."""
    root = ET.fromstring(xml)
    date = pd.to_datetime(root.get("Date"), format="%d.%m.%Y")
    for v in root.iter("Valute"):
        if v.get("Code") == "USD":
            val = float(v.findtext("Value")) / float(v.findtext("Nominal") or 1)
            return date, val / AZM_PER_AZN if date < pd.Timestamp("2006-01-01") else val
    return None


def fetch_azn(start: str = "2000-01-01", end: str | None = None, have: pd.Series | None = None,
              workers: int = 6) -> pd.Series:
    """AZN per USD, daily official rate of cbar.az (one XML per business day); only the days
    missing from ``have`` (an earlier download) are requested."""
    from concurrent.futures import ThreadPoolExecutor
    have = have.dropna() if have is not None else pd.Series(dtype=float)
    days = pd.bdate_range(start, end or pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    days = days[~days.isin(have.index)] if len(have) else days

    def one(d):
        try:
            return parse_cbar(_get(CBAR_URL.format(date=d)))
        except (OSError, ValueError, ET.ParseError):
            return None
    with ThreadPoolExecutor(workers) as ex:
        got = [r for r in ex.map(one, days) if r]
    new = pd.Series(dict(got), dtype=float)
    s = pd.concat([have, new]).groupby(level=0).last().sort_index().rename("AZN")
    s.index.name = "Date"
    return s


def fetch_clp(start: int = 2000, end: int | None = None) -> pd.Series:
    """CLP per USD, "dólar observado" of the Banco Central de Chile, via the mindicador.cl API
    (one request per year; the BCCh's own API needs a registered account)."""
    out = {}
    for y in range(start, (end or pd.Timestamp.today().year) + 1):
        try:
            for r in json.loads(_get(CLP_URL.format(year=y), timeout=60)).get("serie", []):
                out[pd.Timestamp(r["fecha"][:10])] = float(r["valor"])
        except (OSError, ValueError) as exc:
            print(f"  CLP {y}: {exc}")
    return pd.Series(out, name="CLP", dtype=float).sort_index()


def fetch_fx_panel(cache: Path | None = None, start: str = "2000-01-01") -> pd.DataFrame:
    """Daily rates of commodity currencies, units per 1 USD: NOK, BRL (FRED H.10 DEXNOUS, DEXBZUS),
    RUB (cross rate of the official NBK rates in data/fx_rates.csv: USD/KZT ÷ RUB/KZT; cbr.ru does
    not answer from here), AZN (cbar.az), CLP (BCCh via mindicador.cl); data/fx_commodity_panel.csv."""
    out = data_dir()
    cols = {"NOK": fetch_fred("DEXNOUS", start), "BRL": fetch_fred("DEXBZUS", start)}
    fx = out / "fx_rates.csv"
    if fx.exists():
        r = pd.read_csv(fx, index_col=0, parse_dates=True)
        cols["RUB"] = (r["USD"] / r["RUB"]).loc[start:].round(4).rename("RUB")
    path = cache or out / "fx_commodity_panel.csv"
    prev = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    cols["AZN"] = fetch_azn(start, have=prev.get("AZN"))
    cols["CLP"] = fetch_clp(pd.Timestamp(start).year)
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "Date"
    df.to_csv(path)
    return df


def fetch_yahoo(tickers, start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pip install yfinance") from exc
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.ffill().dropna(axis=1, how="all").dropna(how="all")  # drop tickers Yahoo no longer has
    return adjust_demergers(px, NSE_DEMERGERS)


def adjust_demergers(px: pd.DataFrame, events: dict) -> pd.DataFrame:
    """Back-adjust prices before each ex-date by the observed close ratio, so the spin-off
    is not a loss. That day's market move is lost too (ratio method, as for unpaid dividends)."""
    actions = pd.DataFrame({"ticker": list(events), "ex_date": pd.to_datetime(list(events.values())),
                            "method": "ratio", "factor": float("nan")})
    return apply_corporate_actions(px, actions)[0]
