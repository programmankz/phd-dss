"""Data access layer.

Resolution order for every dataset (``source="auto"``):

1. a user CSV in the data directory (``DSS_DATA_DIR`` or ``./data``),
2. the synthetic DEMO generator (flagged with ``is_demo=True``).

``python -m dss.cli fetch`` downloads the real series from the original
sources used in the articles and writes them to the data directory:

* GDP growth  - IMF World Economic Outlook DataMapper API, indicator NGDP_RPCH  [A1]
* FX rates    - National Bank of the Republic of Kazakhstan, daily official rates [A2]
* NSE prices  - Yahoo Finance via ``yfinance`` (optional dependency)             [A4]
* KASE prices - CSV exported from kase.kz (no public API; see README)             [A3]

Expected CSV layouts
--------------------
gdp_growth.csv    : country,1980,1981,...,2024           (wide, % change)
fx_rates.csv      : Date,USD,EUR,CNY,RUB                 (KZT per unit)
prices_<u>.csv    : Date,<TICKER1>,<TICKER2>,...         (close prices)
fundamentals_<u>.csv : ticker,sector,PB,PE
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
# Auto-resolving loaders
# --------------------------------------------------------------------------
def load_gdp(source: str = "auto") -> Dataset:
    path = data_dir() / "gdp_growth.csv"
    if source in ("auto", "csv") and path.exists():
        df = pd.read_csv(path)
        df = df.set_index(df.columns[0])
        df.columns = [int(float(c)) for c in df.columns]
        df.index.name = "country"
        return Dataset(df.apply(pd.to_numeric, errors="coerce"), f"CSV: {path.name}", False)
    if source == "live":
        return Dataset(fetch_imf_gdp(), "IMF WEO DataMapper (NGDP_RPCH)", False)
    return Dataset(demo.gdp_panel(), "DEMO: синтетическая панель", True)


def load_fx(source: str = "auto") -> Dataset:
    path = data_dir() / "fx_rates.csv"
    if source in ("auto", "csv") and path.exists():
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        return Dataset(df[[c for c in ("USD", "EUR", "CNY", "RUB") if c in df]], f"CSV: {path.name}", False)
    return Dataset(demo.fx_rates(), "DEMO: синтетические курсы", True)


def load_prices(universe: str = "kase", source: str = "auto") -> Dataset:
    path = data_dir() / f"prices_{universe}.csv"
    if source in ("auto", "csv") and path.exists():
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna(axis=1, how="all").dropna()
        return Dataset(df, f"CSV: {path.name}", False)
    if source == "live" and universe == "nse":
        return Dataset(fetch_yahoo(NSE_TICKERS), "Yahoo Finance (NSE)", False)
    return Dataset(demo.equity_prices(), "DEMO: синтетические цены (тикеры KASE)", True)


def load_fundamentals(universe: str = "kase", tickers=None) -> Dataset:
    path = data_dir() / f"fundamentals_{universe}.csv"
    if path.exists():
        df = pd.read_csv(path).set_index("ticker")
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
