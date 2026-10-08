"""Synthetic DEMO data.

These generators exist only so that the DSS can be installed and demonstrated
without internet access. They are seeded, shaped to look like the real series
(devaluation regimes of the tenge, commodity cycle, sector structure of KASE),
but they are NOT real data and no result computed on them may be reported as
an empirical finding. Every page of the web UI shows a "DEMO" banner while
they are in use. Real data are loaded by :mod:`dss.data.loaders`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RANDOM_SEED


# --------------------------------------------------------------------------
# Module I - GDP growth panel (wide: country x year), like world_gdp_data.csv
# --------------------------------------------------------------------------
_GLOBAL_SHOCKS = {1982: -1.5, 1983: -1.0, 1991: -0.8, 1998: -1.2, 2001: -0.7,
                  2009: -3.5, 2020: -6.0, 2021: 2.5}

_ARCHETYPES = {
    #            mean  vol  persistence  shock beta
    "stable": (2.3, 1.6, 0.45, 0.9),
    "low":    (1.6, 3.2, 0.35, 0.8),
    "high":   (5.8, 2.6, 0.55, 0.7),
    "crisis": (1.2, 6.0, 0.25, 1.2),
}


def gdp_panel(n_countries: int = 160, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.arange(1980, 2025)
    rows, names = [], []
    kinds = list(_ARCHETYPES)
    for c in range(n_countries):
        kind = kinds[c % 4]
        mu, vol, phi, beta = _ARCHETYPES[kind]
        mu += rng.normal(0, 0.6)
        y, prev = [], mu
        for yr in years:
            x = mu + phi * (prev - mu) + rng.normal(0, vol) + beta * _GLOBAL_SHOCKS.get(yr, 0.0)
            if kind == "crisis" and rng.random() < 0.05:
                x -= rng.uniform(5, 15)
            y.append(x)
            prev = x
        y = np.array(y)
        start = 0
        if rng.random() < 0.15:          # post-1990 states / poor coverage
            start = int(rng.integers(10, 14))
        y[:start] = np.nan
        rows.append(y)
        names.append(f"Country {c + 1:03d} ({kind})")

    # Synthetic "Kazakhstan": transition recession, commodity boom, moderation.
    kz = np.full(len(years), np.nan)
    for i, yr in enumerate(years):
        if yr < 1993:
            continue
        if yr <= 1998:
            base = -9.0 + (yr - 1993) * 1.8
        elif yr <= 2007:
            base = 9.5
        elif yr <= 2014:
            base = 5.0
        else:
            base = 3.6
        kz[i] = base + rng.normal(0, 1.4) + 0.8 * _GLOBAL_SHOCKS.get(yr, 0.0)
    rows.append(kz)
    names.append("Kazakhstan")
    return pd.DataFrame(rows, index=pd.Index(names, name="country"), columns=years)


# --------------------------------------------------------------------------
# Module II - daily official rates, KZT per unit of foreign currency
# --------------------------------------------------------------------------
def _anchored_path(dates, anchors, vol, rng, jumps=()):
    """Log-interpolate anchor points, add OU noise and step devaluations."""
    t = dates.values.astype("datetime64[D]").astype(np.int64)
    at = np.array([np.datetime64(d).astype("datetime64[D]").astype(np.int64) for d, _ in anchors])
    av = np.log([v for _, v in anchors])
    base = np.interp(t, at, av)
    noise = np.zeros(len(t))
    for i in range(1, len(t)):
        noise[i] = 0.985 * noise[i - 1] + rng.normal(0, vol)
    path = base + noise
    return np.exp(path)


def fx_rates(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range("2000-01-01", "2024-12-07", freq="D")
    usd = _anchored_path(dates, [
        ("2000-01-01", 140), ("2002-06-01", 154), ("2007-12-01", 120), ("2009-02-03", 120.5),
        ("2009-02-05", 150), ("2014-02-10", 155), ("2014-02-12", 184), ("2015-08-19", 198),
        ("2015-08-21", 255), ("2016-01-20", 375), ("2018-06-01", 335), ("2019-12-01", 385),
        ("2020-04-01", 440), ("2021-06-01", 427), ("2022-03-10", 500), ("2022-06-01", 432),
        ("2023-06-01", 450), ("2024-06-01", 448), ("2024-12-07", 520)], 0.0028, rng)
    eurusd = _anchored_path(dates, [
        ("2000-01-01", 1.01), ("2001-07-01", 0.85), ("2004-12-01", 1.33), ("2008-07-01", 1.57),
        ("2010-06-01", 1.22), ("2014-05-01", 1.38), ("2015-03-01", 1.08), ("2018-02-01", 1.24),
        ("2022-09-27", 0.97), ("2023-07-01", 1.10), ("2024-12-07", 1.06)], 0.0018, rng)
    usdcny = _anchored_path(dates, [
        ("2000-01-01", 8.28), ("2005-07-20", 8.28), ("2008-07-01", 6.83), ("2010-06-01", 6.83),
        ("2014-01-01", 6.05), ("2016-12-01", 6.95), ("2018-03-01", 6.28), ("2020-05-01", 7.12),
        ("2022-03-01", 6.32), ("2023-09-01", 7.30), ("2024-12-07", 7.27)], 0.0007, rng)
    usdrub = _anchored_path(dates, [
        ("2000-01-01", 27.0), ("2002-12-01", 31.8), ("2008-07-01", 23.4), ("2009-02-01", 36.0),
        ("2013-12-01", 33.0), ("2014-12-16", 68.0), ("2016-01-21", 80.0), ("2017-05-01", 56.0),
        ("2020-04-01", 77.0), ("2021-06-01", 72.0), ("2022-03-10", 125.0), ("2022-06-30", 54.0),
        ("2023-10-01", 98.0), ("2024-06-01", 89.0), ("2024-12-07", 103.0)], 0.0045, rng)
    df = pd.DataFrame({
        "USD": usd,
        "EUR": usd * eurusd,
        "CNY": usd / usdcny,
        "RUB": usd / usdrub,
    }, index=pd.Index(dates, name="Date"))
    return df.round(2)


# --------------------------------------------------------------------------
# Modules III-IV - equity prices and fundamentals
# --------------------------------------------------------------------------
KASE_DEMO = {
    # ticker: (sector, annual drift, annual idio vol, PB, PE)
    "HSBK": ("Финансы", 0.14, 0.24, 1.3, 4.5),
    "KSPI": ("Финансы", 0.20, 0.32, 9.5, 11.0),
    "CCBN": ("Финансы", 0.18, 0.30, 1.1, 3.8),
    "KZTK": ("Телеком", 0.09, 0.26, 1.2, 7.5),
    "KCEL": ("Телеком", 0.07, 0.28, 2.6, 9.0),
    "KEGC": ("Энергетика", 0.05, 0.18, 0.9, 12.0),
    "KZTO": ("Энергетика", 0.04, 0.20, 0.8, 9.5),
    "KMGZ": ("Нефть и газ", 0.12, 0.25, 0.6, 5.5),
    "KZAP": ("Уран", 0.16, 0.34, 3.4, 10.5),
    "AIRA": ("Транспорт", 0.08, 0.36, 4.1, 8.0),
    "BAST": ("Металлургия", 0.10, 0.40, 1.9, 6.5),
    "KZIP": ("Энергетика", 0.03, 0.22, 0.7, 14.0),
}


def equity_prices(tickers: dict | None = None, start: str = "2019-01-01",
                  end: str = "2025-09-30", seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Factor-model GBM prices (market + sector + idiosyncratic, regime drifts)."""
    tickers = tickers or KASE_DEMO
    rng = np.random.default_rng(seed + 2)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    dt = 1 / 252
    market = rng.normal(0.0, 0.16 * np.sqrt(dt), n)
    # crisis-like episodes: early 2020 and 2022
    for a, b, shock in (("2020-02-20", "2020-04-01", -0.012), ("2022-01-03", "2022-03-15", -0.006)):
        mask = (dates >= a) & (dates <= b)
        market[mask] += shock
    sectors = sorted({s for s, *_ in tickers.values()})
    sector_f = {s: rng.normal(0, 0.10 * np.sqrt(dt), n) for s in sectors}
    out = {}
    for tk, (sec, mu, vol, *_rest) in tickers.items():
        regime = np.repeat(rng.normal(mu, 0.10, n // 126 + 1), 126)[:n]  # half-year regimes
        beta = rng.uniform(0.7, 1.3)
        r = (regime - 0.5 * vol ** 2) * dt + beta * market + sector_f[sec] + rng.normal(0, vol * np.sqrt(dt), n)
        out[tk] = 1000 * np.exp(np.cumsum(r)) * rng.uniform(0.5, 3)
    return pd.DataFrame(out, index=pd.Index(dates, name="Date")).round(2)


def fundamentals(tickers: dict | None = None) -> pd.DataFrame:
    tickers = tickers or KASE_DEMO
    return pd.DataFrame(
        [(tk, s, pb, pe) for tk, (s, _m, _v, pb, pe) in tickers.items()],
        columns=["ticker", "sector", "PB", "PE"]).set_index("ticker")
