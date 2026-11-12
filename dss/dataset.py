"""KZ-MacroFin: a weekly labelled dataset of the Kazakhstan macro-financial environment.

One row per week (last trading day of the week, t). Every feature uses only data available at t;
monthly and annual series enter with their publication date (columns ``*_asof``), market series
as of t. Labels look forward from t:

* ``y_dev``     - a devaluation episode of the tenge within the next 13 weeks: the Module II detector,
                  USD/KZT up more than 4 % in a day or more than 8 % in 5 trading days;
* ``y_mstress`` - the KASE index falls over the next 13 weeks (minimum of close / close_t - 1) below
                  the 10th percentile of that quantity on the training part (weeks up to TRAIN_END);
* ``y_rec``     - recession state at t: annual GDP growth of the year of t below zero (whole period)
                  or, from 2021, the cumulative short-term economic indicator (KEI) of the month of t
                  below 100 (data/macro_monthly_kz.csv); a nowcast label, not a forecast;
* ``y_rank_<i>`` - tercile (1 lowest .. 3 highest) of the 4-week forward return of stock i among
                  the stocks quoted at t.

The publication dates of KEI and CPI in macro_monthly_kz.csv are assumptions (see data/README.md),
not the actual release dates. :func:`build` takes the raw inputs as a dict so that the leakage test
can truncate them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import loaders

TRAIN_END = "2018-12-31"           # training part for the y_mstress threshold and the classifiers' first fit
HORIZON_W = 13
DEV_DAY, DEV_5D = 0.04, 0.08
COMMODITY_FX = ("NOK", "BRL", "RUB", "CLP")   # AZN is pegged and left out
LABELS = ("y_dev", "y_mstress", "y_rec")


def load_inputs() -> dict:
    d = loaders.data_dir()
    rates, _ = loaders.load_rates()
    px, _ = loaders.raw_prices("kase")
    return {
        "usd": loaders.load_fx().frame["USD"].dropna(),
        "rates": rates,
        "brent": pd.read_csv(d / "brent.csv", index_col=0, parse_dates=True)["brent"],
        "index": pd.read_csv(d / "index_kase.csv", index_col=0, parse_dates=True)["close"],
        "gdp": loaders.load_gdp().frame.loc["Kazakhstan"].dropna(),
        "macro": pd.read_csv(d / "macro_monthly_kz.csv", parse_dates=["published_kei", "published_cpi"]),
        "commodity": pd.read_csv(d / "fx_commodity_panel.csv", index_col=0, parse_dates=True),
        "stocks": px,
    }


def weeks(index: pd.Series) -> pd.DatetimeIndex:
    """Last trading day (of the KASE index) of every week."""
    d = index.dropna().index
    return pd.DatetimeIndex(pd.Series(d, index=d).groupby(d.to_period("W")).last().to_numpy())


def _asof(series: pd.Series, t) -> float:
    s = series.loc[:t]
    return float(s.iloc[-1]) if len(s) else np.nan


def published_monthly(macro: pd.DataFrame, value: str, published: str, t) -> tuple[float, pd.Timestamp]:
    """Latest value of a monthly series whose publication date is on or before t."""
    m = macro.dropna(subset=[value, published])
    m = m[m[published] <= t]
    if m.empty:
        return np.nan, pd.NaT
    r = m.iloc[-1]
    return float(r[value]), pd.Timestamp(r[published])


def devaluation_days(usd: pd.Series) -> pd.DatetimeIndex:
    """Days on which the Module II detector fires (USD/KZT business days)."""
    s = usd.dropna()
    s = s[s.index.dayofweek < 5]
    up1 = s.pct_change() > DEV_DAY
    up5 = s.pct_change(5) > DEV_5D
    return s.index[up1 | up5]


@dataclass
class KZMacroFin:
    table: pd.DataFrame          # weeks x (features, *_asof, labels)
    features: list
    mstress_threshold: float


def build(inp: dict, end=None) -> KZMacroFin:
    idx = inp["index"].loc[:end] if end is not None else inp["index"]
    usd, brent, rates = inp["usd"], inp["brent"], inp["rates"]
    if end is not None:
        usd, brent, rates = usd.loc[:end], brent.loc[:end], rates.loc[:end]
    cfx = inp["commodity"].loc[:end] if end is not None else inp["commodity"]
    W = weeks(idx)
    W = W[W >= pd.Timestamp("2008-03-01")]
    lu, lb, li = np.log(usd), np.log(brent), np.log(idx)
    rows = []
    for t in W:
        r = {"date": t}
        r["usd_ret_4w"] = _asof(lu, t) - _asof(lu, t - pd.Timedelta(weeks=4))
        r["usd_ret_13w"] = _asof(lu, t) - _asof(lu, t - pd.Timedelta(weeks=13))
        du = lu.loc[t - pd.Timedelta(weeks=13):t].diff().dropna()
        r["usd_vol_13w"] = float(du.std() * np.sqrt(252)) if len(du) > 5 else np.nan
        r["brent_ret_4w"] = _asof(lb, t) - _asof(lb, t - pd.Timedelta(weeks=4))
        r["brent_ret_13w"] = _asof(lb, t) - _asof(lb, t - pd.Timedelta(weeks=13))
        cm = [np.log(_asof(cfx[c], t) / _asof(cfx[c], t - pd.Timedelta(weeks=4)))
              for c in COMMODITY_FX if c in cfx]
        r["commodity_fx_4w"] = float(np.nanmean(cm)) if np.isfinite(cm).any() else np.nan
        r["base_rate"] = _asof(rates["kzt"], t)
        r["base_rate_chg_13w"] = r["base_rate"] - _asof(rates["kzt"], t - pd.Timedelta(weeks=13))
        r["rate_spread"] = r["base_rate"] - _asof(rates["usd"], t)
        r["kase_ret_4w"] = _asof(li, t) - _asof(li, t - pd.Timedelta(weeks=4))
        r["kase_ret_13w"] = _asof(li, t) - _asof(li, t - pd.Timedelta(weeks=13))
        di = li.loc[t - pd.Timedelta(weeks=13):t].diff().dropna()
        r["kase_vol_13w"] = float(di.std() * np.sqrt(252)) if len(di) > 5 else np.nan
        hi = idx.loc[t - pd.Timedelta(weeks=52):t]
        r["kase_dd_52w"] = float(hi.iloc[-1] / hi.max() - 1) if len(hi) else np.nan
        r["cpi_yoy"], r["cpi_yoy_asof"] = published_monthly(inp["macro"], "cpi_yoy", "published_cpi", t)
        r["kei"], r["kei_asof"] = published_monthly(inp["macro"], "kei", "published_kei", t)
        m = inp["macro"].dropna(subset=["kei_nominal_bn_kzt", "published_kei"]).copy()
        m["yoy"] = m["kei_nominal_bn_kzt"].pct_change(12)
        r["kei_nominal_yoy"], r["kei_nominal_asof"] = published_monthly(m, "yoy", "published_kei", t)
        y = t.year - 1 if (t.month, t.day) >= (4, 1) else t.year - 2
        r["gdp_growth_last"] = float(inp["gdp"].get(y, np.nan))
        r["gdp_growth_asof"] = pd.Timestamp(f"{y + 1}-04-01")
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("date")
    for c in [c for c in tab.columns if c.endswith("_asof")]:
        tab[c] = pd.to_datetime(tab[c])
    feats = [c for c in tab.columns if not c.endswith("_asof")]
    tab = add_labels(tab, inp, end)
    thr = tab.attrs["mstress_threshold"]
    return KZMacroFin(tab, feats, thr)


def add_labels(tab: pd.DataFrame, inp: dict, end=None) -> pd.DataFrame:
    usd = inp["usd"].loc[:end] if end is not None else inp["usd"]
    idx = inp["index"].loc[:end] if end is not None else inp["index"]
    last = idx.index.max()
    dev = devaluation_days(usd)
    h = pd.Timedelta(weeks=HORIZON_W)
    complete = tab.index + h <= last
    tab["y_dev"] = [float(((dev > t) & (dev <= t + h)).any()) for t in tab.index]
    fwd_dd = [float(idx.loc[t:t + h].iloc[1:].min() / idx.loc[:t].iloc[-1] - 1) if c else np.nan
              for t, c in zip(tab.index, complete)]
    tab["fwd_min_13w"] = fwd_dd
    train = tab.loc[:TRAIN_END, "fwd_min_13w"].dropna()
    thr = float(train.quantile(0.10))
    tab["y_mstress"] = np.where(tab["fwd_min_13w"].notna(), (tab["fwd_min_13w"] < thr).astype(float), np.nan)
    tab["y_dev"] = tab["y_dev"].where(complete)
    gdp, macro = inp["gdp"], inp["macro"].copy()
    macro["month"] = pd.PeriodIndex(macro["month"], freq="M")
    kei = macro.set_index("month")["kei"]
    yrec = []
    for t in tab.index:
        g = gdp.get(t.year, np.nan)
        k = kei.get(t.to_period("M"), np.nan) if t.year >= 2021 else np.nan
        if np.isnan(g) and np.isnan(k):
            yrec.append(np.nan)
        else:
            yrec.append(float((g < 0 if np.isfinite(g) else False) or (k < 100 if np.isfinite(k) else False)))
    tab["y_rec"] = yrec
    # 4-week forward return terciles of the stocks quoted at t
    px = inp["stocks"].loc[:end] if end is not None else inp["stocks"]
    pw = px.reindex(tab.index, method="ffill")
    quoted = px.notna().cummax().reindex(tab.index, method="ffill").fillna(False).astype(bool)
    fwd = pw.shift(-4) / pw - 1
    fwd = fwd.where(quoted)
    rk = fwd.rank(axis=1, pct=True)
    for c in px.columns:
        tab[f"y_rank_{c}"] = np.ceil(rk[c] * 3).clip(1, 3)
    tab.attrs["mstress_threshold"] = thr
    return tab


# ----------------------------------------------------------------------------- regime classifiers
def walk_forward_classifiers(ds: KZMacroFin, label: str, refit_weeks: int = 52, seed: int = 42) -> pd.DataFrame:
    """Expanding-window logistic regression and gradient boosting (sigmoid-calibrated) for ``label``;
    first fit on weeks up to TRAIN_END, refit every ``refit_weeks``. Training rows are those whose
    13-week label window has closed before the first predicted week. Returns weeks x (y, p_logit, p_gb)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [f for f in ds.features if f not in ("kei",)]      # KEI index exists only from 2021
    tab = ds.table
    test_weeks = tab.index[tab.index > pd.Timestamp(TRAIN_END)]
    out = []
    lag = pd.Timedelta(weeks=HORIZON_W) if label != "y_rec" else pd.Timedelta(weeks=0)
    for i in range(0, len(test_weeks), refit_weeks):
        block = test_weeks[i:i + refit_weeks]
        if label == "y_rec":     # the GDP part of the label of year Y is known from 1 April Y + 1
            known = pd.to_datetime([f"{y + 1}-04-01" for y in tab.index.year]) < block[0]
        else:
            known = tab.index + lag < block[0]
        tr = tab[known].dropna(subset=[label])
        if tr[label].nunique() < 2:
            continue
        X, y = tr[feats], tr[label].astype(int)
        logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
        n_pos = int(y.sum())
        gb_base = make_pipeline(SimpleImputer(strategy="median"),
                                GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                           subsample=0.8, random_state=seed))
        gb = (CalibratedClassifierCV(gb_base, method="sigmoid", cv=3).fit(X, y) if n_pos >= 6
              else gb_base.fit(X, y))
        te = tab.loc[block]
        out.append(pd.DataFrame({"y": te[label], "p_logit": logit.predict_proba(te[feats])[:, 1],
                                 "p_gb": gb.predict_proba(te[feats])[:, 1]}, index=block))
    return pd.concat(out) if out else pd.DataFrame(columns=["y", "p_logit", "p_gb"])


def detection_lag(y: pd.Series, p: pd.Series, threshold: float = 0.5) -> tuple[float, int, int]:
    """For every run of y = 1 (an episode), weeks from its first week until p first exceeds ``threshold``
    within the run; returns (mean lag over detected episodes, detected, episodes)."""
    y = y.fillna(0).astype(int)
    runs = (y.diff().fillna(y.iloc[0]) == 1).cumsum().where(y == 1)
    lags, n = [], 0
    for _, g in y[runs.notna()].groupby(runs[runs.notna()]):
        n += 1
        hit = p.reindex(g.index) > threshold
        if hit.any():
            lags.append(int(np.argmax(hit.to_numpy())))
    return (float(np.mean(lags)) if lags else np.nan), len(lags), n


def classifier_metrics(pred: pd.DataFrame) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    d = pred.dropna(subset=["y"])
    y = d["y"].astype(int)
    out = {"n_weeks": len(d), "n_positive": int(y.sum()), "brier_climatology": float(((y - y.mean()) ** 2).mean())}
    for m in ("p_logit", "p_gb"):
        out[f"auc_{m}"] = float(roc_auc_score(y, d[m])) if y.nunique() == 2 else np.nan
        out[f"brier_{m}"] = float(brier_score_loss(y, d[m]))
        lag, det, n = detection_lag(y, d[m])
        out[f"lag_{m}"], out[f"detected_{m}"], out["episodes"] = lag, det, n
    return out
