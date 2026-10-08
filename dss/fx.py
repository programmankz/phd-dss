"""Module II - tenge exchange-rate risk.  [A2]

Reproduces the analysis of [A2] (descriptive statistics in the layout of
Table 1, Pearson correlations including the date, linear-trend regression and
R^2 per currency) and turns it into signals for the DSS:

* devaluation-shock detector (2009, 2014, 2015 ... in real data);
* the "single risk factor" check - share of variance of USD/EUR/CNY levels
  explained by their first principal component;
* an FX risk index in [0, 1] combining volatility regime and momentum of the
  tenge against each currency (RUB treated as a separate factor);
* expected annual depreciation of the tenge, used to convert returns to KZT
  and to size the USD hedge of the safe part of the portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics with the rows of [A2] Table 1 (Excel conventions)."""
    out = pd.DataFrame({
        "Среднее": df.mean(),
        "Стандартная ошибка": df.std() / np.sqrt(df.count()),
        "Медиана": df.median(),
        "Мода": df.round(2).mode().iloc[0],
        "Стандартное отклонение": df.std(),
        "Дисперсия выборки": df.var(),
        "Эксцесс": df.kurt(),
        "Асимметрия": df.skew(),
        "Размах": df.max() - df.min(),
        "Минимум": df.min(),
        "Максимум": df.max(),
        "Сумма": df.sum(),
        "Счёт": df.count(),
    }).T
    return out[df.columns]


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix including the date ([A2] Fig. 3)."""
    d = df.copy()
    d.insert(0, "Date", d.index.map(pd.Timestamp.toordinal))
    return d.corr()


def trend_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rate_t = w * ordinal(date) + b per currency ([A2] Eq. 1-2, Fig. 5)."""
    x = df.index.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    rows, fitted = [], {}
    for c in df.columns:
        m = LinearRegression().fit(x, df[c].to_numpy())
        fitted[c] = m.predict(x)
        rows.append({"currency": f"{c}/KZT", "Наклон, тг/год": m.coef_[0] * 365.25,
                     "R2": m.score(x, df[c].to_numpy())})
    return pd.DataFrame(rows).set_index("currency"), pd.DataFrame(fitted, index=df.index)


def detect_shocks(df: pd.DataFrame, col: str = "USD", threshold: float = 0.04,
                  min_gap_days: int = 120) -> pd.DataFrame:
    """Devaluation episodes: daily KZT depreciation above ``threshold`` or
    cumulative 5-day move above 2 x threshold, merged within ``min_gap_days``."""
    r = np.log(df[col]).diff()
    r5 = np.log(df[col]).diff(5)
    hits = r[(r > threshold) | (r5 > 2 * threshold)]
    events, last = [], None
    for d in hits.index:
        if last is None or (d - last).days > min_gap_days:
            window = df[col].loc[d - pd.Timedelta(days=10): d + pd.Timedelta(days=60)]
            before = df[col].loc[: d - pd.Timedelta(days=10)].iloc[-1] if (df.index < d - pd.Timedelta(days=10)).any() else window.iloc[0]
            events.append({"date": d.date(), "before": before, "after_60d": window.iloc[-1],
                           "move_pct": (window.iloc[-1] / before - 1) * 100})
        last = d
    return pd.DataFrame(events)


def common_factor_share(df: pd.DataFrame, cols=("USD", "EUR", "CNY")) -> float:
    """Share of variance of standardised levels explained by PC1."""
    z = df[list(cols)]
    z = (z - z.mean()) / z.std()
    ev = np.linalg.eigvalsh(np.cov(z.to_numpy().T))
    return float(ev.max() / ev.sum())


@dataclass
class FxResult:
    period: tuple
    n_obs: int
    stats: pd.DataFrame
    corr: pd.DataFrame
    regression: pd.DataFrame
    fitted: pd.DataFrame
    shocks: pd.DataFrame
    pc1_share: float
    risk_components: pd.DataFrame
    fx_risk: float
    exp_depreciation: float     # expected annual KZT depreciation vs USD, fraction
    regime: str
    rates: pd.DataFrame


def risk_signal(df: pd.DataFrame, vol_w: int = config.FX_VOL_WINDOW,
                mom_w: int = config.FX_MOMENTUM_WINDOW) -> tuple[pd.DataFrame, float]:
    """Per-currency risk components and the weighted FX risk index in [0, 1].

    vol_pct - percentile of the current 30-day realised volatility in its history
    mom_pct - percentile of the current 60-day depreciation of the tenge
    """
    lr = np.log(df).diff().dropna()
    vol = lr.rolling(vol_w).std() * np.sqrt(365)
    mom = np.log(df).diff(mom_w)
    rows = []
    for c in df.columns:
        v, mmt = vol[c].dropna(), mom[c].dropna()
        vp = float((v < v.iloc[-1]).mean())
        mp = float((mmt < mmt.iloc[-1]).mean())
        rows.append({"currency": c, "vol_30d": v.iloc[-1], "vol_pct": vp,
                     "depr_60d": mmt.iloc[-1], "mom_pct": mp,
                     "risk": 0.5 * vp + 0.5 * mp,
                     "weight": config.FX_RISK_WEIGHTS.get(c, 0.0)})
    comp = pd.DataFrame(rows).set_index("currency")
    w = comp["weight"] / comp["weight"].sum()
    return comp, float((comp["risk"] * w).sum())


def expected_depreciation(df: pd.DataFrame, col: str = "USD") -> float:
    """Blend of 1-year and 5-year log drift of USD/KZT, annualised."""
    s = np.log(df[col])
    end = s.index[-1]
    d1 = s.iloc[-1] - s.loc[: end - pd.Timedelta(days=365)].iloc[-1]
    d5 = (s.iloc[-1] - s.loc[: end - pd.Timedelta(days=5 * 365)].iloc[-1]) / 5
    return float(np.expm1(0.5 * d1 + 0.5 * d5))


def run(rates: pd.DataFrame) -> FxResult:
    rates = rates.dropna()
    reg, fitted = trend_regression(rates)
    comp, risk = risk_signal(rates)
    regime = "Высокий" if risk >= 0.66 else "Повышенный" if risk >= 0.4 else "Спокойный"
    return FxResult(
        period=(rates.index[0].date(), rates.index[-1].date()), n_obs=len(rates),
        stats=describe(rates), corr=correlations(rates), regression=reg, fitted=fitted,
        shocks=detect_shocks(rates), pc1_share=common_factor_share(rates),
        risk_components=comp, fx_risk=risk, exp_depreciation=expected_depreciation(rates),
        regime=regime, rates=rates,
    )
