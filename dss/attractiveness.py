"""Investment attractiveness of Kazakhstan: the KASE index as a leading indicator of GDP growth.

The indicator is binary, A_t = 1 if the chosen condition on real GDP growth holds in period t and 0
otherwise. The user chooses the definition:

* ``sign``  - growth is positive, g_t > 0 (quarterly data by default);
* ``accel`` - growth accelerates, g_t - g_{t-1} > 0 (annual data by default);
* ``panel`` - g_t > 0, annual, classifiers trained on the country panel of [A1] and applied to
  Kazakhstan (the panel has no stock index, so the KASE index is not a predictor in this mode).

Every forecast of period t uses only information available before t: growth up to t-1 and the
return of the KASE index over ``window_months`` ending on the last trading day before t.
The main model is ARIMA (d by the Dickey-Fuller test, p and q by AIC); ARIMA + KASE is the same
model with the index return as an exogenous regressor, estimated on the same sample. The ARIMA grid
of [A1] on the full growth history is kept for comparison; AR(1) with and without the KASE return shows what the index adds;
logistic regression, random forest and gradient boosting are the machine learning models. All
models are compared out of sample on an expanding window. Nothing here is tuned on the sample.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import config, macro
from .data import loaders

DEFINITIONS = {
    "sign": {"label": "Рост ВВП положителен: g > 0", "freq": "quarterly"},
    "accel": {"label": "Рост ВВП ускоряется: ΔY = g(t) − g(t−1) > 0", "freq": "annual"},
    "panel": {"label": "Рост ВВП положителен, обучение на панели стран [A1]", "freq": "annual"},
}
FREQS = {"annual": "годовые", "quarterly": "квартальные"}
WINDOW_MONTHS = 6            # KASE return window; set in advance, the lag profile is reported for reference
ROLL = {"annual": 3, "quarterly": 4}        # periods in the rolling mean and standard deviation
MIN_TRAIN = {"annual": 10, "quarterly": 24}
FEATURES = ["g_lag1", "g_mean", "g_std", "kase"]
FEATURE_LABELS = {"g_lag1": "Рост в предыдущем периоде", "g_mean": "Средний темп роста",
                  "g_std": "Разброс темпа роста (ст. откл.)", "kase": "Доходность индекса KASE"}
ARIMA_REF = "ARIMA, сетка [A1]"
ARIMAX = "ARIMA + KASE"
REG_MODELS = ("ARIMA", ARIMAX, ARIMA_REF, "AR(1)", "AR(1) + KASE")
ARIMA_GRID = {0: ((1, 0), (2, 0), (1, 1), (2, 1)), 1: ((0, 1), (1, 0), (2, 0), (1, 1))}   # (p, q) by order of integration
CLUSTER_START = 2000         # country clustering period starts with the KASE index
ML_MODELS = ("Логистическая регрессия", "Случайный лес", "Градиентный бустинг")
BASE = "Постоянный прогноз (доля в обучении)"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def kase_return(index: pd.Series, periods: pd.PeriodIndex, window_months: int = WINDOW_MONTHS,
                shift_periods: int = 0) -> pd.Series:
    """Return of the index, %, over ``window_months`` ending on the last trading day before each
    period (``shift_periods`` moves the window further back by whole periods)."""
    out = {}
    for p in periods:
        end = (p - shift_periods).start_time - pd.Timedelta(days=1)
        start = end - pd.DateOffset(months=window_months)
        a, b = index[:start], index[:end]
        if len(a) and len(b) and (start - a.index[-1]).days <= 31 and (end - b.index[-1]).days <= 31:
            out[p] = (b.iloc[-1] / a.iloc[-1] - 1.0) * 100.0
    return pd.Series(out, dtype=float)


def growth_series(freq: str, gdp_wide: pd.DataFrame | None = None,
                  country: str = config.TARGET_COUNTRY) -> pd.Series:
    if freq == "quarterly":
        return loaders.load_gdp_quarterly_kz()
    wide = gdp_wide if gdp_wide is not None else loaders.load_gdp().frame
    s = wide.loc[country].dropna().astype(float)
    return pd.Series(s.to_numpy(), index=pd.PeriodIndex([int(y) for y in s.index], freq="Y"), name="growth")


def target(g: pd.Series, definition: str) -> pd.Series:
    y = (g.diff() > 0) if definition == "accel" else (g > 0)
    y = y.astype(float)
    if definition == "accel":
        y.iloc[0] = np.nan
    return y


def build_dataset(definition: str, freq: str, g: pd.Series, index: pd.Series,
                  window_months: int = WINDOW_MONTHS) -> pd.DataFrame:
    k = ROLL[freq]
    past = g.shift(1)
    df = pd.DataFrame({"g": g, "y": target(g, definition), "g_lag1": past,
                       "g_mean": past.rolling(k).mean(), "g_std": past.rolling(k).std(ddof=0),
                       "kase": kase_return(index, g.index, window_months)})
    return df.dropna()


# --------------------------------------------------------------------------
# Small econometric helpers (numpy only)
# --------------------------------------------------------------------------
def ols_hac(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS with a constant and Newey-West (Bartlett) standard errors."""
    n = len(y)
    Z = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ beta
    lags = int(np.floor(4 * (n / 100.0) ** (2 / 9))) if lags is None else lags
    u = Z * e[:, None]
    S = u.T @ u
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        G = u[j:].T @ u[:-j]
        S += w * (G + G.T)
    bread = np.linalg.inv(Z.T @ Z)
    cov = bread @ S @ bread * n / max(n - Z.shape[1], 1)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), max(n - Z.shape[1], 1))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(e @ e) / tss if tss > 0 else np.nan
    return {"beta": beta, "se": se, "p": p, "resid": e, "r2": r2,
            "sigma": float(np.sqrt(e @ e / max(n - Z.shape[1], 1)))}


def arch_lm(resid: np.ndarray, lags: int = 1) -> tuple[float, float]:
    """Engle's ARCH-LM test: n * R^2 of e_t^2 on its own lags."""
    e2 = resid ** 2
    y = e2[lags:]
    X = np.column_stack([e2[lags - j - 1:len(e2) - j - 1] for j in range(lags)])
    lm = len(y) * ols_hac(y, X, lags=0)["r2"]
    return float(lm), float(stats.chi2.sf(lm, lags))


def breusch_pagan(resid: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Breusch-Pagan (Koenker) test: n * R^2 of e^2 on the regressors."""
    X = np.asarray(X, float).reshape(len(resid), -1)
    lm = len(resid) * ols_hac(resid ** 2, X, lags=0)["r2"]
    return float(lm), float(stats.chi2.sf(lm, X.shape[1]))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """One-step Diebold-Mariano test on squared errors (positive statistic: the second model is better),
    with the small-sample correction of Harvey, Leybourne and Newbold."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(n)) * np.sqrt((n - 1) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), n - 1))


def adf_stat(y: np.ndarray) -> tuple[float, float]:
    """Augmented Dickey-Fuller statistic (constant, one lagged difference) and the 5% critical value
    from MacKinnon's response surface for the sample size."""
    y = np.asarray(y, float)
    dy = np.diff(y)
    X = np.column_stack([y[1:-1], dy[:-1]])
    r = ols_hac(dy[1:], X, lags=0)
    n = len(dy) - 1
    plain_se = r["sigma"] * np.sqrt(np.diag(np.linalg.inv(np.column_stack([np.ones(n), X]).T
                                                          @ np.column_stack([np.ones(n), X]))))
    return float(r["beta"][1] / plain_se[1]), float(-2.8621 - 2.738 / n - 8.36 / n ** 2)


def _arma_css(y: np.ndarray, p: int, q: int, x: np.ndarray | None = None, x_next: float = 0.0,
              const: bool = True) -> dict:
    """ARMA(p, q) with an optional constant and exogenous regressor by conditional least squares
    (q in {0, 1}); one-step forecast (``x_next`` is the regressor of the forecast period)."""
    from scipy.optimize import least_squares
    y = np.asarray(y, float)
    n = len(y)
    k = 0 if x is None else 1
    xv = np.zeros(n) if x is None else np.asarray(x, float)

    def resid(theta):
        c, phi, ma = theta[0] * const, theta[1:1 + p], (theta[1 + p] if q else 0.0)
        beta = theta[-1] if k else 0.0
        e = np.zeros(n)
        for t in range(p, n):
            e[t] = y[t] - c - phi @ y[t - p:t][::-1] - beta * xv[t] - ma * e[t - 1]
        return e[p:]

    start = np.r_[y.mean() * const, np.zeros(p + q + k)]
    lo = np.r_[-np.inf, -0.99 * np.ones(p + q), -np.inf * np.ones(k)]
    hi = np.r_[np.inf, 0.99 * np.ones(p + q), np.inf * np.ones(k)]
    sol = least_squares(resid, start, bounds=(lo, hi))
    e = resid(sol.x)
    m = len(e)
    sigma2 = float(e @ e / m)
    c, phi, ma = sol.x[0] * const, sol.x[1:1 + p], (sol.x[1 + p] if q else 0.0)
    beta = float(sol.x[-1]) if k else 0.0
    fc = float(c + phi @ y[n - p:][::-1] + beta * x_next + ma * e[-1])
    return {"aic": m * np.log(sigma2) + 2 * (p + q + k + 1 + const), "forecast": fc, "sigma": float(np.sqrt(sigma2)), "beta": beta}


def arima_forecast(y: np.ndarray, x: np.ndarray | None = None, x_next: float | None = None) -> dict:
    """ARIMA(p, d, q): d by the Dickey-Fuller test at 5%, p and q by AIC within ``ARIMA_GRID[d]``.
    With ``x`` (aligned with ``y``) and ``x_next`` the same order is re-estimated with the regressor
    (ARIMAX); for d = 1 the regressor is differenced as well. Returns both forecasts."""
    y = np.asarray(y, float)
    stat, crit = adf_stat(y)
    d = 0 if stat < crit else 1
    z = y if d == 0 else np.diff(y)
    const = d == 0                      # no drift in the differenced model, as in [A1]
    fits = {(p, q): _arma_css(z, p, q, const=const) for p, q in ARIMA_GRID[d]}
    (p, q), best = min(fits.items(), key=lambda kv: kv[1]["aic"])
    out = {"order": (p, d, q), "forecast": best["forecast"] + (y[-1] if d else 0.0), "sigma": best["sigma"],
           "adf": stat, "adf_crit": crit}
    if x is not None:
        x = np.asarray(x, float)
        zx, nxt = (x, float(x_next)) if d == 0 else (np.diff(x), float(x_next) - x[-1])
        ex = _arma_css(z, p, q, zx, nxt, const=const)
        out.update(forecast_x=ex["forecast"] + (y[-1] if d else 0.0), sigma_x=ex["sigma"], beta=ex["beta"])
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _classifiers(seed: int) -> dict:
    return {ML_MODELS[0]: LogisticRegression(C=1.0, max_iter=1000),
            ML_MODELS[1]: RandomForestClassifier(n_estimators=300, max_depth=3, random_state=seed),
            ML_MODELS[2]: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                     random_state=seed)}


# --------------------------------------------------------------------------
# One-step forecasts
# --------------------------------------------------------------------------
def _prob(definition: str, mean: float, sd: float, prev: float) -> float:
    """P(A = 1) from a point forecast of growth and its standard error (normal errors)."""
    threshold = prev if definition == "accel" else 0.0
    return float(stats.norm.cdf((mean - threshold) / max(sd, 1e-9)))


def _step(definition: str, hist: pd.Series, train: pd.DataFrame, x: pd.Series, seed: int) -> dict:
    """Forecasts of all models for one period. ``hist`` is growth before the period (ARIMA uses
    all of it), ``train`` holds the rows with complete predictors, ``x`` the predictors of the period."""
    out = {}
    prev = float(x["g_lag1"])
    with warnings.catch_warnings():          # optimiser start-value and convergence messages on short samples
        warnings.simplefilter("ignore")
        fit, _ = macro.arima_select(hist.to_numpy(float), 1)
    sd = float((fit.upper[0] - fit.lower[0]) / 3.92)
    out[ARIMA_REF] = (float(fit.forecast[0]), _prob(definition, float(fit.forecast[0]), sd, prev))
    # ARIMA and ARIMA + KASE share the sample (periods with the index), the order and the estimator
    own = arima_forecast(np.r_[train["g"].to_numpy(float)], train["kase"].to_numpy(float), float(x["kase"]))
    out["ARIMA"] = (own["forecast"], _prob(definition, own["forecast"], own["sigma"], prev))
    out[ARIMAX] = (own["forecast_x"], _prob(definition, own["forecast_x"], own["sigma_x"], prev))
    for name, cols in (("AR(1)", ["g_lag1"]), ("AR(1) + KASE", ["g_lag1", "kase"])):
        r = ols_hac(train["g"].to_numpy(float), train[cols].to_numpy(float), lags=0)
        mean = float(r["beta"][0] + r["beta"][1:] @ x[cols].to_numpy(float))
        out[name] = (mean, _prob(definition, mean, r["sigma"], prev))
    yt = train["y"].to_numpy(int)
    share = float(yt.mean())
    out[BASE] = (np.nan, share)
    mu, sg = train[FEATURES].mean(), train[FEATURES].std(ddof=0).replace(0, 1.0)
    Xt, xs = ((train[FEATURES] - mu) / sg).to_numpy(), ((x[FEATURES] - mu) / sg).to_numpy(float)[None, :]
    for name, model in _classifiers(seed).items():
        if len(np.unique(yt)) < 2:
            out[name] = (np.nan, share)
        else:
            model.fit(Xt, yt)
            out[name] = (np.nan, float(model.predict_proba(xs)[0, 1]))
    return out


def walk_forward(definition: str, data: pd.DataFrame, g: pd.Series, min_train: int,
                 seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window forecasts: probabilities and growth forecasts by model."""
    probs, means = {}, {}
    for i in range(min_train, len(data)):
        period = data.index[i]
        res = _step(definition, g[g.index < period], data.iloc[:i], data.iloc[i], seed)
        probs[period] = {k: v[1] for k, v in res.items()}
        means[period] = {k: v[0] for k, v in res.items() if k in REG_MODELS}
    return pd.DataFrame(probs).T, pd.DataFrame(means).T


def metrics(y: pd.Series, g: pd.Series, probs: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    yv = y.to_numpy(int)
    for name in probs.columns:
        p = probs[name].to_numpy(float)
        rows[name] = {"Точность": float(((p >= 0.5).astype(int) == yv).mean()),
                      "Оценка Брайера": float(((p - yv) ** 2).mean()), "ROC-AUC": _auc(yv, p),
                      "RMSE роста, п.п.": (float(np.sqrt(((means[name] - g) ** 2).mean()))
                                           if name in means else np.nan)}
    order = [BASE, *REG_MODELS, *ML_MODELS]
    return pd.DataFrame(rows).T.loc[[m for m in order if m in rows]]


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class Attractiveness:
    definition: str
    freq: str
    window_months: int
    data: pd.DataFrame                 # g, y and predictors by period
    probs: pd.DataFrame                # out-of-sample P(A = 1) by model
    means: pd.DataFrame                # out-of-sample growth forecasts of the regression models
    metrics: pd.DataFrame
    coefficients: pd.DataFrame         # full-sample AR(1) + KASE with Newey-West errors
    logit: pd.DataFrame                # standardised logistic coefficients
    lag_profile: pd.DataFrame          # correlation of growth with the KASE return by lead
    diagnostics: dict
    signal: dict                       # forecast for the next period
    panel: pd.DataFrame | None = None  # panel mode: metrics on all countries
    notes: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return DEFINITIONS[self.definition]["label"]


def _diagnostics(data: pd.DataFrame, g: pd.Series, means: pd.DataFrame) -> dict:
    full = ols_hac(data["g"].to_numpy(float), data[["g_lag1", "kase"]].to_numpy(float))
    e = full["resid"]
    half = len(data) // 2
    def _dm(a, b):
        return (diebold_mariano((means[a] - g.reindex(means.index)).to_numpy(float),
                                (means[b] - g.reindex(means.index)).to_numpy(float))
                if len(means) else (np.nan, np.nan))
    dm = _dm("AR(1)", "AR(1) + KASE")
    return {"arch_lm": arch_lm(e, 1), "breusch_pagan": breusch_pagan(e, data[["g_lag1", "kase"]].to_numpy(float)),
            "std_first_half": float(data["g"].iloc[:half].std(ddof=1)),
            "std_second_half": float(data["g"].iloc[half:].std(ddof=1)),
            "split": str(data.index[half]), "mean_growth": float(data["g"].mean()),
            "std_growth": float(data["g"].std(ddof=1)), "r2": full["r2"], "dm_ar_vs_kase": dm,
            "n": int(len(data)), "share_positive": float(data["y"].mean()),
            "dm_arima_vs_arimax": _dm("ARIMA", ARIMAX),
            "arima": arima_forecast(data["g"].to_numpy(float), data["kase"].to_numpy(float), 0.0)}


def _coefficients(data: pd.DataFrame) -> pd.DataFrame:
    r = ols_hac(data["g"].to_numpy(float), data[["g_lag1", "kase"]].to_numpy(float))
    return pd.DataFrame({"Коэффициент": r["beta"], "Ст. ошибка (HAC)": r["se"], "p": r["p"]},
                        index=["Константа", FEATURE_LABELS["g_lag1"], FEATURE_LABELS["kase"]])


def _logit(data: pd.DataFrame) -> tuple[pd.DataFrame, LogisticRegression | None, pd.Series, pd.Series]:
    mu, sg = data[FEATURES].mean(), data[FEATURES].std(ddof=0).replace(0, 1.0)
    y = data["y"].to_numpy(int)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), None, mu, sg
    model = LogisticRegression(C=1.0, max_iter=1000).fit(((data[FEATURES] - mu) / sg).to_numpy(), y)
    table = pd.DataFrame({"Коэффициент (на 1 ст. откл.)": model.coef_[0]},
                         index=[FEATURE_LABELS[f] for f in FEATURES])
    return table, model, mu, sg


def _lag_profile(g: pd.Series, index: pd.Series, window_months: int, leads: int = 4) -> pd.DataFrame:
    rows = {}
    for k in range(leads + 1):
        x = kase_return(index, g.index, window_months, shift_periods=k)
        both = pd.concat([g, x], axis=1).dropna()
        if len(both) > 5:
            r, p = stats.pearsonr(both.iloc[:, 0], both.iloc[:, 1])
            rows[k] = {"Корреляция": float(r), "p": float(p), "n": int(len(both))}
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int).astype(str)
    out.index.name = "Сдвиг окна назад, периодов"
    return out


def _signal(definition: str, data: pd.DataFrame, g: pd.Series, index: pd.Series, freq: str,
            window_months: int, seed: int) -> dict:
    period = g.index[-1] + 1
    k = ROLL[freq]
    kase = kase_return(index, pd.PeriodIndex([period]), window_months)
    if kase.empty:
        return {"period": str(period), "available": False}
    x = pd.Series({"g_lag1": float(g.iloc[-1]), "g_mean": float(g.iloc[-k:].mean()),
                   "g_std": float(g.iloc[-k:].std(ddof=0)), "kase": float(kase.iloc[0])})
    res = _step(definition, g, data, x, seed)
    table, model, mu, sg = _logit(data)
    contrib = (pd.Series(model.coef_[0] * ((x[FEATURES] - mu) / sg).to_numpy(float),
                         index=[FEATURE_LABELS[f] for f in FEATURES]) if model is not None else pd.Series(dtype=float))
    probs = pd.Series({name: v[1] for name, v in res.items()})
    return {"period": str(period), "available": True, "x": x, "probs": probs,
            "growth": pd.Series({name: v[0] for name, v in res.items() if name in REG_MODELS}),
            "contributions": contrib, "value": int(probs[ARIMAX] >= 0.5)}


def _panel(gdp_wide: pd.DataFrame, country: str, first_test: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Panel mode: classifiers of positive growth on the [A1] predictors, retrained every year on
    all countries and earlier years; returns Kazakhstan probabilities, all-country metrics and next-year probabilities."""
    feats = macro.build_features(macro.to_long(gdp_wide)).dropna(subset=macro.FEATURES)
    feats["y"] = (feats["growth"] > 0).astype(int)
    models = {ML_MODELS[0]: lambda: LogisticRegression(C=1.0, max_iter=1000),
              ML_MODELS[2]: lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                              random_state=seed)}
    preds = []
    for year in range(first_test, int(feats["year"].max()) + 1):
        train, test = feats[feats["year"] < year], feats[feats["year"] == year]
        if test.empty:
            continue
        mu, sg = train[macro.FEATURES].mean(), train[macro.FEATURES].std(ddof=0).replace(0, 1.0)
        block = test[["country", "year", "y", "growth"]].copy()
        block[BASE] = float(train["y"].mean())
        for name, make in models.items():
            m = make().fit(((train[macro.FEATURES] - mu) / sg).to_numpy(), train["y"].to_numpy())
            block[name] = m.predict_proba(((test[macro.FEATURES] - mu) / sg).to_numpy())[:, 1]
        preds.append(block)
    allp = pd.concat(preds, ignore_index=True)
    names = [BASE, *models]
    nxt = macro.next_year_features(macro.to_long(gdp_wide), country)
    mu, sg = feats[macro.FEATURES].mean(), feats[macro.FEATURES].std(ddof=0).replace(0, 1.0)
    future = pd.Series({name: float(make().fit(((feats[macro.FEATURES] - mu) / sg).to_numpy(), feats["y"].to_numpy())
                                    .predict_proba(((nxt[macro.FEATURES] - mu) / sg).to_numpy())[0, 1])
                        for name, make in models.items()})
    world = pd.DataFrame({n: {"Точность": float(((allp[n] >= 0.5).astype(int) == allp["y"]).mean()),
                              "Оценка Брайера": float(((allp[n] - allp["y"]) ** 2).mean()),
                              "ROC-AUC": _auc(allp["y"].to_numpy(), allp[n].to_numpy())} for n in names}).T
    kz = allp[allp["country"] == country].set_index("year")
    kz.index = pd.PeriodIndex(kz.index.astype(int), freq="Y")
    return kz[names], world, future


def run(definition: str = "sign", freq: str | None = None, window_months: int = WINDOW_MONTHS,
        gdp_wide: pd.DataFrame | None = None, index: pd.Series | None = None,
        growth: pd.Series | None = None, country: str = config.TARGET_COUNTRY,
        seed: int = config.RANDOM_SEED) -> Attractiveness:
    if definition not in DEFINITIONS:
        raise ValueError(f"unknown definition: {definition}")
    freq = "annual" if definition == "panel" else (freq or DEFINITIONS[definition]["freq"])
    if freq not in FREQS:
        raise ValueError(f"unknown frequency: {freq}")
    index = index if index is not None else loaders.load_index_kase_long()
    g = growth if growth is not None else growth_series(freq, gdp_wide, country)
    base_def = "sign" if definition == "panel" else definition
    data = build_dataset(base_def, freq, g, index, window_months)
    probs, means = walk_forward(base_def, data, g, MIN_TRAIN[freq], seed)
    notes, panel = [], None
    if definition == "panel":
        wide = gdp_wide if gdp_wide is not None else loaders.load_gdp().frame
        kz, panel, future = _panel(wide, country, int(probs.index[0].year), seed)
        kz = kz.reindex(probs.index)
        probs = pd.concat([probs[[c for c in REG_MODELS]], kz], axis=1)
        notes.append("В режиме панели машинное обучение обучается на всех странах по признакам [A1]; "
                     "индекса KASE среди признаков нет, он участвует только в модели AR(1) + KASE.")
    y, gt = data["y"].reindex(probs.index), data["g"].reindex(probs.index)
    table = metrics(y, gt, probs, means)
    share = float(y.mean())
    if min(share, 1 - share) * len(y) < 3:
        notes.append(f"В проверочном периоде класс 0 встречается {int(round((1 - share) * len(y)))} раз из {len(y)}: "
                     "точность и ROC-AUC на такой выборке не различают модели.")
    signal = _signal(base_def, data, g, index, freq, window_months, seed)
    if definition == "panel" and signal["available"]:
        signal["probs"] = pd.concat([signal["probs"].drop(list(ML_MODELS)), future])
    return Attractiveness(definition, freq, window_months, data, probs, means, table, _coefficients(data),
                          _logit(data)[0], _lag_profile(g, index, window_months), _diagnostics(data, g, means),
                          signal, panel, notes)


# --------------------------------------------------------------------------
# Country clusters by the mean and the variability of growth
# --------------------------------------------------------------------------
def country_clusters(gdp_wide: pd.DataFrame | None = None, k: int = 2, start: int = CLUSTER_START,
                     min_years: int = 20, country: str = config.TARGET_COUNTRY,
                     seed: int = config.RANDOM_SEED) -> dict:
    """K-means on the standardised mean and standard deviation of annual growth since ``start``.
    Returns the country table, cluster profiles, the silhouette by number of clusters and the
    cluster of ``country``. In [A1] the same clustering uses K = 4 and adds skewness."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    wide = gdp_wide if gdp_wide is not None else loaders.load_gdp().frame
    x = wide[[c for c in wide.columns if int(c) >= start]]
    x = x[x.notna().sum(axis=1) >= min_years]
    table = pd.DataFrame({"mean": x.mean(axis=1), "std": x.std(axis=1, ddof=1)})
    z = ((table - table.mean()) / table.std(ddof=0)).to_numpy()
    sil = {}
    for kk in range(2, 7):
        labels = KMeans(kk, n_init=20, random_state=seed).fit_predict(z)
        sil[kk] = float(silhouette_score(z, labels))
    labels = KMeans(k, n_init=20, random_state=seed).fit_predict(z)
    order = pd.Series(table["mean"].groupby(labels).mean()).sort_values().index      # clusters by mean growth
    table["cluster"] = pd.Series(labels, index=table.index).map({c: i + 1 for i, c in enumerate(order)})
    profiles = table.groupby("cluster").agg(countries=("mean", "size"), mean=("mean", "mean"), std=("std", "mean"))
    return {"table": table, "profiles": profiles, "silhouette": pd.Series(sil, name="silhouette"),
            "k": k, "start": start, "country_cluster": int(table.loc[country, "cluster"]) if country in table.index else None}
