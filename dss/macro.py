"""Module I - macroeconomic forecasting.  [A1]

* Feature engineering exactly as in [A1] Appendix A.1 (past information only).
* Recession classification: Logistic Regression, Random Forest, Gradient
  Boosting, SVM (RBF); regression of GDP growth: Linear, Ridge, RF, GB.
* Chronological split: train <= 2014, test 2015-2024.
* Country clustering: static K-means on (mean, std, skew) + DTW K-medoids.
* Kazakhstan: ARIMA(p,1,q) chosen by AIC, a "Prophet-style" trend+cycle model,
  an "LSTM-style" 5-year sliding-window network; consensus with a predefined
  plausibility rule; scenarios (Table 9) and sensitivity (Table 10).
* Output for the integration module: recession probability, growth forecast,
  business-cycle phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .clustering import dtw_matrix, kmedoids_pam

FEATURES = ["lag1", "lag2", "lag3", "rolling_mean3", "rolling_mean5",
            "rolling_std3", "cumulative_mean", "year_norm"]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.reset_index().melt(id_vars=wide.index.name or "index", var_name="year", value_name="growth")
    long.columns = ["country", "year", "growth"]
    long["year"] = long["year"].astype(int)
    return long.dropna().sort_values(["country", "year"]).reset_index(drop=True)


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    """[A1] A.1 - every predictor uses only values observed before year t."""
    df = long.sort_values(["country", "year"]).copy()
    g = df.groupby("country")["growth"]
    for k in (1, 2, 3):
        df[f"lag{k}"] = g.shift(k)
    past = g.shift(1)
    pg = past.groupby(df["country"])
    df["rolling_mean3"] = pg.transform(lambda s: s.rolling(3).mean())
    df["rolling_mean5"] = pg.transform(lambda s: s.rolling(5).mean())
    df["rolling_std3"] = pg.transform(lambda s: s.rolling(3).std(ddof=0))
    df["cumulative_mean"] = pg.transform(lambda s: s.expanding().mean())
    ymin, ymax = df["year"].min(), df["year"].max()
    df["year_norm"] = (df["year"] - ymin) / max(ymax - ymin, 1)
    df["recession"] = (df["growth"] < 0).astype(int)
    return df


def next_year_features(long: pd.DataFrame, country: str) -> pd.DataFrame:
    """Feature row for the first unobserved year of ``country`` (nowcast input)."""
    s = long[long.country == country].sort_values("year")
    y, yr = s["growth"].to_numpy(), int(s["year"].max()) + 1
    ymin, ymax = long["year"].min(), long["year"].max() + 1
    row = {"lag1": y[-1], "lag2": y[-2], "lag3": y[-3],
           "rolling_mean3": y[-3:].mean(), "rolling_mean5": y[-5:].mean(),
           "rolling_std3": y[-3:].std(ddof=0), "cumulative_mean": y.mean(),
           "year_norm": (yr - ymin) / (ymax - ymin)}
    return pd.DataFrame([row], index=[yr])[FEATURES]


def mcnemar(y, a, b) -> tuple[float, float]:
    """McNemar chi2 (continuity-corrected) for two classifiers' predictions."""
    ca, cb = (a == y), (b == y)
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(chi2), float(stats.chi2.sf(chi2, 1))


# --------------------------------------------------------------------------
# Univariate forecasters for the target country
# --------------------------------------------------------------------------
@dataclass
class ArimaFit:
    order: tuple
    aic: float
    params: np.ndarray
    sigma2: float
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _arima_p1q(y: np.ndarray, p: int, q: int, h: int) -> ArimaFit:
    """ARIMA(p,1,q) by conditional least squares (q in {0,1}), no drift."""
    dy = np.diff(y)
    n = len(dy)

    def resid(theta):
        phi, ma = theta[:p], (theta[p] if q else 0.0)
        e = np.zeros(n)
        for t in range(n):
            ar = sum(phi[i] * dy[t - 1 - i] for i in range(p) if t - 1 - i >= 0)
            e[t] = dy[t] - ar - (ma * e[t - 1] if t > 0 else 0.0)
        return e[p:]

    from scipy.optimize import least_squares
    theta0 = np.zeros(p + q)
    sol = least_squares(resid, theta0, bounds=(-0.99, 0.99))
    e = resid(sol.x)
    m = len(e)
    sigma2 = float(e @ e / m)
    aic = m * np.log(sigma2) + 2 * (p + q + 1)
    phi, ma = sol.x[:p], (sol.x[p] if q else 0.0)
    # recursive forecast of differences, then integrate
    hist, last_e = list(dy), (e[-1] if len(e) else 0.0)
    fc_d = []
    for i in range(h):
        val = sum(phi[j] * hist[-1 - j] for j in range(p)) + (ma * last_e if i == 0 else 0.0)
        fc_d.append(val)
        hist.append(val)
    fc = y[-1] + np.cumsum(fc_d)
    # psi-weights of the integrated process for interval widths
    psi = np.zeros(h)
    psi[0] = 1.0
    for i in range(1, h):
        psi[i] = sum(phi[j] * psi[i - 1 - j] for j in range(min(p, i)))
        if i == 1:
            psi[i] += ma
    cum = np.cumsum(psi)
    se = np.sqrt(sigma2 * np.cumsum(cum ** 2))
    return ArimaFit((p, 1, q), float(aic), sol.x, sigma2, fc, fc - 1.96 * se, fc + 1.96 * se)


def arima_select(y: np.ndarray, h: int,
                 grid=((1, 1, 0), (2, 1, 0), (3, 1, 0), (2, 1, 1))) -> tuple[ArimaFit, pd.DataFrame]:
    """AIC-based order selection over the grid of [A1] Table 4."""
    try:  # use statsmodels' exact MLE when it is installed
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        fits = []
        for order in grid:
            res = ARIMA(y, order=order).fit()
            f = res.get_forecast(h)
            ci = f.conf_int(alpha=0.05)
            fits.append(ArimaFit(order, float(res.aic), res.params, float(res.params[-1]),
                                 f.predicted_mean, ci[:, 0], ci[:, 1]))
    except ImportError:
        fits = [_arima_p1q(y, p, q, h) for p, _, q in grid]
    table = pd.DataFrame({"order": [f"ARIMA{f.order}" for f in fits], "AIC": [f.aic for f in fits]})
    best = min(fits, key=lambda f: f.aic)
    return best, table


def prophet_style(years: np.ndarray, y: np.ndarray, h: int, changepoints: int = 3) -> np.ndarray:
    """Piecewise-linear trend with changepoints + one low-frequency harmonic."""
    t = (years - years[0]) / (years[-1] - years[0])
    cps = np.linspace(0, 1, changepoints + 2)[1:-1]

    def design(tt):
        cols = [np.ones_like(tt), tt] + [np.maximum(tt - c, 0) for c in cps]
        cols += [np.sin(2 * np.pi * tt * 1.5), np.cos(2 * np.pi * tt * 1.5)]
        return np.column_stack(cols)

    beta, *_ = np.linalg.lstsq(design(t), y, rcond=None)
    tf = (np.arange(years[-1] + 1, years[-1] + 1 + h) - years[0]) / (years[-1] - years[0])
    return design(tf) @ beta


def lstm_style(y: np.ndarray, h: int, window: int = 5, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Simplified sequence learner: 5-period sliding window, one hidden layer of
    32 units ([A1] A.5). Implemented with scikit-learn's MLP to stay light."""
    X = np.array([y[i:i + window] for i in range(len(y) - window)])
    t = y[window:]
    mu, sd = y.mean(), y.std()
    net = MLPRegressor(hidden_layer_sizes=(32,), max_iter=3000, random_state=seed, alpha=1e-2)
    net.fit((X - mu) / sd, (t - mu) / sd)
    hist, out = list(y), []
    for _ in range(h):
        x = (np.array(hist[-window:]) - mu) / sd
        val = float(net.predict(x[None, :])[0] * sd + mu)
        out.append(val)
        hist.append(val)
    return np.array(out)


def plausible(fc: np.ndarray, history: np.ndarray, recent: int = 15) -> bool:
    """Predefined inclusion rule for the consensus ([A1] Sec. 5.3.1): every
    forecast must lie inside the range observed over the last ``recent`` years,
    and the path must not turn negative when recent average growth is positive."""
    h = history[-recent:]
    inside = bool(np.all((fc >= h.min()) & (fc <= h.max())))
    sign_ok = not (h.mean() > 0 and np.any(fc < 0))
    return inside and sign_ok


def interp_effect(points, x: float) -> float:
    xs, ys = zip(*sorted(points))
    return float(np.interp(x, xs, ys))


def sensitivity(values: dict) -> dict:
    """Effect (p.p.) on next-year growth of each risk factor ([A1] Table 10)."""
    out = {}
    for key, spec in config.SENSITIVITY.items():
        x = values.get(key, spec["baseline"])
        out[key] = interp_effect(spec["points"], x)
    out["total"] = sum(out.values())
    return out


# --------------------------------------------------------------------------
# Module result
# --------------------------------------------------------------------------
@dataclass
class MacroResult:
    n_obs: int
    n_countries: int
    test_n: int
    classification: pd.DataFrame
    regression: pd.DataFrame
    robustness: pd.DataFrame
    importance: pd.Series
    mcnemar: tuple
    clusters_static: pd.DataFrame
    clusters_dtw: pd.DataFrame
    dtw_profiles: pd.DataFrame
    country_history: pd.Series
    arima_table: pd.DataFrame
    forecasts: pd.DataFrame
    consensus_models: list
    recession_prob: float
    nowcast_gb: float
    phase: str
    kz_static_cluster: str = ""
    kz_dtw_cluster: str = ""
    extra: dict = field(default_factory=dict)

    def scenario_path(self, scenario: str) -> pd.Series:
        base = self.forecasts["ARIMA"]
        off = config.SCENARIOS[scenario]["offsets"][:len(base)]
        return base + np.array(off)


def _cluster_names(stats_df: pd.DataFrame, labels: np.ndarray) -> dict:
    g = stats_df.groupby(labels)[["mean", "std"]].mean()
    names = {}
    crisis = g["std"].idxmax()
    names[crisis] = "Кризисный"
    rest = g.drop(crisis)
    high = rest["mean"].idxmax()
    names[high] = "Высокий рост"
    rest = rest.drop(high)
    if len(rest):
        low = rest["mean"].idxmin()
        names[low] = "Низкий рост / стагнация"
        for c in rest.index.drop(low):
            names[c] = "Стабильный рост"
    return names


def recession_classifiers(seed: int = config.RANDOM_SEED) -> dict:
    """Recession classifiers of [A1] Sec. 4 (unfitted)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(100, max_depth=6, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.10, max_depth=4, random_state=seed),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
    }


def run(gdp_wide: pd.DataFrame, country: str = config.TARGET_COUNTRY,
        train_end: int = config.MACRO_TRAIN_END_YEAR, horizon: int = config.MACRO_FORECAST_HORIZON,
        dtw: bool = True) -> MacroResult:
    seed = config.RANDOM_SEED
    long = to_long(gdp_wide)
    feats = build_features(long).dropna(subset=FEATURES)
    train, test = feats[feats.year <= train_end], feats[feats.year > train_end]
    scaler = StandardScaler().fit(train[FEATURES])
    Xtr, Xte = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])

    # ---- recession classification
    clfs = recession_classifiers(seed)
    rows, preds = [], {}
    for name, m in clfs.items():
        m.fit(Xtr, train.recession)
        p = m.predict(Xte)
        pr = m.predict_proba(Xte)[:, 1]
        preds[name] = p
        rows.append({"model": name, "accuracy": accuracy_score(test.recession, p),
                     "roc_auc": roc_auc_score(test.recession, pr),
                     "recall_recession": recall_score(test.recession, p, zero_division=0),
                     "f1_recession": f1_score(test.recession, p, zero_division=0)})
    cls_table = pd.DataFrame(rows).set_index("model")
    mc = mcnemar(test.recession.to_numpy(), preds["Gradient Boosting"], preds["Random Forest"])

    # ---- growth regression
    regs = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(100, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed),
    }
    rrows = []
    for name, m in regs.items():
        m.fit(Xtr, train.growth)
        p = m.predict(Xte)
        rrows.append({"model": name, "MAE": mean_absolute_error(test.growth, p),
                      "RMSE": float(np.sqrt(mean_squared_error(test.growth, p))),
                      "R2": r2_score(test.growth, p)})
    reg_table = pd.DataFrame(rrows).set_index("model")

    # ---- robustness: without rolling features, alternative split  ([A1] Table 11)
    def _gb_rmse(cols, tr, te):
        sc = StandardScaler().fit(tr[cols])
        m = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed)
        m.fit(sc.transform(tr[cols]), tr.growth)
        p = m.predict(sc.transform(te[cols]))
        return float(np.sqrt(mean_squared_error(te.growth, p))), r2_score(te.growth, p)

    no_roll = [f for f in FEATURES if not f.startswith(("rolling", "cumulative"))]
    rob = {"Базовая (все признаки)": _gb_rmse(FEATURES, train, test),
           "Без скользящих признаков": _gb_rmse(no_roll, train, test),
           "Альтернативный сплит (2012)": _gb_rmse(FEATURES, feats[feats.year <= 2012], feats[feats.year > 2012])}
    crisis = feats[feats.year.isin([2009, 2020])]
    rob["Кризисные годы (2009, 2020)"] = _gb_rmse(FEATURES, feats[~feats.year.isin([2009, 2020]) & (feats.year <= 2019)], crisis)
    robustness = pd.DataFrame(rob, index=["RMSE", "R2"]).T

    importance = pd.Series(clfs["Random Forest"].feature_importances_, index=FEATURES).sort_values(ascending=False)

    # ---- target-country nowcast with models re-fitted on all data
    allX = StandardScaler().fit(feats[FEATURES])
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.10, max_depth=4, random_state=seed)
    gbc.fit(allX.transform(feats[FEATURES]), feats.recession)
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed)
    gbr.fit(allX.transform(feats[FEATURES]), feats.growth)
    xnext = allX.transform(next_year_features(long, country))
    rec_prob = float(gbc.predict_proba(xnext)[0, 1])
    nowcast = float(gbr.predict(xnext)[0])

    # ---- country clustering: static K-means on (mean, std, skew)
    st = long.groupby("country")["growth"].agg(["mean", "std", lambda s: stats.skew(s)]).dropna()
    st.columns = ["mean", "std", "skew"]
    Xs = StandardScaler().fit_transform(st)
    km = KMeans(4, n_init=20, random_state=seed).fit(Xs)
    names = _cluster_names(st, km.labels_)
    pc = PCA(2, random_state=seed).fit_transform(Xs)
    static = st.assign(cluster=[names[c] for c in km.labels_], pc1=pc[:, 0], pc2=pc[:, 1])

    # ---- DTW K-medoids on z-normalised trajectories with complete data
    ser = gdp_wide.loc[:, gdp_wide.columns >= 1993].dropna()
    dtw_df, profiles = pd.DataFrame(), pd.DataFrame()
    if dtw and len(ser) >= 8:
        Z = ser.sub(ser.mean(axis=1), axis=0).div(ser.std(axis=1).replace(0, 1), axis=0)
        D = dtw_matrix(Z.to_numpy(), window=3)
        lab, med, _ = kmedoids_pam(D, 4)
        dstats = pd.DataFrame({"mean": ser.mean(axis=1), "std": ser.std(axis=1)})
        dn = _cluster_names(dstats, lab)
        dtw_df = dstats.assign(cluster=[dn[c] for c in lab])
        profiles = pd.DataFrame({dn[lab[i]]: Z.iloc[i].to_numpy() for i in med}, index=ser.columns)

    # ---- target-country forecasts
    hist = long[long.country == country].set_index("year")["growth"]
    yv, yrs = hist.to_numpy(), hist.index.to_numpy()
    best, aic_table = arima_select(yv, horizon)
    fyears = np.arange(yrs[-1] + 1, yrs[-1] + 1 + horizon)
    fc = pd.DataFrame({"ARIMA": best.forecast, "ARIMA_lo": best.lower, "ARIMA_hi": best.upper,
                       "Prophet-style": prophet_style(yrs, yv, horizon),
                       "LSTM-style": lstm_style(yv, horizon)}, index=fyears)
    candidates = ["ARIMA", "Prophet-style", "LSTM-style"]
    included = [c for c in candidates if plausible(fc[c].to_numpy(), yv)]
    if not included:
        included = ["ARIMA"]
    fc["Консенсус"] = fc[included].mean(axis=1)
    imf = pd.Series(config.IMF_WEO_KZ_BENCHMARK).reindex(fyears)
    if country == "Kazakhstan" and imf.notna().any():
        fc["IMF WEO"] = imf
    aic_table["selected"] = aic_table["order"] == f"ARIMA{best.order}"

    # ---- business-cycle phase
    trend = yv[-5:].mean()
    nxt = fc["ARIMA"].iloc[0]
    if rec_prob >= 0.5 or nxt < 0:
        phase = "Спад"
    elif yv[-1] < trend and nxt > yv[-1]:
        phase = "Восстановление"
    elif nxt >= trend:
        phase = "Расширение"
    else:
        phase = "Замедление"

    return MacroResult(
        n_obs=len(feats), n_countries=long.country.nunique(), test_n=len(test),
        classification=cls_table, regression=reg_table, robustness=robustness,
        importance=importance, mcnemar=mc, clusters_static=static, clusters_dtw=dtw_df,
        dtw_profiles=profiles, country_history=hist, arima_table=aic_table, forecasts=fc,
        consensus_models=included, recession_prob=rec_prob, nowcast_gb=nowcast, phase=phase,
        kz_static_cluster=static.loc[country, "cluster"] if country in static.index else "—",
        kz_dtw_cluster=dtw_df.loc[country, "cluster"] if country in dtw_df.index else "—",
        extra={"arima_order": best.order, "train_end": train_end},
    )
