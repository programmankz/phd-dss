"""Calculations of article [A1] as they are in the authors' notebook (GDP.ipynb).

The functions repeat the notebook cell by cell; only the data file is replaced
(``world_gdp_data.csv`` of the article is IMF WEO, the repository holds World Bank WDI).

Two experiments on recession classification are in the notebook:

* ``same_year`` (cells 5, 17-20): the class of year t is predicted from features of which the
  rolling statistics and the cumulative mean include the growth of year t itself; the scaler is
  fitted on the whole sample; train < 2015. This gives the 91.6 / 97.9 / 95.1 % of the article's text.
* ``next_year`` (cells 26-32): the class of year t+1, 24 features, train <= 2019, test 2020-2023.
  This gives Logistic Regression 0.896 / ROC-AUC 0.762 of the article's figure. The country
  statistics among the features are computed over the whole sample, including the test years.

Kazakhstan (cells 13-15): AR(3) with an intercept on first differences by OLS, a quadratic trend
with a 5-year "seasonality", and a linear regression on a 5-year window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

SAME_YEAR_FEATURES = ["lag1", "lag2", "lag3", "rolling_mean3", "rolling_std3", "rolling_mean5", "cumulative_mean", "year_norm"]
NEXT_YEAR_FEATURES = ["year", "decade", "covid_year", "crisis_2008", "gdp_lag_1", "gdp_lag_2", "gdp_lag_3", "gdp_lag_5",
                      "gdp_ma_3y", "gdp_ma_5y", "gdp_ma_10y", "gdp_std_3y", "gdp_std_5y", "gdp_yoy_change", "gdp_cum_mean",
                      "gdp_dev_from_mean", "is_recession", "prev_recession", "country_encoded", "country_mean", "country_std",
                      "country_min", "country_max", "country_median"]
# outputs stored in the notebook (IMF WEO data of the article)
NOTEBOOK = {
    "same_year": {"Random Forest": 0.9163, "Gradient Boosting": 0.9793, "SVM (RBF)": 0.9514},
    "next_year_clf": {"Logistic Regression": (0.896, 0.762), "Random Forest": (0.198, 0.668)},
    "next_year_reg": {"Linear Regression": (3.072, 6.179, -0.134), "Ridge Regression": (3.073, 6.180, -0.135),
                      "Random Forest": (11.077, 13.380, -4.320), "Gradient Boosting": (10.640, 13.421, -4.352)},
    "kz": {"AR(3), d=1": [4.34, 4.64, 4.99, 5.23, 5.45], "Prophet-style": [-3.39, -5.34, -7.11, -5.62, -8.26],
           "LSTM-style": [6.84, 6.25, 6.19, 5.53, 6.33]},
    "clusters": {"High Growth": 9, "Low Growth": 119, "Moderate Growth": 1, "Volatile/Crisis": 65},
    "dbscan_outliers": 19, "best_k": 2,
}


def _long(wide: pd.DataFrame) -> pd.DataFrame:
    """All country-years (missing values kept), as ``melt`` in the notebook."""
    long = wide.rename_axis("country_name").reset_index().melt(id_vars="country_name", var_name="year", value_name="gdp_growth")
    long["year"] = long["year"].astype(int)
    return long.sort_values(["country_name", "year"]).reset_index(drop=True)


# ------------------------------------------------------------------ cells 5, 17-20
def same_year_dataset(wide: pd.DataFrame, past_only: bool = False) -> pd.DataFrame:
    """Feature table of cell 5. ``past_only`` shifts the rolling statistics by one year, which
    removes the growth of year t from the predictors (not in the notebook, used as a control)."""
    d = _long(wide)
    g = d.groupby("country_name")["gdp_growth"]
    for k in (1, 2, 3):
        d[f"lag{k}"] = g.shift(k)
    src = g.shift(1).groupby(d["country_name"]) if past_only else g
    d["rolling_mean3"] = src.transform(lambda x: x.rolling(3).mean())
    d["rolling_std3"] = src.transform(lambda x: x.rolling(3).std())
    d["rolling_mean5"] = src.transform(lambda x: x.rolling(5).mean())
    d["cumulative_mean"] = src.transform(lambda x: x.expanding().mean())
    y0, y1 = int(d["year"].min()), int(d["year"].max())
    d["year_norm"] = (d["year"] - y0) / (y1 - y0)
    d = d.dropna(subset=["gdp_growth", "lag1", "lag2"])
    return d.dropna(subset=SAME_YEAR_FEATURES + ["gdp_growth"]).reset_index(drop=True)


def same_year_classification(wide: pd.DataFrame, past_only: bool = False, split_year: int = 2015) -> pd.DataFrame:
    d = same_year_dataset(wide, past_only)
    X = StandardScaler().fit_transform(d[SAME_YEAR_FEATURES].to_numpy())
    y = (d["gdp_growth"] < 0).astype(int).to_numpy()
    tr = (d["year"] < split_year).to_numpy()
    models = {"Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
              "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
              "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42)}
    rows = {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        rows[name] = {"accuracy": accuracy_score(y[~tr], m.predict(X[~tr])),
                      "roc_auc": roc_auc_score(y[~tr], m.predict_proba(X[~tr])[:, 1])}
    out = pd.DataFrame(rows).T
    out.attrs.update(train=int(tr.sum()), test=int((~tr).sum()), crisis_test=float(y[~tr].mean()))
    return out


# ------------------------------------------------------------------ cells 26-32
def next_year_dataset(wide: pd.DataFrame) -> pd.DataFrame:
    d = _long(wide).dropna(subset=["gdp_growth"]).reset_index(drop=True)
    g = d.groupby("country_name")["gdp_growth"]
    for lag in (1, 2, 3, 5):
        d[f"gdp_lag_{lag}"] = g.shift(lag)
    for w in (3, 5, 10):
        d[f"gdp_ma_{w}y"] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
    for w in (3, 5):
        d[f"gdp_std_{w}y"] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=2).std())
    d["gdp_yoy_change"] = g.diff()
    d["gdp_cum_mean"] = g.transform(lambda x: x.shift(1).expanding().mean())
    d["is_recession"] = (d["gdp_growth"] < 0).astype(int)
    d["prev_recession"] = d.groupby("country_name")["is_recession"].shift(1)
    d["decade"] = (d["year"] // 10) * 10
    d["covid_year"] = d["year"].isin([2020, 2021]).astype(int)
    d["crisis_2008"] = d["year"].isin([2008, 2009]).astype(int)
    d["gdp_dev_from_mean"] = g.transform(lambda x: x - x.mean())
    stats = g.agg(country_mean="mean", country_std="std", country_min="min", country_max="max",
                  country_median="median").reset_index()
    d = d.merge(stats, on="country_name", how="left")
    d["country_encoded"] = LabelEncoder().fit_transform(d["country_name"])
    d["target_reg"] = d.groupby("country_name")["gdp_growth"].shift(-1)
    d["target_clf"] = (d["target_reg"] < 0).astype(float)
    return d.dropna(subset=["target_reg"]).reset_index(drop=True)


def next_year_models(wide: pd.DataFrame, train_end: int = 2019) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(regression table, classification table). XGBoost and LightGBM of the notebook are optional
    libraries and are left out; the notebook's own best models are the linear ones."""
    d = next_year_dataset(wide)
    tr, te = d[d["year"] <= train_end], d[d["year"] > train_end]
    imp = SimpleImputer(strategy="median").fit(tr[NEXT_YEAR_FEATURES])
    sc = StandardScaler().fit(imp.transform(tr[NEXT_YEAR_FEATURES]))
    Xtr, Xte = sc.transform(imp.transform(tr[NEXT_YEAR_FEATURES])), sc.transform(imp.transform(te[NEXT_YEAR_FEATURES]))
    regs = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)}
    r = {}
    for name, m in regs.items():
        p = m.fit(Xtr, tr["target_reg"]).predict(Xte)
        r[name] = {"MAE": mean_absolute_error(te["target_reg"], p),
                   "RMSE": mean_squared_error(te["target_reg"], p) ** 0.5, "R2": r2_score(te["target_reg"], p)}
    clfs = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)}
    c = {}
    for name, m in clfs.items():
        m.fit(Xtr, tr["target_clf"])
        pred = m.predict(Xte)
        c[name] = {"accuracy": accuracy_score(te["target_clf"], pred),
                   "roc_auc": roc_auc_score(te["target_clf"], m.predict_proba(Xte)[:, 1]),
                   "predicted_recessions": float(pred.mean())}
    reg, clf = pd.DataFrame(r).T, pd.DataFrame(c).T
    clf.attrs.update(train=len(tr), test=len(te), recession_train=float(tr["target_clf"].mean()),
                     recession_test=float(te["target_clf"].mean()),
                     recession_last_train_year=float(tr.loc[tr["year"] == train_end, "target_clf"].mean()))
    return reg, clf


# ------------------------------------------------------------------ cells 13-15, Kazakhstan
def ar_fit(series: np.ndarray, p: int = 3) -> np.ndarray:
    """AR(p) with an intercept on the first differences, OLS."""
    diff = np.diff(series)
    X = np.array([diff[i - p:i][::-1] for i in range(p, len(diff))])
    return np.linalg.lstsq(np.hstack([np.ones((len(X), 1)), X]), diff[p:], rcond=None)[0]


def ar_forecast(series: np.ndarray, coef: np.ndarray, p: int = 3, steps: int = 5) -> np.ndarray:
    diff = list(np.diff(series))
    for _ in range(steps):
        diff.append(coef[0] + np.dot(coef[1:], np.array(diff[-p:][::-1])))
    return series[-1] + np.cumsum(diff[-steps:])


def prophet_style(series: np.ndarray, steps: int = 5) -> np.ndarray:
    """Quadratic trend plus the mean residual of every fifth year."""
    x = np.arange(len(series))
    coef = np.polyfit(x, series, 2)
    resid = series - np.polyval(coef, x)
    seasonal = np.array([np.mean(resid[i::5]) for i in range(5)])
    return np.polyval(coef, np.arange(len(series), len(series) + steps)) + np.array([seasonal[i % 5] for i in range(steps)])


def lstm_style(series: np.ndarray, steps: int = 5, window: int = 5) -> np.ndarray:
    """Linear regression on the last ``window`` values, iterated."""
    X = np.array([series[i - window:i] for i in range(window, len(series))])
    coef = np.linalg.lstsq(np.hstack([np.ones((len(X), 1)), X]), series[window:], rcond=None)[0]
    hist = list(series)
    for _ in range(steps):
        hist.append(coef[0] + np.dot(coef[1:], np.array(hist[-window:])))
    return np.array(hist[-steps:])


def country_forecasts(wide: pd.DataFrame, country: str = "Kazakhstan", steps: int = 5) -> pd.DataFrame:
    s = wide.loc[country].dropna()
    y = s.to_numpy(dtype=float)
    years = np.arange(int(s.index[-1]) + 1, int(s.index[-1]) + 1 + steps)
    out = pd.DataFrame({"AR(3), d=1": ar_forecast(y, ar_fit(y), steps=steps), "Prophet-style": prophet_style(y, steps),
                        "LSTM-style": lstm_style(y, steps)}, index=pd.Index(years, name="year"))
    out.attrs["ar_coef"] = ar_fit(y)
    return out


# ------------------------------------------------------------------ cells 22-23, clustering
def clusters(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    """(country table with K-means (K=4) and DBSCAN labels, silhouette by K, number of DBSCAN outliers)."""
    last5 = list(wide.columns[-5:])
    st = pd.DataFrame({"mean": wide.mean(axis=1), "std": wide.std(axis=1), "min": wide.min(axis=1), "max": wide.max(axis=1),
                       "recession_pct": (wide < 0).sum(axis=1) / wide.count(axis=1),
                       "recent5": wide[last5].mean(axis=1)}).dropna()
    X = StandardScaler().fit_transform(st.to_numpy())
    sil = pd.Series({k: silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X))
                     for k in range(2, 9)}, name="silhouette")
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    order = st.groupby(labels)["mean"].mean().sort_values(ascending=False).index
    names = dict(zip(order, ["High Growth", "Moderate Growth", "Low Growth", "Volatile/Crisis"]))
    st["cluster"] = [names[c] for c in labels]
    st["dbscan"] = DBSCAN(eps=1.5, min_samples=5).fit_predict(X)
    return st, sil, int((st["dbscan"] == -1).sum())


def load_article_data() -> pd.DataFrame:
    """data/world_gdp_data.csv of the article (IMF WEO, 196 countries, 1980-2024) as country x year."""
    from dss.data.loaders import data_dir
    df = pd.read_csv(data_dir() / "world_gdp_data.csv", encoding="latin1")
    wide = df.set_index("country_name").drop(columns="indicator_name")
    wide.columns = [int(c) for c in wide.columns]
    return wide
