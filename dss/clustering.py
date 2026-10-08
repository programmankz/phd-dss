"""Module III - clustering of assets (and of countries for Module I).  [A3], [A1]

Implemented from scratch with numpy so that the DSS has no exotic dependencies:

* K-means (scikit-learn)           - hard clustering, mean centres, baseline
* K-medoids / PAM (BUILD + SWAP)   - hard clustering, real assets as centres,
                                     works on any dissimilarity, e.g. 1 - rho
* Fuzzy C-means (Bezdek)           - soft clustering, membership matrix U
* DTW distance (vectorised over pairs) for time-series clustering of countries

Quality: silhouette, Davies-Bouldin, within-cluster inertia ([A3], [A1]).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED


# --------------------------------------------------------------------------
# Algorithms
# --------------------------------------------------------------------------
def kmedoids_pam(D: np.ndarray, k: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """Partitioning Around Medoids on a precomputed dissimilarity matrix.

    Returns (labels, medoid indices, total cost sum_k sum_{i in C_k} d(x_i, m_k)).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if not 1 <= k <= n:
        raise ValueError("k must be in [1, n]")
    # BUILD: greedy initialisation
    medoids = [int(np.argmin(D.sum(axis=1)))]
    while len(medoids) < k:
        nearest = D[:, medoids].min(axis=1)
        gains = np.maximum(nearest[:, None] - D, 0).sum(axis=0)
        gains[medoids] = -np.inf
        medoids.append(int(np.argmax(gains)))
    medoids = np.array(medoids)

    def cost(meds):
        return D[:, meds].min(axis=1).sum()

    best = cost(medoids)
    for _ in range(max_iter):
        improved = False
        for mi in range(k):
            others = np.setdiff1d(np.arange(n), medoids)
            # vectorised: try every candidate as a replacement of medoid mi
            keep = np.delete(medoids, mi)
            base = D[:, keep].min(axis=1) if len(keep) else np.full(n, np.inf)
            costs = np.minimum(base[:, None], D[:, others]).sum(axis=0)
            j = int(np.argmin(costs))
            if costs[j] < best - 1e-12:
                medoids[mi] = others[j]
                best = costs[j]
                improved = True
        if not improved:
            break
    labels = np.argmin(D[:, medoids], axis=1)
    return labels, medoids, float(best)


def fuzzy_cmeans(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 300,
                 tol: float = 1e-6, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuzzy C-means. Minimises sum_i sum_j u_ij^m ||x_i - c_j||^2, sum_j u_ij = 1.

    Returns (centres c x d, membership U n x c, objective).
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    U = rng.random((X.shape[0], c))
    U /= U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        Um = U ** m
        centres = Um.T @ X / Um.sum(axis=0)[:, None]
        d = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
        d = np.fmax(d, 1e-12)
        inv = d ** (-2.0 / (m - 1.0))
        U_new = inv / inv.sum(axis=1, keepdims=True)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    d = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
    return centres, U, float(((U ** m) * d ** 2).sum())


def dtw_matrix(series: np.ndarray, window: int | None = None) -> np.ndarray:
    """Pairwise DTW distances between equal-length rows of ``series``.

    Vectorised over all pairs (loops only over the n x n DTW lattice), with an
    optional Sakoe-Chiba band ``window``.
    """
    S = np.asarray(series, dtype=float)
    n, T = S.shape
    ii, jj = np.triu_indices(n, k=1)
    a, b = S[ii], S[jj]
    w = T if window is None else max(window, 1)
    acc = np.full((len(ii), T + 1, T + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for i in range(1, T + 1):
        lo, hi = max(1, i - w), min(T, i + w)
        for j in range(lo, hi + 1):
            cost = (a[:, i - 1] - b[:, j - 1]) ** 2
            acc[:, i, j] = cost + np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
    D = np.zeros((n, n))
    D[ii, jj] = np.sqrt(acc[:, T, T])
    return D + D.T


def correlation_distance(returns: pd.DataFrame) -> np.ndarray:
    """d_ij = 1 - rho_ij of asset returns ([A3]: 'K-medoids divides assets by correlation')."""
    return np.clip(1.0 - returns.corr().to_numpy(), 0.0, 2.0)


def choose_k(X: np.ndarray, k_range=range(2, 7), D: np.ndarray | None = None) -> dict:
    """Elbow (inertia) + silhouette for k selection."""
    out = {"k": [], "inertia": [], "silhouette": []}
    n = len(X) if D is None else len(D)
    for k in k_range:
        if k >= n:
            break
        if D is None:
            km = KMeans(k, n_init=10, random_state=RANDOM_SEED).fit(X)
            lab, inertia = km.labels_, km.inertia_
            sil = silhouette_score(X, lab) if len(set(lab)) > 1 else np.nan
        else:
            lab, _, inertia = kmedoids_pam(D, k)
            sil = silhouette_score(D, lab, metric="precomputed") if len(set(lab)) > 1 else np.nan
        out["k"].append(k)
        out["inertia"].append(float(inertia))
        out["silhouette"].append(float(sil))
    best = out["k"][int(np.nanargmax(out["silhouette"]))] if out["k"] else 2
    out["best_k"] = best
    return out


# --------------------------------------------------------------------------
# Asset clustering for portfolio construction  [A3]
# --------------------------------------------------------------------------
def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def asset_features(prices: pd.DataFrame, fundamentals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Price-behaviour, technical and (if available) valuation features per asset."""
    r = np.log(prices).diff().dropna()
    mkt = r.mean(axis=1)
    feats = pd.DataFrame(index=prices.columns)
    feats["ann_return"] = r.mean() * 252
    feats["ann_vol"] = r.std() * np.sqrt(252)
    feats["sharpe"] = feats["ann_return"] / feats["ann_vol"]
    feats["max_dd"] = [max_drawdown(prices[c]) for c in prices]
    feats["beta"] = r.apply(lambda s: np.cov(s, mkt)[0, 1] / mkt.var())
    feats["mom_6m"] = prices.iloc[-1] / prices.iloc[-min(126, len(prices) - 1)] - 1
    if fundamentals is not None:
        for col in ("PB", "PE"):
            if col in fundamentals.columns:
                feats[col] = fundamentals[col].reindex(prices.columns)
    return feats.dropna(axis=1, how="any")


@dataclass
class ClusterResult:
    features: pd.DataFrame
    k: int
    labels: dict                      # method -> pd.Series of int labels
    medoids: list                     # tickers chosen as medoids
    U: pd.DataFrame                   # fuzzy memberships (assets x clusters)
    metrics: pd.DataFrame             # method x {silhouette, davies_bouldin, inertia}
    pca: pd.DataFrame                 # 2-D projection for plotting
    k_selection: dict
    names: dict = field(default_factory=dict)  # cluster -> readable name


def _name_clusters(feats: pd.DataFrame, labels: pd.Series) -> dict:
    g = feats.groupby(labels)
    names = {}
    for c, sub in g:
        ret, vol = sub["ann_return"].mean(), sub["ann_vol"].mean()
        tag = "доходность выше медианы" if ret >= feats["ann_return"].median() else "доходность ниже медианы"
        risk = "риск выше" if vol >= feats["ann_vol"].median() else "риск ниже"
        names[int(c)] = f"К{int(c) + 1}: {tag}, {risk} ({len(sub)} шт.)"
    return names


def dissimilarity(X: np.ndarray, returns: pd.DataFrame, kind: str = "features") -> np.ndarray:
    """K-medoids dissimilarity: Euclidean on standardised features (default),
    1 - rho of returns, or their 50/50 blend (each scaled by its median).

    With weak correlation structure, pure 1 - rho makes PAM isolate weakly
    correlated stocks as singleton clusters, which is useless for
    diversification limits; the feature space (return, risk, beta, momentum,
    P/B, P/E) gives balanced groups and is the default."""
    Df = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    if kind == "features":
        return Df
    Dc = correlation_distance(returns)
    if kind == "correlation":
        return Dc
    return 0.5 * Df / np.median(Df[Df > 0]) + 0.5 * Dc / np.median(Dc[Dc > 0])


def cluster_assets(prices: pd.DataFrame, fundamentals: pd.DataFrame | None = None,
                   k: int | None = None, m: float = 2.0, distance: str = "features") -> ClusterResult:
    feats = asset_features(prices, fundamentals)
    X = StandardScaler().fit_transform(feats)            # equal influence of features
    r = np.log(prices).diff().dropna()
    D = dissimilarity(X, r, distance)
    kmax = max(2, min(6, len(feats) // 3))
    sel = choose_k(X, k_range=range(2, kmax + 1), D=D)
    k = k or sel["best_k"]

    km = KMeans(k, n_init=20, random_state=RANDOM_SEED).fit(X)
    pam_labels, med_idx, pam_cost = kmedoids_pam(D, k)
    _, U, fcm_obj = fuzzy_cmeans(X, k, m=m)
    fcm_labels = U.argmax(axis=1)

    idx = prices.columns
    labels = {"K-means": pd.Series(km.labels_, idx),
              "K-medoids": pd.Series(pam_labels, idx),
              "Fuzzy C-means": pd.Series(fcm_labels, idx)}

    def _metrics(lab, use_D=False):
        if len(set(lab)) < 2:
            return np.nan, np.nan
        sil = silhouette_score(D, lab, metric="precomputed") if use_D else silhouette_score(X, lab)
        return sil, davies_bouldin_score(X, lab)

    rows = []
    for name, lab, inertia, use_D in (("K-means", km.labels_, km.inertia_, False),
                                      ("K-medoids", pam_labels, pam_cost, True),
                                      ("Fuzzy C-means", fcm_labels, fcm_obj, False)):
        sil, db = _metrics(lab, use_D)
        rows.append({"method": name, "silhouette": sil, "davies_bouldin": db, "inertia": inertia})
    metrics = pd.DataFrame(rows).set_index("method")

    coords = PCA(2, random_state=RANDOM_SEED).fit_transform(X)
    pca = pd.DataFrame(coords, index=idx, columns=["pc1", "pc2"])
    Udf = pd.DataFrame(U, index=idx, columns=[f"К{j + 1}" for j in range(k)])
    return ClusterResult(feats, k, labels, [idx[i] for i in med_idx], Udf, metrics, pca, sel,
                         _name_clusters(feats, labels["K-medoids"]))


def cluster_portfolio_backtest(prices: pd.DataFrame, k: int, rebalance: str = "ME",
                               window: int = 252, m: float = 2.0, tc_bps: float = 10.0,
                               risk_free: float = 0.0) -> pd.DataFrame:
    """Rolling cluster-based portfolios -> the indicator table of [A3] (Table 2).

    At every rebalance date the trailing ``window`` days are re-clustered:
    K-means / K-medoids pick the best trailing-Sharpe asset of every cluster
    (equal weights); Fuzzy C-means spreads each cluster's 1/k budget over its
    members in proportion to u_ij^m * max(Sharpe, 0).
    """
    r = prices.pct_change().dropna()
    dates = r.index
    rebal = set(r.groupby(r.index.to_period(rebalance[0])).tail(1).index)
    strategies = ("K-means", "K-medoids", "Fuzzy C-means", "Равные веса")
    w = {s: pd.Series(1 / r.shape[1], index=r.columns) for s in strategies}
    rets = {s: [] for s in strategies}
    turnover = {s: [] for s in strategies}
    for t_i, t in enumerate(dates):
        for s in strategies:
            day = float((w[s] * r.loc[t]).sum())
            rets[s].append(day)
            grown = w[s] * (1 + r.loc[t])
            w[s] = grown / grown.sum()
        if t in rebal and t_i >= window:
            hist = prices.loc[:t].iloc[-window:]
            hr = np.log(hist).diff().dropna()
            sharpe = hr.mean() / hr.std()
            feats = asset_features(hist)
            X = StandardScaler().fit_transform(feats)
            km = KMeans(k, n_init=10, random_state=RANDOM_SEED).fit(X).labels_
            pam, _, _ = kmedoids_pam(dissimilarity(X, hr), k)
            _, U, _ = fuzzy_cmeans(X, k, m=m)
            new = {}
            for name, lab in (("K-means", km), ("K-medoids", pam)):
                pick = [sharpe[lab == c].idxmax() for c in np.unique(lab)]
                new[name] = pd.Series(0.0, index=r.columns)
                new[name][pick] = 1 / len(pick)
            score = (U ** m) * np.maximum(sharpe.to_numpy(), 0)[:, None] + 1e-9
            fw = (score / score.sum(axis=0)).sum(axis=1) / k
            new["Fuzzy C-means"] = pd.Series(fw, index=r.columns)
            new["Равные веса"] = pd.Series(1 / r.shape[1], index=r.columns)
            for s in strategies:
                to = float((new[s] - w[s]).abs().sum() / 2)
                turnover[s].append(to)
                rets[s][-1] -= to * tc_bps / 1e4
                w[s] = new[s]
    rows = []
    for s in strategies:
        rs = pd.Series(rets[s], index=dates)
        rows.append(performance(rs, risk_free) | {"turnover": float(np.mean(turnover[s])) if turnover[s] else 0.0,
                                                  "strategy": s})
    return pd.DataFrame(rows).set_index("strategy")


def performance(daily: pd.Series, risk_free: float = 0.0) -> dict:
    """Total/annual return, annual volatility, Sharpe, max drawdown."""
    curve = (1 + daily).cumprod()
    years = len(daily) / 252
    ann = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = daily.std() * np.sqrt(252)
    return {"total_return": float(curve.iloc[-1] - 1), "annual_return": float(ann),
            "annual_vol": float(vol), "sharpe": float((ann - risk_free) / vol) if vol > 0 else np.nan,
            "max_drawdown": max_drawdown(curve)}
