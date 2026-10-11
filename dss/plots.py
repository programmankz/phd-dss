"""Charts for the web UI and the HTML report (matplotlib -> base64 PNG).

Colours follow the defence presentation: Module I blue, II teal, III purple,
IV orange, navy for the integration layer.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .numfmt import num  # noqa: E402

NAVY, TEXT, MUTED, GRID = "#14213D", "#1F2937", "#5B6576", "#D9DEE6"
M1, M2, M3, M4, GOLD = "#1F5FAD", "#0F8B8D", "#7A4EA3", "#C2700E", "#E0A526"
SERIES = [M1, M4, M2, M3, GOLD, "#8A93A3", "#B03A48", "#3E7C59"]

from matplotlib import font_manager as _fm  # noqa: E402

_AVAILABLE = {f.name for f in _fm.fontManager.ttflist}
_FONTS = [f for f in ("Calibri", "Carlito", "Segoe UI", "Arial", "DejaVu Sans") if f in _AVAILABLE] or ["DejaVu Sans"]

plt.rcParams.update({
    "font.family": _FONTS, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
})


def to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- Module I
def kz_forecast(macro, scenario: str = "base", country: str = "Казахстан") -> str:
    from .config import SCENARIOS
    h, fc = macro.country_history, macro.forecasts
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(h.index, h.values, color=TEXT, lw=1.6, label="Факт")
    x = fc.index
    ax.fill_between(x, fc["ARIMA_lo"], fc["ARIMA_hi"], color=M1, alpha=0.10, lw=0, label="ARIMA 95% ДИ")
    ax.plot(x, fc["ARIMA"], color=M1, lw=2.2, marker="o", ms=4, label=f"ARIMA{macro.extra['arima_order']}")
    ax.plot(x, fc["LSTM-style"], color=M4, lw=1.4, ls="--", label="LSTM-style")
    ax.plot(x, fc["Prophet-style"], color=MUTED, lw=1.2, ls=":", label="Prophet-style")
    if "IMF WEO" in fc:
        ax.plot(x, fc["IMF WEO"], color=GOLD, lw=1.4, marker="s", ms=3, label="IMF WEO (окт. 2024)")
    if scenario != "base":
        ax.plot(x, macro.scenario_path(scenario), color="#B03A48", lw=2, label=SCENARIOS[scenario]["label"])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"{country}: рост ВВП, % — факт и прогноз")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    return to_b64(fig)


def importance(series: pd.Series) -> str:
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.barh(s.index, s.values, color=M1)
    for i, v in enumerate(s.values):
        ax.text(v + 0.004, i, f"{num(v, 2)}", va="center", fontsize=8, color=TEXT)
    ax.set_title("Важность признаков (Random Forest)")
    ax.grid(axis="y", visible=False)
    return to_b64(fig)


def country_clusters(static: pd.DataFrame, highlight=("Kazakhstan",)) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    cols = {"Стабильный рост": M1, "Высокий рост": M2, "Низкий рост / стагнация": GOLD, "Кризисный": "#B03A48"}
    for name, sub in static.groupby("cluster"):
        ax.scatter(sub.pc1, sub.pc2, s=16, color=cols.get(name, MUTED), alpha=0.75, label=name)
    for h in highlight:
        if h in static.index:
            r = static.loc[h]
            ax.scatter(r.pc1, r.pc2, s=90, facecolor="none", edgecolor=NAVY, lw=2)
            ax.annotate(h, (r.pc1, r.pc2), xytext=(6, 6), textcoords="offset points", fontsize=9, color=NAVY, weight="bold")
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    ax.set_title("K-Means (K = 4) по (среднее, σ, асимметрия): PCA")
    ax.legend(fontsize=8, loc="best")
    return to_b64(fig)


def dtw_profiles(profiles: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    cols = {"Стабильный рост": M1, "Высокий рост": M2, "Низкий рост / стагнация": GOLD, "Кризисный": "#B03A48"}
    for c in profiles:
        ax.plot(profiles.index, profiles[c], lw=1.8, color=cols.get(c, MUTED), label=c)
    ax.set_title("DTW-кластеры: медоидные траектории (z-оценки)")
    ax.legend(fontsize=8)
    return to_b64(fig)


# ---------------------------------------------------------------- Module II
def fx_rates(rates: pd.DataFrame, shocks: pd.DataFrame | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for c, col in zip(["USD", "EUR", "CNY"], [M1, M4, M2]):
        if c in rates:
            ax.plot(rates.index, rates[c], lw=1.2, color=col, label=f"{c}/KZT")
    ax2 = ax.twinx()
    if "RUB" in rates:
        ax2.plot(rates.index, rates["RUB"], lw=1.0, color=M3, alpha=0.8, label="RUB/KZT (пр. ось)")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    if shocks is not None and len(shocks):
        for d in shocks["date"]:
            ax.axvline(pd.Timestamp(d), color="#B03A48", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("Официальные курсы НБ РК, тенге за единицу (пунктир — выявленные шоки)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=4, fontsize=8, loc="upper left")
    return to_b64(fig)


def heatmap(corr: pd.DataFrame, title: str, cmap="Blues") -> str:
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    im = ax.imshow(corr.values, cmap=cmap, vmin=min(0, corr.values.min()), vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=0 if len(corr) <= 8 else 90)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr) if len(corr) <= 8 else 0):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{num(v, 2)}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.75 else TEXT)
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return to_b64(fig)


def r2_bars(reg: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    cols = [M2 if v >= 0.5 else "#B03A48" for v in reg["R2"]]
    ax.bar(reg.index, reg["R2"], color=cols, width=0.6)
    for i, v in enumerate(reg["R2"]):
        ax.text(i, v + 0.02, f"{num(v, 2)}", ha="center", fontsize=9, color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_title("Линейный тренд: R² по валютам")
    ax.grid(axis="x", visible=False)
    return to_b64(fig)


def fx_risk(comp: pd.DataFrame, total: float) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    s = comp["risk"].sort_values()
    cols = ["#B03A48" if v >= 0.66 else GOLD if v >= 0.4 else M2 for v in s.values]
    ax.barh([f"{c}/KZT" for c in s.index], s.values, color=cols)
    ax.axvline(total, color=NAVY, lw=2, label=f"Индекс = {num(total, 2)}")
    ax.set_xlim(0, 1)
    ax.set_title("Компоненты индекса валютного риска")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    return to_b64(fig)


# ---------------------------------------------------------------- Module III
def asset_clusters(cl) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    lab = cl.labels["K-medoids"]
    for k in sorted(lab.unique()):
        idx = lab[lab == k].index
        ax.scatter(cl.pca.loc[idx, "pc1"], cl.pca.loc[idx, "pc2"], s=70, color=SERIES[k % len(SERIES)],
                   alpha=0.85, label=cl.names.get(int(k), f"К{k + 1}"))
    for t in cl.pca.index:
        ax.annotate(t, (cl.pca.loc[t, "pc1"], cl.pca.loc[t, "pc2"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    for m in cl.medoids:
        ax.scatter(cl.pca.loc[m, "pc1"], cl.pca.loc[m, "pc2"], s=260, facecolor="none", edgecolor=NAVY, lw=1.8, zorder=5)
    ax.set_title("K-medoids: PCA-проекция признаков (○ — медоиды)")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    return to_b64(fig)


def memberships(U: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.0, 0.32 * len(U) + 1.0))
    ax.imshow(U.values, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(U)), U.index)
    ax.set_xticks(range(U.shape[1]), U.columns)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            v = U.values[i, j]
            ax.text(j, i, f"{num(v, 2)}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else TEXT)
    ax.grid(False)
    ax.set_title("Fuzzy C-means: матрица U")
    return to_b64(fig)


def k_selection(sel: dict) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(sel["k"], sel["inertia"], marker="o", color=M3, label="Стоимость (локоть)")
    ax2 = ax.twinx()
    ax2.plot(sel["k"], sel["silhouette"], marker="s", ls="--", color=M4, label="Силуэт")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.axvline(sel["best_k"], color=MUTED, ls=":")
    ax.set_xticks(sel["k"])
    ax.set_xlabel("k")
    ax.set_title("Выбор числа кластеров")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    return to_b64(fig)


# ---------------------------------------------------------------- Module IV
def rank_signals(latest: pd.DataFrame) -> str:
    s = latest["rank_signal"].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(s) + 1.0))
    ax.barh(s.index, s.values, color=M4)
    ax.set_xlim(0, 1.05)
    ax.set_title("Ранговые сигналы sᵢ = Rank(r̂ᵢ)/N (последняя дата)")
    ax.grid(axis="y", visible=False)
    return to_b64(fig)


def model_ic(metrics: pd.DataFrame) -> str:
    m = metrics["IC"].dropna()
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(m) + 1.0))
    ax.barh(m.index, m.values, color=[M4 if v > 0 else MUTED for v in m.values])
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.set_title("Информационный коэффициент (IC) вне выборки")
    ax.grid(axis="y", visible=False)
    return to_b64(fig)


def equity_curves(curves: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = {"Signal Weighted": M4, "Max Signal-to-Risk": M1, "Min Variance": M2, "Equal Weighted": GOLD}
    for c in curves:
        ax.plot(curves.index, curves[c], lw=2 if c in ("Signal Weighted", "Max Signal-to-Risk") else 1.2,
                color=cols.get(c, MUTED), ls="-" if c in cols else "--", label=c)
    ax.set_title("Бэктест вне выборки: стоимость портфеля (база = 100), с учётом издержек")
    ax.legend(ncol=3, fontsize=8)
    return to_b64(fig)


# ---------------------------------------------------------------- Decision
def allocation(alloc: pd.DataFrame) -> str:
    a = alloc[alloc["weight"] > 0.001].sort_values("weight")
    fig, ax = plt.subplots(figsize=(5.6, 0.3 * len(a) + 1.0))
    cols = [M2 if g != "Рисковая часть" else M4 for g in a["group"]]
    labels = [i.replace("Безрисковая часть: ", "") for i in a.index]
    ax.barh(labels, a["weight"] * 100, color=cols)
    for i, v in enumerate(a["weight"] * 100):
        ax.text(v + 0.4, i, f"{num(v, 1)}%", va="center", fontsize=8)
    ax.set_title("Рекомендуемая структура портфеля, %")
    ax.grid(axis="y", visible=False)
    return to_b64(fig)
