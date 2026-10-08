"""Integration module - aggregation of the four signals into one decision.

Scientific novelty #1 of the dissertation: macro and FX risks are not inputs
to the return forecast but act as the *risk budget* and the *constraints* of
the optimiser:

    Module I   -> phase of the cycle, P(recession), growth path  -> risk budget
    Module II  -> FX risk index, expected KZT depreciation       -> USD hedge
                                                                    of the safe part
    Module III -> clusters / memberships                         -> diversification caps
    Module IV  -> rank signals s, covariance Sigma               -> weights inside
                                                                    the risky part

risky_share = clip(base(profile) x (1 - 0.6 P_rec) x m(scenario) x m(phase)
                   - 0.05 x 1[growth < 0],  min, max)
usd_hedge   = clip(0.15 + 0.60 x FX_risk, 0, 0.80)  of the safe part
w_risky     = argmax w's / sqrt(w'Sigma w)  s.t. sum w = 1, w_i <= w_max,
              sum_{i in C_k} w_i <= cap   (or  sum_i u_ik w_i <= cap)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .portfolio import covariance, feasible_limits, max_signal_to_risk

PHASE_MULT = {"Расширение": 1.05, "Восстановление": 1.00, "Замедление": 0.92, "Спад": 0.75}
SAFE_KZT = "Безрисковая часть: KZT (депозиты, ГЦБ РК)"
SAFE_USD = "Безрисковая часть: USD-хедж (депозиты, UST)"


@dataclass
class Decision:
    profile: config.InvestorProfile
    scenario: str
    risky_share: float
    usd_hedge_share: float
    allocation: pd.DataFrame          # asset, group, cluster, weight, signal
    risky_weights: pd.Series
    exp_vol: float                     # annualised, whole portfolio
    var95_1m: float                    # parametric 1-month VaR, fraction of capital
    cluster_exposure: pd.Series
    steps: list = field(default_factory=list)   # explanation trail, one per module
    warnings: list = field(default_factory=list)


def risk_budget(profile: config.InvestorProfile, rec_prob: float, scenario: str,
                phase: str, growth_next: float) -> tuple[float, list]:
    base = profile.base_risky_share
    m_rec = 1 - 0.6 * rec_prob
    m_sc = config.SCENARIO_RISK_MULT[scenario]
    m_ph = PHASE_MULT.get(phase, 1.0)
    share = base * m_rec * m_sc * m_ph - (0.05 if growth_next < 0 else 0.0)
    share = float(np.clip(share, profile.min_risky_share, profile.max_risky_share))
    trail = [f"базовая доля рисковых активов ({profile.label.lower()} профиль) — {base:.0%}",
             f"× (1 − 0,6·P(рецессии)) = × {m_rec:.2f}",
             f"× сценарий «{config.SCENARIOS[scenario]['label']}» = × {m_sc:.2f}",
             f"× фаза цикла «{phase}» = × {m_ph:.2f}"]
    if growth_next < 0:
        trail.append("− 5 п.п. за отрицательный прогноз роста")
    return share, trail


def decide(macro, fx, clusters, signals, prices: pd.DataFrame, settings: config.Settings,
           sectors: pd.Series | None = None) -> Decision:
    profile = config.PROFILES[settings.profile]
    sc = settings.scenario
    growth = macro.scenario_path(sc)
    g_next = float(growth.iloc[0])
    steps, warns = [], []

    # ---- Module I -> risk budget
    risky, trail = risk_budget(profile, macro.recession_prob, sc, macro.phase, g_next)
    steps.append({"module": "I", "title": "Макропрогноз → бюджет риска",
                  "facts": [f"P(рецессии в {growth.index[0]} г.) = {macro.recession_prob:.0%} (Gradient Boosting)",
                            f"прогноз роста ВВП {growth.index[0]} г. = {g_next:.2f}% (ARIMA{macro.extra['arima_order']}, сценарий {config.SCENARIOS[sc]['label']})",
                            f"фаза цикла: {macro.phase}"],
                  "result": f"доля рисковых активов = {risky:.0%}", "trail": trail})

    # ---- Module II -> safe part and FX hedge
    usd = float(np.clip(0.15 + 0.60 * fx.fx_risk, 0.0, 0.80))
    steps.append({"module": "II", "title": "Валютный риск → хедж безрисковой части",
                  "facts": [f"индекс валютного риска = {fx.fx_risk:.2f} (режим: {fx.regime.lower()})",
                            f"ожидаемое ослабление тенге к USD ≈ {fx.exp_depreciation:.1%} в год",
                            f"USD, EUR, CNY — один фактор: PC1 объясняет {fx.pc1_share:.0%} дисперсии"],
                  "result": f"в USD-инструментах {usd:.0%} безрисковой части", "trail": []})
    if fx.fx_risk >= 0.66:
        warns.append("Высокий валютный риск: рассмотрите увеличение валютного хеджа сверх модельного.")

    # ---- Module III -> caps
    cols = list(prices.columns)
    labels = clusters.labels["K-medoids"].reindex(cols)
    U = clusters.U.reindex(cols).to_numpy() if settings.use_fuzzy_caps else None
    wm, cc = profile.max_asset_weight, profile.cluster_cap
    if U is None:
        wm, cc, relaxed = feasible_limits(labels.to_numpy(), wm, cc)
        if relaxed:
            warns.append(f"Лимиты профиля несовместимы со структурой кластеров и ослаблены до "
                         f"≤ {cc:.0%} на кластер и ≤ {wm:.0%} на актив.")
    steps.append({"module": "III", "title": "Кластеризация → ограничения диверсификации",
                  "facts": [f"K-medoids: {clusters.k} кластера, медоиды {', '.join(clusters.medoids)}",
                            f"силуэт = {clusters.metrics.loc['K-medoids', 'silhouette']:.2f}",
                            "ограничение " + ("по матрице принадлежности U (FCM)" if U is not None else "по меткам K-medoids")],
                  "result": f"≤ {cc:.0%} на кластер, ≤ {wm:.0%} на актив", "trail": []})

    # ---- Module IV -> weights within the risky part
    s = signals.ranks.iloc[-1].reindex(cols).fillna(0.5).to_numpy()
    rets = prices.pct_change().dropna().iloc[-config.COV_WINDOW:]
    cov = covariance(rets, settings.cov_method)
    w = max_signal_to_risk(s, cov, profile.max_asset_weight, labels.to_numpy(), U, profile.cluster_cap)
    wr = pd.Series(w, index=cols).sort_values(ascending=False)
    top = ", ".join(f"{a} {v:.0%}" for a, v in wr.head(3).items())
    steps.append({"module": "IV", "title": "Ранговые сигналы → веса (max сигнал/риск)",
                  "facts": [f"модели: {', '.join(signals.models)}; сглаживание 10; s = Rank/N",
                            f"IC ансамбля вне выборки = {signals.metrics.iloc[-1]['IC']:.3f}",
                            f"ковариация: {settings.cov_method}, окно {config.COV_WINDOW} дн."],
                  "result": f"крупнейшие позиции: {top}", "trail": []})
    ic = signals.metrics.iloc[-1]["IC"]
    if not np.isfinite(ic) or ic <= 0:
        warns.append("Сигналы Модуля IV не показали положительной предсказательной силы вне выборки (IC ≤ 0): "
                     "веса определяются в основном ковариацией и ограничениями.")

    # ---- final allocation
    sig = pd.Series(s, index=cols)
    alloc = pd.DataFrame({"weight": wr * risky, "group": "Рисковая часть",
                          "cluster": [f"К{int(labels[a]) + 1}" for a in wr.index],
                          "sector": [sectors.get(a, "") if sectors is not None else "" for a in wr.index],
                          "signal": sig.reindex(wr.index)})
    safe = 1 - risky
    alloc.loc[SAFE_KZT] = [safe * (1 - usd), "Безрисковая часть", "", "", np.nan]
    alloc.loc[SAFE_USD] = [safe * usd, "Безрисковая часть", "", "", np.nan]
    alloc.index.name = "asset"
    vol_r = float(np.sqrt(w @ cov @ w))
    exp_vol = risky * vol_r
    var95 = 1.645 * exp_vol * np.sqrt(21 / 252)
    exp_cl = wr.groupby(labels.reindex(wr.index)).sum()
    exp_cl.index = [clusters.names.get(int(c), str(c)) for c in exp_cl.index]
    if g_next < 1.0:
        warns.append("Прогноз роста ниже 1%: высокий риск рецессии, пересмотрите горизонт инвестирования.")
    return Decision(profile, sc, risky, usd, alloc, wr, exp_vol, float(var95), exp_cl * risky, steps, warns)
