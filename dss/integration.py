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

The coefficients live in :data:`dss.config.RULES`; :func:`sensitivity_analysis`
varies each of them and checks that the rules stay monotone (higher risk ->
fewer risky assets and more hedge), as required by Sec. 10 of the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import config
from .portfolio import covariance, feasible_limits, max_signal_to_risk

PHASE_MULT = config.PHASE_RISK_MULT       # kept for backward compatibility
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


def _num(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def risk_budget(profile: config.InvestorProfile, rec_prob: float, scenario: str,
                phase: str, growth_next: float,
                rules: config.IntegrationRules = config.RULES) -> tuple[float, list]:
    """Rule 1: share of risky assets from Module I."""
    base = profile.base_risky_share
    m_rec = 1 - rules.rec_penalty * rec_prob
    m_sc = rules.scenario_mult[scenario]
    m_ph = rules.phase_mult.get(phase, 1.0)
    share = base * m_rec * m_sc * m_ph - (rules.neg_growth_cut if growth_next < 0 else 0.0)
    share = float(np.clip(share, profile.min_risky_share, profile.max_risky_share))
    trail = [f"базовая доля рисковых активов ({profile.label.lower()} профиль) — {base:.0%}",
             f"× (1 − {_num(rules.rec_penalty)}·P(рецессии)) = × {m_rec:.2f}",
             f"× сценарий «{config.SCENARIOS[scenario]['label']}» = × {m_sc:.2f}",
             f"× фаза цикла «{phase}» = × {m_ph:.2f}"]
    if growth_next < 0:
        trail.append(f"− {rules.neg_growth_cut * 100:g} п.п. за отрицательный прогноз роста")
    return share, trail


def fx_hedge(fx_risk: float, rules: config.IntegrationRules = config.RULES) -> float:
    """Rule 2: USD share of the safe part from the Module II FX risk index."""
    return float(np.clip(rules.hedge_base + rules.hedge_slope * fx_risk, 0.0, rules.hedge_max))


def decide(macro, fx, clusters, signals, prices: pd.DataFrame, settings: config.Settings,
           sectors: pd.Series | None = None, rules: config.IntegrationRules = config.RULES) -> Decision:
    profile = config.PROFILES[settings.profile]
    sc = settings.scenario
    growth = macro.scenario_path(sc)
    g_next = float(growth.iloc[0])
    steps, warns = [], []

    # ---- Module I -> risk budget
    risky, trail = risk_budget(profile, macro.recession_prob, sc, macro.phase, g_next, rules)
    steps.append({"module": "I", "title": "Макропрогноз → бюджет риска",
                  "facts": [f"P(рецессии в {growth.index[0]} г.) = {macro.recession_prob:.0%} (Gradient Boosting)",
                            f"прогноз роста ВВП {growth.index[0]} г. = {g_next:.2f}% (ARIMA{macro.extra['arima_order']}, сценарий {config.SCENARIOS[sc]['label']})",
                            f"фаза цикла: {macro.phase}"],
                  "result": f"доля рисковых активов = {risky:.0%}", "trail": trail})

    # ---- Module II -> safe part and FX hedge
    usd = fx_hedge(fx.fx_risk, rules)
    steps.append({"module": "II", "title": "Валютный риск → хедж безрисковой части",
                  "facts": [f"индекс валютного риска = {fx.fx_risk:.2f} (режим: {fx.regime.lower()})",
                            f"ожидаемое ослабление тенге к USD ≈ {fx.exp_depreciation:.1%} в год",
                            f"USD, EUR, CNY — один фактор: PC1 объясняет {fx.pc1_share:.0%} дисперсии"],
                  "result": f"в USD-инструментах {usd:.0%} безрисковой части", "trail": []})
    if fx.fx_risk >= rules.fx_risk_warn:
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
    elif cc < 1.0 / U.shape[1]:
        cc = 1.0 / U.shape[1] + 1e-6       # same relaxation as the optimiser's fuzzy constraints
        warns.append(f"Лимит профиля на кластер ниже 1/k при k = {U.shape[1]} нечётких кластерах "
                     f"и ослаблен до {cc:.0%} по матрице U.")
    steps.append({"module": "III", "title": "Кластеризация → ограничения диверсификации",
                  "facts": [f"K-medoids: {clusters.k} кластера, медоиды {', '.join(clusters.medoids)}",
                            f"силуэт = {clusters.metrics.loc['K-medoids', 'silhouette']:.2f}",
                            "ограничение " + ("по матрице принадлежности U (FCM)" if U is not None else "по меткам K-medoids")],
                  "result": f"≤ {cc:.0%} на кластер, ≤ {wm:.0%} на актив", "trail": []})

    # ---- Module IV -> weights within the risky part
    s = signals.ranks.iloc[-1].reindex(cols).fillna(0.5).to_numpy()
    rets = prices.pct_change().dropna().iloc[-config.COV_WINDOW:]
    cov = covariance(rets, settings.cov_method)
    w, ok = max_signal_to_risk(s, cov, profile.max_asset_weight, labels.to_numpy(), U, profile.cluster_cap,
                               return_ok=True)
    if not ok:
        warns.append("Оптимизатор Модуля IV не сошёлся: показан допустимый портфель, который распределяет "
                     "вес между кластерами в пределах лимитов без учёта сигналов. Попробуйте другой метод "
                     "ковариации или профиль.")
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
    if g_next < rules.low_growth_warn:
        warns.append(f"Прогноз роста ниже {_num(rules.low_growth_warn)}%: высокий риск рецессии, "
                     "пересмотрите горизонт инвестирования.")
    return Decision(profile, sc, risky, usd, alloc, wr, exp_vol, float(var95), exp_cl * risky, steps, warns)


# --------------------------------------------------------------------------
# Sensitivity of the integration rules (Sec. 10 of the spec)
# --------------------------------------------------------------------------
SENS_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
SCENARIO_ORDER = ("bull", "base", "bear", "stress")             # from benign to adverse
PHASE_ORDER = ("Расширение", "Восстановление", "Замедление", "Спад")


def _scaled(rules: config.IntegrationRules, name: str, f: float, scenario: str, phase: str):
    if name == "scenario_mult":
        return replace(rules, scenario_mult=rules.scenario_mult | {scenario: rules.scenario_mult[scenario] * f})
    if name == "phase_mult":
        return replace(rules, phase_mult=rules.phase_mult | {phase: rules.phase_mult.get(phase, 1.0) * f})
    return replace(rules, **{name: getattr(rules, name) * f})


def sensitivity_analysis(profile: config.InvestorProfile, rec_prob: float, scenario: str, phase: str,
                         growth_next: float, fx_risk: float,
                         rules: config.IntegrationRules = config.RULES) -> dict:
    """One-at-a-time sensitivity of the decision to each rule coefficient and a
    check that the rules are monotone over the whole tested range.

    Returns ``coefficients`` (coefficient x factor -> output share, plus the arc
    elasticity between x0.75 and x1.25), ``risk_grid`` (risky share over
    P(recession) x scenario), ``hedge_grid`` (USD share over the FX risk index)
    and ``monotonicity`` (property -> holds, worst case).
    """
    def d(r=rules, p=rec_prob, sc=scenario, ph=phase, g=growth_next, prof=profile):
        return risk_budget(prof, p, sc, ph, g, r)[0]

    def h(r=rules, x=fx_risk):
        return fx_hedge(x, r)

    coefs = [("rec_penalty", "Штраф за P(рецессии), правило 1", d),
             ("neg_growth_cut", "Снижение при росте < 0, правило 1", d),
             ("scenario_mult", f"Множитель сценария «{config.SCENARIOS[scenario]['label']}», правило 1", d),
             ("phase_mult", f"Множитель фазы «{phase}», правило 1", d),
             ("hedge_base", "Базовая доля USD-хеджа, правило 2", h),
             ("hedge_slope", "Наклон хеджа по индексу FX-риска, правило 2", h)]
    rows = {}
    for name, label, fn in coefs:
        out = {f"×{f:g}".replace(".", ","): fn(r=_scaled(rules, name, f, scenario, phase)) for f in SENS_FACTORS}
        lo, mid, hi = (fn(r=_scaled(rules, name, f, scenario, phase)) for f in (0.75, 1.0, 1.25))
        value = (rules.scenario_mult[scenario] if name == "scenario_mult" else
                 rules.phase_mult.get(phase, 1.0) if name == "phase_mult" else getattr(rules, name))
        rows[label] = {"Значение": value, "Выход": "доля рисковых" if fn is d else "USD-хедж"} | out | {
            "Эластичность": (hi - lo) / 0.5 / mid if mid else np.nan}
    coefficients = pd.DataFrame(rows).T

    probs = np.round(np.linspace(0, 1, 11), 2)
    risk_grid = pd.DataFrame({config.SCENARIOS[sc]["label"]: [d(p=p, sc=sc) for p in probs]
                              for sc in SCENARIO_ORDER}, index=pd.Index(probs, name="P(рецессии)"))
    fxs = np.round(np.linspace(0, 1, 11), 2)
    hedge_grid = pd.DataFrame({"USD-хедж": [h(x=x) for x in fxs]}, index=pd.Index(fxs, name="Индекс FX-риска"))

    # monotonicity over every profile, scenario and phase: at the current
    # coefficients and for every one-at-a-time variant of the tested range
    def margins(variants) -> dict:
        m = {"Доля рисковых не растёт с P(рецессии)": [],
             "Доля рисковых не растёт при ухудшении сценария (Bull → Stress)": [],
             "Доля рисковых не растёт при ухудшении фазы цикла": [],
             "Доля рисковых не растёт при отрицательном прогнозе роста": [],
             "USD-хедж не убывает с индексом FX-риска": []}
        for r in variants:
            m["USD-хедж не убывает с индексом FX-риска"].append(np.diff([h(r=r, x=x) for x in fxs]).min())
            for prof in config.PROFILES.values():
                for ph in PHASE_ORDER:
                    for sc in SCENARIO_ORDER:
                        ds = [d(r=r, p=p, sc=sc, ph=ph, prof=prof) for p in probs]
                        m["Доля рисковых не растёт с P(рецессии)"].append(-np.diff(ds).max())
                        m["Доля рисковых не растёт при отрицательном прогнозе роста"].append(
                            d(r=r, sc=sc, ph=ph, prof=prof, g=1.0) - d(r=r, sc=sc, ph=ph, prof=prof, g=-1.0))
                    ds = [d(r=r, sc=sc, ph=ph, prof=prof) for sc in SCENARIO_ORDER]
                    m["Доля рисковых не растёт при ухудшении сценария (Bull → Stress)"].append(-np.diff(ds).max())
                ds = [d(r=r, ph=ph, prof=prof) for ph in PHASE_ORDER]
                m["Доля рисковых не растёт при ухудшении фазы цикла"].append(-np.diff(ds).max())
        return {k: min(v) for k, v in m.items()}

    now = margins([rules])
    span = margins([_scaled(rules, n, f, scenario, phase) for n, _, _ in coefs for f in SENS_FACTORS])
    yes = {True: "да", False: "нет"}
    monotonicity = pd.DataFrame({"При текущих коэффициентах": [yes[v >= -1e-12] for v in now.values()],
                                 "При изменении одного коэффициента ×0,5…×1,5": [yes[v >= -1e-12]
                                                                                 for v in span.values()],
                                 "Худший случай, п.п.": [100 * v + 0.0 for v in span.values()]},
                                index=pd.Index(list(now), name="Свойство"))
    return {"coefficients": coefficients, "risk_grid": risk_grid, "hedge_grid": hedge_grid,
            "monotonicity": monotonicity}
