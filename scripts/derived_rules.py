"""Expert integration rules (5)-(6) versus the derived rules d*, h* (rules="derived").

For every profile gamma is calibrated on the training part before the test (mean d* = d0) and
fixed; then Tables 4 and 5 are run with both rule sets on the same engine. Also reports the
inputs of the derived rules at the first test date and d*(pi) on a grid of pi (monotonicity).

Outputs: docs/results/derived_rules.csv, derived_rules_gamma.csv, docs/derived_rules.md.

Usage::

    python scripts/derived_rules.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
from dss import config, integration  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

TRAIN_START = "2016-04-01"
SAMPLES = {"Таблица 4 (2023–2026, модули I–IV)": ("2023-01-16", e2e.TABLE4, True, "Full DSS"),
           "Таблица 5 (2020–2026, модули I–II)": ("2020-01-06", e2e.TABLE5, False, "Macro and currency rules")}


def run(inp, start, end, configs, m4, profile, mode, tc):
    eng = e2e.Engine(inp, profile, start, end, need_m4=m4, rules_mode=mode, log=lambda *a: None)
    cal = None
    if mode == "derived":
        eng.calibrate_gamma(TRAIN_START, pd.Timestamp(start) - pd.Timedelta(days=1))
        cal = eng.calibration
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    rows = {}
    for cfg in configs:
        r, lg = e2e.run_config(eng, cfg, rets, tc)
        rows[cfg.name] = e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)
    return pd.DataFrame(rows).T, eng, cal


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--end", default="2026-10-09")
    p.add_argument("--tc", type=float, default=10.0)
    a = p.parse_args(argv)
    inp = e2e.load_inputs(e2e.KASE7)
    tables, gammas, inputs, grids = [], [], {}, {}
    for key, (start, configs, m4, full) in SAMPLES.items():
        for prof in config.PROFILES:
            for mode in ("expert", "derived"):
                cfgs = configs if prof == "moderate" else [c for c in configs if c.name in ("EW + KZT", full)]
                t, eng, cal = run(inp, start, a.end, cfgs, m4, prof, mode, a.tc)
                t.insert(0, "rules", mode)
                t.insert(0, "profile", prof)
                t.insert(0, "sample", key)
                tables.append(t)
                print(f"{key} {prof} {mode}: {full} Sharpe {t.loc[full, 'sharpe']:.2f}, mean d {t.loc[full, 'mean_d']:.2f}")
                if cal:
                    gammas.append({"sample": key, "profile": prof, **cal})
                    if prof == "moderate":
                        t0 = eng.rebal[0]
                        der = eng.derived_at(t0)
                        inputs[key] = (t0, der, eng.macro_at(t0)["rec_prob"])
                        grids[key] = [(pi, integration.derived_risk_budget(pi, der["moments"], der["r_kzt"], eng.gamma,
                                                                           eng.profile),
                                       integration.mixture_moments(pi, der["moments"])[0] - der["r_kzt"])
                                      for pi in np.linspace(0, 1, 11)]
    res = pd.concat(tables)
    res.index.name = "configuration"
    out = ROOT / "docs" / "results"
    res.to_csv(out / "derived_rules.csv")
    g = pd.DataFrame(gammas)
    g.to_csv(out / "derived_rules_gamma.csv", index=False)
    (ROOT / "docs" / "derived_rules.md").write_text(markdown(res, g, inputs, grids), encoding="utf-8")
    print("written docs/derived_rules.md")
    return 0


def markdown(res, g, inputs, grids) -> str:
    out = ["# Выведенные правила интеграции d*, h* против экспертных правил (5)–(6)", "",
           "Результаты `python scripts/derived_rules.py`; таблицы `docs/results/derived_rules.csv` и "
           "`derived_rules_gamma.csv`. Экспертные правила статьи остаются вариантом по умолчанию (`rules=\"expert\"`).", "",
           "## Правила", "",
           "Из функции полезности E[R] − γ/2·Var[R] всего портфеля в тенге, R = d·R_E + (1 − d)[(1 − h)·r_KZT + "
           "h·(r_USD + δ)], условия первого порядка дают", "",
           "* d* = clip((μ(π) − r_KZT) / (γ·σ²(π)), d_min, d_max) при h = 0;",
           "* h* = clip((m/γ − d·Cov(R_E, δ)) / ((1 − d)·σ²_δ), 0, h_max), m = r_USD + E[δ] − r_KZT.", "",
           "μ(π), σ²(π) — моменты смеси двух режимов с весом π = P(рецессии) из Модуля I: μ(π) = (1 − π)μ_N + πμ_R, "
           "σ²(π) = (1 − π)σ²_N + πσ²_R + π(1 − π)(μ_N − μ_R)². Рисковая часть для оценки моментов — "
           "равновзвешенный рынок всех акций `prices_kase.csv` (история с 2008 г., с дивидендами); неблагоприятный "
           "режим — календарные годы, в которых опубликованный рост ВВП Казахстана ниже нижнего квартиля "
           "2000…Y (в данных с 2008 г. отрицательный рост был только в 2020 г., поэтому режим «рост < 0» "
           "оценить нельзя). Все моменты — только по дням до даты решения. E[δ] — ожидаемое ослабление тенге "
           "Модуля II; σ²_δ и Cov(R_E, δ) — по последним 756 торговым дням. Фаза цикла не используется. "
           "γ калибруется для каждого профиля на обучающей части (недельные даты с 01.04.2016 — первый винтаж "
           "Модуля I с тестовым годом — до начала теста) так, чтобы среднее d* равнялось d0, и затем фиксируется.", "",
           "## Калибровка γ", "",
           "| Выборка | Профиль | γ | Дат | Среднее d* на обучении | d0 | Доля дат с μ(π) > r_KZT | Достигнуто |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        d0 = config.PROFILES[r["profile"]].base_risky_share
        out.append(f"| {r['sample']} | {r['profile']} | {num(r['gamma'], 3)} | {r['n_dates']} | {num(r['mean_d_train'])} | "
                   f"{num(d0)} | {pct(r['share_positive_premium'], 0)} | {'да' if r['attained'] else 'нет'} |")
    out += ["", "Если ожидаемая избыточная доходность μ(π) − r_KZT отрицательна, d* = d_min при любом γ > 0; если "
            "положительна — d* убывает по γ от d_max. Поэтому среднее d* не может превысить долю дат с "
            "положительной премией × d_max + остальное × d_min. Когда цель d0 выше этой границы, калибровка "
            "упирается в нижнюю границу γ = 0,05 («нет» в последнем столбце), и правило работает как переключатель "
            "по знаку оценки премии.", ""]
    for key, (t0, der, pi) in inputs.items():
        m = der["moments"]
        out += [f"## Входы на первую дату теста: {key}, {t0:%d.%m.%Y}", "",
                f"μ_N = {pct(m.mu_n)}, σ_N = {pct(np.sqrt(m.var_n))} ({m.n_n} дн.); μ_R = {pct(m.mu_r)}, "
                f"σ_R = {pct(np.sqrt(m.var_r))} ({m.n_r} дн.; годы {', '.join(map(str, der['adverse_years']))}); "
                f"π = {num(pi)}; r_KZT = {pct(der['r_kzt'])}; m = {pct(der['m'])}; σ_δ = {pct(np.sqrt(der['var_delta']))}; "
                f"Cov(R_E, δ) = {num(der['cov_e_delta'], 4)}.", "",
                "| π | μ(π) − r_KZT | d* |", "|---|---|---|"]
        out += [f"| {num(pi_, 1)} | {pct(prem)} | {pct(d, 0)} |" for pi_, d, prem in grids[key]]
        mono = all(b[1] <= a_[1] + 1e-12 for a_, b in zip(grids[key], grids[key][1:]))
        out += ["", ("d* не возрастает по π на этой дате." if mono else
                     "**Контрпример монотонности:** d* возрастает по π, потому что в оценках по истории KASE "
                     "μ_R > μ_N — в годы низкого роста (2009, 2015–2016) тенговые цены акций росли вслед за "
                     "девальвацией. Это свойство данных, а не ошибка формулы: правило монотонно, только если "
                     "μ_R ≤ μ_N и σ_R ≥ σ_N (тест `test_derived_d_monotone_when_adverse_is_worse`)."), ""]
    out += ["## Результаты", "", "| Выборка | Профиль | Конфигурация | Правила | Доходность, % | Шарп | Макс. просадка, % | "
            "Средн. d, % | Средн. h, % | Оборот, раз/год |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.iterrows():
        out.append(f"| {r['sample']} | {r['profile']} | {name} | {r['rules']} | {pct(r['annual_return'])} | {num(r['sharpe'])} | "
                   f"{pct(r['max_drawdown'])} | {pct(r['mean_d'], 0)} | {pct(r['mean_h'], 0)} | {num(r['turnover_per_year'], 1)} |")
    out += ["", "## Оценка", "",
            "Выведенные правила заменяют восемь экспертных коэффициентов одним параметром γ на профиль, но на данных "
            "KASE их поведение определяется знаком и шумом исторических оценок премии: при калибровке, не достигающей "
            "d0, правило держит d* у верхней или нижней границы. Более высокий Шарп выведенных правил в тесте "
            "2023–2026 (если он есть в таблице) объясняется тем, что правило держало почти всю рисковую долю в "
            "растущем рынке, а не умением предсказывать; проверять его нужно на периодах с падением рынка и с "
            "поправкой на множественные сравнения (раздел 9). Для диссертации это аргумент в пользу явной "
            "постановки (utility-based) с осторожностью к оценкам μ: усадка оценок премии к априорному значению — "
            "следующий шаг, но её параметр должен быть зафиксирован до теста.", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
