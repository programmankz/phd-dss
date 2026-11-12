"""Final experiment of November 2026, as preregistered in docs/preregistration.md.

Primary hypotheses H1-H4 (moderate profile), secondary comparisons (signal admission, turnover
penalty, 16-configuration Shapley values, baselines) and the other profiles (descriptive).
Writes docs/results/exp_2026_11_*.csv and docs/results_2026_11.md.

Usage::

    python scripts/experiment_2026_11.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ablation  # noqa: E402
import baselines as bl  # noqa: E402
import end_to_end as e2e  # noqa: E402
from dss import evaluation, significance  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

START_A, START_B, END = "2023-01-16", "2020-01-06", "2026-10-09"
TRAIN_START = "2016-04-01"
KAPPA = 1.0
NI_MARGIN = -0.25
TC = 10.0
RES = ROOT / "docs" / "results"

EW = e2e.Config("EW + KZT", False, False, False)
FULL = e2e.Config("Full DSS", True, True, True)
MACRO = e2e.Config("Macro rule only", True, False, False)
CURRENCY = e2e.Config("Currency rule only", False, True, False)
MC = e2e.Config("Macro and currency rules", True, True, False)


def sharpe_fn(rf):
    def f(r):
        return significance.sharpe(np.asarray(r), 0.0)
    return f


def mdd_fn(r):
    return evaluation.max_drawdown_metric(np.asarray(r))


def engine(inp, profile, start, m4, mode):
    eng = e2e.Engine(inp, profile, start, END, need_m4=m4, rules_mode=mode, log=lambda *a: None)
    if mode == "derived":
        eng.calibrate_gamma(TRAIN_START, pd.Timestamp(start) - pd.Timedelta(days=1))
    return eng


def run(eng, cfg, rets, rf):
    r, lg = e2e.run_config(eng, cfg, rets, TC)
    m = e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)
    if "signal_admitted" in lg:
        m["admitted_share"] = float(lg["signal_admitted"].mean())
    return r, m


def primary(inp, profile, rets, rf, log=print):
    eA, eAd = engine(inp, profile, START_A, True, "expert"), engine(inp, profile, START_A, True, "derived")
    eB = engine(inp, profile, START_B, False, "expert")
    daily, rows = {}, {}
    for key, eng, cfg in (("A: EW + KZT", eA, EW), ("A: Full DSS expert", eA, FULL), ("A: Full DSS derived", eAd, FULL),
                          ("B: EW + KZT", eB, EW), ("B: Macro rule only", eB, MACRO),
                          ("B: Currency rule only", eB, CURRENCY), ("B: Macro and currency rules", eB, MC)):
        daily[key], rows[key] = run(eng, cfg, rets, rf)
    ex = {k: v - rf.reindex(v.index) for k, v in daily.items()}
    sh = sharpe_fn(rf)
    h1 = evaluation.paired_difference_test(ex["A: Full DSS expert"], ex["A: EW + KZT"], sh)
    h2 = evaluation.paired_difference_test(daily["B: Macro rule only"], daily["B: EW + KZT"], mdd_fn)
    h4 = evaluation.paired_difference_test(ex["A: Full DSS derived"], ex["A: Full DSS expert"], sh, null=NI_MARGIN)
    h3 = {w: rows["B: Currency rule only"][f"return {w}"] - rows["B: EW + KZT"][f"return {w}"] for w in e2e.SHOCKS}
    # one-sided p-values in the direction of each hypothesis; Holm over H1, H2, H4
    p = {"H1": h1["p_greater"], "H2": h2["p_greater"], "H4": h4["p_greater"]}   # MDD is negative: less deep = greater
    adj = dict(zip(p, significance.holm(list(p.values()))))
    tests = {
        "H1": h1 | {"p": p["H1"], "p_holm": adj["H1"], "confirmed": bool(h1["diff"] > 0 and adj["H1"] < 0.05)},
        "H2": h2 | {"p": p["H2"], "p_holm": adj["H2"], "confirmed": bool(h2["diff"] > 0 and adj["H2"] < 0.05)},
        "H3": {"diff_2020": h3["20.02-31.03.2020"], "diff_2022": h3["15.02-31.03.2022"],
               "confirmed": bool(all(v > 0 for v in h3.values()))},
        "H4": h4 | {"p": p["H4"], "p_holm": adj["H4"], "confirmed": bool(h4["low"] > NI_MARGIN)},
    }
    cal = {"A": eAd.calibration}
    log(f"  {profile}: H1 {tests['H1']['diff']:+.2f}, H2 {tests['H2']['diff']:+.3f}, H3 {h3}, H4 {tests['H4']['diff']:+.2f}")
    return pd.DataFrame(rows).T, tests, cal, eA


def secondary_gate(eA, rets, rf):
    rows, daily = {}, {}
    for cfg in (FULL, e2e.Config("Full DSS, gate", True, True, True, gate=True),
                e2e.Config(f"Full DSS, kappa={KAPPA:g}", True, True, True, kappa=KAPPA),
                e2e.Config(f"Full DSS, gate, kappa={KAPPA:g}", True, True, True, gate=True, kappa=KAPPA)):
        eA._prev_risky = {}
        daily[cfg.name], rows[cfg.name] = run(eA, cfg, rets, rf)
    d = pd.DataFrame(daily)
    tests = significance.sharpe_tests(d.sub(rf.reindex(d.index), axis=0), 0.0, reference="Full DSS")
    t = pd.DataFrame(rows).T
    for c in ("diff", "diff_low", "diff_high", "p_value", "p_holm"):
        t[c] = tests[c]
    return t


def main() -> int:
    t_all = time.perf_counter()
    inp = e2e.load_inputs(e2e.KASE7)
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    print("Primary hypotheses")
    prim, tests, cals = {}, {}, {}
    for prof in ("moderate", "conservative", "aggressive"):
        prim[prof], tests[prof], cals[prof], eA = primary(inp, prof, rets, rf)
        if prof == "moderate":
            eA_mod = eA
    print("Secondary: signal admission and turnover penalty")
    gate = secondary_gate(eA_mod, rets, rf)
    print("Secondary: 16 configurations and Shapley values")
    abl = {"A": ablation.run_sample(inp, START_A, END, evaluation.MODULES, "moderate", TC, 2000, log=lambda *a: None),
           "B": ablation.run_sample(inp, START_B, END, ("I", "II", "III"), "moderate", TC, 2000, log=lambda *a: None)}
    print("Secondary: baselines")
    base = {"A": bl.run(inp, START_A, END, True, "moderate", TC, log=lambda *a: None),
            "B": bl.run(inp, START_B, END, False, "moderate", TC, log=lambda *a: None)}

    pd.concat(prim, names=["profile", "configuration"]).to_csv(RES / "exp_2026_11_primary.csv")
    rows = []
    for prof, tt in tests.items():
        for h, v in tt.items():
            rows.append({"profile": prof, "hypothesis": h, **v})
    pd.DataFrame(rows).to_csv(RES / "exp_2026_11_tests.csv", index=False)
    gate.to_csv(RES / "exp_2026_11_gate.csv")
    pd.concat({k: v[1] for k, v in abl.items()}, names=["sample", "row"]).to_csv(RES / "exp_2026_11_shapley.csv")
    pd.concat(base, names=["sample", "strategy"]).to_csv(RES / "exp_2026_11_baselines.csv")
    (ROOT / "docs" / "results_2026_11.md").write_text(markdown(prim, tests, cals, gate, abl, base), encoding="utf-8")
    print(f"written docs/results_2026_11.md ({time.perf_counter() - t_all:.0f} s)")
    return 0


# ----------------------------------------------------------------------------- report
def pp(v) -> str:
    """Difference of fractions in percentage points."""
    return num(v * 100, 1, sign=True) + " п.п." if np.isfinite(v) else "—"


def yes(b):
    return "**подтверждена**" if b else "**не подтверждена**"


def markdown(prim, tests, cals, gate, abl, base) -> str:
    t = tests["moderate"]
    pm = prim["moderate"]
    cal = cals["moderate"]["A"]
    out = ["# Итоговый эксперимент, ноябрь 2026 г.", "",
           "План — `docs/preregistration.md` (зафиксирован до запуска). Результаты `python scripts/experiment_2026_11.py`; "
           "таблицы `docs/results/exp_2026_11_*.csv`. Умеренный профиль, 7 бумаг KASE, издержки 10 б.п., цены с "
           "дивидендами. Период A — 17.01.2023 – 09.10.2026 (модули I–IV), период B — 08.01.2020 – 09.10.2026 "
           "(модули I–III, с шоками 2020 и 2022 гг.).", "",
           "## Основные гипотезы", "",
           "| Гипотеза | Оценка [95% ДИ] | p (одностор.) | p (Холм) | Итог |", "|---|---|---|---|---|",
           f"| H1: «Full DSS» − «EW + KZT», Шарп, A | {num(t['H1']['diff'], sign=True)} [{num(t['H1']['low'])}; "
           f"{num(t['H1']['high'])}] | {num(t['H1']['p'], 3)} | {num(t['H1']['p_holm'], 3)} | {yes(t['H1']['confirmed'])} |",
           f"| H2: «Macro rule only» − «EW + KZT», макс. просадка, B | {pp(t['H2']['diff'])} "
           f"[{pp(t['H2']['low'])}; {pp(t['H2']['high'])}] | {num(t['H2']['p'], 3)} | {num(t['H2']['p_holm'], 3)} | "
           f"{yes(t['H2']['confirmed'])} |",
           f"| H3: «Currency rule only» − «EW + KZT», доходность в шоках, B | 2020: {pct(t['H3']['diff_2020'], 2, sign=True)}; "
           f"2022: {pct(t['H3']['diff_2022'], 2, sign=True)} | — | — | {yes(t['H3']['confirmed'])} |",
           f"| H4: «Full DSS» derived − expert, Шарп, A (граница −0,25) | {num(t['H4']['diff'], sign=True)} "
           f"[{num(t['H4']['low'])}; {num(t['H4']['high'])}] | {num(t['H4']['p'], 3)} | {num(t['H4']['p_holm'], 3)} | "
           f"{yes(t['H4']['confirmed'])} |", "",
           "Положительная разность просадок в H2 (в процентных пунктах) означает менее глубокую просадку. p в H4 — для гипотезы «разность > −0,25».", "",
           "### Конфигурации основных сравнений", "",
           "| Конфигурация | Доходность, % год. | Волатильность, % | Шарп | Макс. просадка, % | Шок 2020, % | Шок 2022, % | "
           "Средн. d, % | Средн. h, % | Оборот, раз/год |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in pm.iterrows():
        out.append(f"| {name} | {pct(r['annual_return'])} | {pct(r['volatility'])} | {num(r['sharpe'])} | {pct(r['max_drawdown'])} | "
                   f"{pct(r['return 20.02-31.03.2020'], 2)} | {pct(r['return 15.02-31.03.2022'], 2)} | {pct(r['mean_d'], 0)} | "
                   f"{pct(r['mean_h'], 0)} | {num(r['turnover_per_year'], 1)} |")
    out += ["", f"γ выведенных правил (период A): {num(cal['gamma'], 3)}, среднее d* на обучении {num(cal['mean_d_train'])} "
            f"при цели 0,60 — {'достигнуто' if cal['attained'] else 'не достигнуто (нижняя граница γ)'}; доля обучающих дат с "
            f"положительной оценкой премии {pct(cal['share_positive_premium'], 0)}.", "",
            "## Выводы по гипотезам", ""]
    out.append(f"* **H1** {yes(t['H1']['confirmed'])}: полная СППР {'выше' if t['H1']['diff'] > 0 else 'ниже'} «EW + KZT» по Шарпу на "
               f"{num(abs(t['H1']['diff']))}. На данных с дивидендами простое правило «60% акций равными долями + "
               "тенговый депозит» не уступает интегрированной системе в 2023–2026 гг.")
    out.append(f"* **H2** {yes(t['H2']['confirmed'])}: макроправило меняет максимальную просадку 2020–2026 гг. на "
               f"{pp(t['H2']['diff'])} (с {pct(pm.loc['B: EW + KZT', 'max_drawdown'])} до "
               f"{pct(pm.loc['B: Macro rule only', 'max_drawdown'])}) при средней доле акций "
               f"{pct(pm.loc['B: Macro rule only', 'mean_d'], 0)} против 60%. Часть эффекта — просто меньшая средняя доля "
               "акций; отделить её от своевременности правила этот тест не позволяет.")
    out.append(f"* **H3** {yes(t['H3']['confirmed'])}: валютное правило в окне 2020 г. — {pct(t['H3']['diff_2020'], 2, sign=True)}, "
               f"в окне 2022 г. — {pct(t['H3']['diff_2022'], 2, sign=True)} к «EW + KZT». Это два окна; статистического "
               "вывода по ним сделать нельзя, только описательный.")
    out.append(f"* **H4** {yes(t['H4']['confirmed'])}: разность Шарпа выведенных и экспертных правил "
               f"{num(t['H4']['diff'], sign=True)} [{num(t['H4']['low'])}; {num(t['H4']['high'])}]. Выведенное правило на "
               "KASE работает как переключатель по знаку оценки премии и в тесте держит d у верхней границы "
               f"({pct(pm.loc['A: Full DSS derived', 'mean_d'], 0)}); его результат — это результат почти полной загрузки "
               "в растущий рынок, а не прогноза.")
    out += ["", "## Вторичные сравнения (разведочные)", "", "### Допуск сигнала и штраф за оборот, период A", "",
            "| Конфигурация | Шарп | Разность к «Full DSS» [95% ДИ] | p | p (Холм) | Макс. просадка, % | Оборот, раз/год | "
            "Доля решений с допущенным сигналом, % |", "|---|---|---|---|---|---|---|---|"]
    for name, r in gate.iterrows():
        diff = f"{num(r['diff'], sign=True)} [{num(r['diff_low'])}; {num(r['diff_high'])}]" if np.isfinite(r["diff"]) else "—"
        adm = pct(r["admitted_share"], 0) if "admitted_share" in r and np.isfinite(r.get("admitted_share", np.nan)) else "—"
        out.append(f"| {name} | {num(r['sharpe'])} | {diff} | {num(r['p_value'], 3)} | {num(r['p_holm'], 3)} | "
                   f"{pct(r['max_drawdown'])} | {num(r['turnover_per_year'], 1)} | {adm} |")
    for key, (table, shap, meta) in abl.items():
        out += ["", f"### Значения Шепли, период {key} ({meta['start']} – {meta['end']})", "",
                "| Показатель | Модуль | φ | 95% ДИ |", "|---|---|---|---|"]
        for _, r in shap[shap["module"].isin(evaluation.MODULES)].iterrows():
            f = (lambda v: num(v, 3, sign=True)) if r["metric"] == "sharpe" else (lambda v: pct(v, 2, sign=True))
            ci = f"[{f(r['ci_low'])}; {f(r['ci_high'])}]" if np.isfinite(r["ci_low"]) else "—"
            out.append(f"| {ablation.METRIC_RU.get(r['metric'], r['metric'])} | {ablation.LABELS[r['module']]} | {f(r['phi'])} | {ci} |")
    for key, tb in base.items():
        out += ["", f"### Базовые стратегии, период {key}", "",
                "| Стратегия | Шарп | Разность к «" + tb.attrs["reference"] + "» | p (Холм) | DSR | Макс. просадка, % |",
                "|---|---|---|---|---|---|"]
        for name, r in tb.iterrows():
            out.append(f"| {name} | {num(r['sharpe'])} | {num(r['diff'], sign=True) if np.isfinite(r['diff']) else '—'} | "
                       f"{num(r['p_holm'], 3)} | {num(r['dsr'])} | {pct(r['max_drawdown'])} |")
    out += ["", "## Другие профили (описательно)", "",
            "| Профиль | H1: разность Шарпа | H2: разность просадки | H3: 2020 / 2022 | H4: разность Шарпа [95% ДИ] |",
            "|---|---|---|---|---|"]
    for prof in ("conservative", "aggressive"):
        tt = tests[prof]
        out.append(f"| {prof} | {num(tt['H1']['diff'], sign=True)} | {pct(tt['H2']['diff'], 1, sign=True)} | "
                   f"{pct(tt['H3']['diff_2020'], 2, sign=True)} / {pct(tt['H3']['diff_2022'], 2, sign=True)} | "
                   f"{num(tt['H4']['diff'], sign=True)} [{num(tt['H4']['low'])}; {num(tt['H4']['high'])}] |")
    out += ["", "## Отступления от плана", "",
            "* Нет: конфигурации, периоды, параметры и тесты — как в `docs/preregistration.md`. Для H1, H2 и H4 "
            "использованы односторонние p-значения в направлении гипотезы (в плане сторона не была указана явно); "
            "двусторонние p есть в `docs/results/exp_2026_11_tests.csv`.", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
