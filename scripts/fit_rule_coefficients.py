"""Exploratory: coefficients of the expert rules fitted on 2008-2019 instead of being set by hand.

The coefficients of rules (5)-(6) - rec_penalty of the risky share, hedge_base and hedge_slope of the dollar
share - are chosen on a grid by the Sharpe ratio of the excess return over the tenge rate on 2008-2019 (three
stocks with a history from January 2008) and compared with the expert values on 2020-2026 (seven stocks), which
is not used in the fit. Each rule is fitted in its own configuration (macro rule only, currency rule only).
The defaults in dss.config are not changed. Not preregistered: the result is descriptive.
Writes docs/results/fit_rules_*.csv and docs/fit_rule_coefficients.md.

Usage::

    python scripts/fit_rule_coefficients.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
import experiment_2008_2019 as x  # noqa: E402
from dss import config, evaluation  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

RES = ROOT / "docs" / "results"
PENALTY = np.round(np.arange(0.0, 1.51, 0.1), 2)
BASE = np.round(np.arange(0.0, 0.81, 0.05), 2)
SLOPE = np.round(np.arange(0.0, 1.21, 0.1), 2)
NEAR = 0.02          # Sharpe distance from the optimum that is treated as indistinguishable


def stats(r: pd.Series, rf: pd.Series, lg: pd.DataFrame, windows: dict) -> dict:
    m = e2e.metrics(r, rf, 0.0, lg)
    out = {k: m[k] for k in ("annual_return", "volatility", "sharpe", "max_drawdown", "mean_d", "mean_h")}
    for name, (a, b) in windows.items():
        out[f"window {name}"] = x.window_return(r, a, b)
    return out


class Period:
    def __init__(self, tickers, start, end, windows):
        self.inp = e2e.load_inputs(tickers)
        self.rets = e2e.asset_returns(self.inp)
        self.rf = self.rets[e2e.KZT]
        self.eng = e2e.Engine(self.inp, "moderate", start, end, need_m4=False, log=lambda *a: None)
        self.windows = windows

    def run(self, cfg, rules):
        self.eng.rules = rules
        r, lg = e2e.run_config(self.eng, cfg, self.rets, x.TC)
        self.eng.rules = config.RULES
        return r, lg

    def row(self, cfg, rules):
        r, lg = self.run(cfg, rules)
        return stats(r, self.rf, lg, self.windows)


def fit(train: Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    d_rows = [{"rec_penalty": p} | train.row(x.MACRO, replace(config.RULES, rec_penalty=float(p))) for p in PENALTY]
    h_rows = [{"hedge_base": b, "hedge_slope": s}
              | train.row(x.CUR, replace(config.RULES, hedge_base=float(b), hedge_slope=float(s)))
              for b in BASE for s in SLOPE]
    return pd.DataFrame(d_rows), pd.DataFrame(h_rows)


def compare(test: Period, fitted: config.IntegrationRules) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = {}, []
    for cfg in (x.EW, x.MACRO, x.CUR, x.MC):
        re_, le = test.run(cfg, config.RULES)
        rows[(cfg.name, "expert")] = stats(re_, test.rf, le, test.windows)
        if cfg is x.EW:
            continue
        rf_, lf = test.run(cfg, fitted)
        rows[(cfg.name, "fitted")] = stats(rf_, test.rf, lf, test.windows)
        ex_f, ex_e = rf_ - test.rf.reindex(rf_.index), re_ - test.rf.reindex(re_.index)
        sh = evaluation.paired_difference_test(ex_f, ex_e, x.sharpe_fn)
        dd = evaluation.paired_difference_test(rf_, re_, x.mdd_fn)
        tests.append({"configuration": cfg.name, "sharpe_diff": sh["diff"], "sharpe_low": sh["low"], "sharpe_high": sh["high"],
                      "sharpe_p_two_sided": sh["p_two_sided"], "mdd_diff": dd["diff"], "mdd_low": dd["low"],
                      "mdd_high": dd["high"], "mdd_p_two_sided": dd["p_two_sided"]})
    t = pd.DataFrame(rows).T
    t.index.names = ["configuration", "coefficients"]
    return t, pd.DataFrame(tests)


RU = {"EW + KZT": "«EW + KZT»: 60% акций + тенге", "Macro rule only": "Правило для доли акций",
      "Currency rule only": "Правило для доли доллара", "Macro and currency rules": "Оба правила"}
KIND = {"expert": "назначенные", "fitted": "подобранные"}


def markdown(grid_d, grid_h, fitted, table, tests, train_tests) -> str:
    e = config.RULES
    exp_d = grid_d.loc[np.isclose(grid_d["rec_penalty"], e.rec_penalty)].iloc[0]
    exp_h = grid_h.loc[np.isclose(grid_h["hedge_base"], e.hedge_base) & np.isclose(grid_h["hedge_slope"], e.hedge_slope)].iloc[0]
    near_d = int((grid_d["sharpe"] >= grid_d["sharpe"].max() - NEAR).sum())
    near_h = int((grid_h["sharpe"] >= grid_h["sharpe"].max() - NEAR).sum())
    L = ["# Подбор коэффициентов экспертных правил по данным 2008–2019 гг.", "",
         "Разведочный расчёт, в план проверки гипотез не входил. Значения по умолчанию в `dss.config` не изменены.", "",
         "## Постановка", "",
         f"Коэффициенты правил (5)–(6) выбираются на сетке по коэффициенту Шарпа избыточной доходности над тенговой ставкой "
         f"на периоде {x.START_C[:4]}–{x.END_C[:4]} гг. (три акции с историей с января 2008 г.: {', '.join(x.KASE3)}). "
         "Каждое правило подбирается в своей конфигурации: доля акций без валютного правила, доля доллара без правила доли акций. "
         f"Сравнение с назначенными значениями выполнено на периоде {x.START_B[:4]}–{x.END_B[:4]} гг. (семь акций), "
         "который в подборе не участвовал. Издержки 10 б.п., бутстреп блочный.", "",
         f"Сетка: чувствительность к вероятности неблагоприятного режима от {num(PENALTY[0], 1)} до {num(PENALTY[-1], 1)} с шагом 0,1; "
         f"базовая доля доллара от {num(BASE[0], 2)} до {num(BASE[-1], 2)} с шагом 0,05; "
         f"наклон по индексу валютного риска от {num(SLOPE[0], 1)} до {num(SLOPE[-1], 1)} с шагом 0,1.", "",
         "## Результат подбора", "",
         "| Коэффициент | Назначено | Подобрано | Шарп при назначенном | Шарп при подобранном | Значений сетки в пределах 0,02 Шарпа от лучшего |",
         "|---|---|---|---|---|---|",
         f"| Чувствительность доли акций | {num(e.rec_penalty, 2)} | {num(fitted.rec_penalty, 2)} | {num(exp_d['sharpe'], 3)} | "
         f"{num(grid_d['sharpe'].max(), 3)} | {near_d} из {len(grid_d)} |",
         f"| Базовая доля доллара и наклон | {num(e.hedge_base, 2)} и {num(e.hedge_slope, 2)} | {num(fitted.hedge_base, 2)} и "
         f"{num(fitted.hedge_slope, 2)} | {num(exp_h['sharpe'], 3)} | {num(grid_h['sharpe'].max(), 3)} | {near_h} из {len(grid_h)} |", "",
         "Разность Шарпа «подобранные минус назначенные» на периоде подбора:", "",
         "| Коэффициенты | Разность | 95% интервал | p |", "|---|---|---|---|"]
    names = {"rec_penalty": "Чувствительность доли акций", "hedge_base, hedge_slope": "Базовая доля доллара и наклон"}
    for _, r in train_tests.iterrows():
        L.append(f"| {names[r['coefficients']]} | {num(r['sharpe_diff'], 3, sign=True)} | [{num(r['sharpe_low'], 3)}; "
                 f"{num(r['sharpe_high'], 3)}] | {num(r['sharpe_p_two_sided'], 3)} |")
    L += ["", f"## Сравнение на {x.START_B[:4]}–{x.END_B[:4]} гг.", "",
          "| Конфигурация | Коэффициенты | Доходность | Шарп | Макс. просадка | Средняя d | Средняя h | Доходность в окне 2020 | Доходность в окне 2022 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (c, k), r in table.iterrows():
        L.append(f"| {RU[c]} | {'—' if c == 'EW + KZT' else KIND[k]} | {pct(r['annual_return'], 1)} | {num(r['sharpe'], 2)} | "
                 f"{pct(r['max_drawdown'], 1)} | {pct(r['mean_d'], 0)} | {pct(r['mean_h'], 0)} | "
                 f"{pct(r['window 2020'], 1)} | {pct(r['window 2022'], 1)} |")
    L += ["", "Разности «подобранные минус назначенные» на тестовом периоде:", "",
          "| Конфигурация | Шарп | 95% интервал | p | Макс. просадка, п.п. | 95% интервал | p |", "|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        L.append(f"| {RU[r['configuration']]} | {num(r['sharpe_diff'], 2, sign=True)} | [{num(r['sharpe_low'], 2)}; {num(r['sharpe_high'], 2)}] | "
                 f"{num(r['sharpe_p_two_sided'], 3)} | {num(r['mdd_diff'] * 100, 1, sign=True)} | [{num(r['mdd_low'] * 100, 1)}; "
                 f"{num(r['mdd_high'] * 100, 1)}] | {num(r['mdd_p_two_sided'], 3)} |")
    L += ["", "Полные сетки: `docs/results/fit_rules_grid_d.csv`, `docs/results/fit_rules_grid_h.csv`.", ""]
    tail = ROOT / "docs" / "fit_rule_coefficients_conclusions.md"
    if tail.exists():
        L.append(tail.read_text(encoding="utf-8"))
    return "\n".join(L)


def main() -> int:
    t0 = time.perf_counter()
    train = Period(x.KASE3, x.START_C, x.END_C, x.WINDOWS)
    grid_d, grid_h = fit(train)
    bd, bh = grid_d.loc[grid_d["sharpe"].idxmax()], grid_h.loc[grid_h["sharpe"].idxmax()]
    fitted = replace(config.RULES, rec_penalty=float(bd["rec_penalty"]), hedge_base=float(bh["hedge_base"]),
                     hedge_slope=float(bh["hedge_slope"]))
    print(f"fitted on 2008-2019: rec_penalty {fitted.rec_penalty}, hedge_base {fitted.hedge_base}, hedge_slope {fitted.hedge_slope}")
    test = Period(e2e.KASE7, x.START_B, x.END_B, x.WINDOWS_B)
    table, tests = compare(test, fitted)
    train_tests = []
    for cfg, name in ((x.MACRO, "rec_penalty"), (x.CUR, "hedge_base, hedge_slope")):
        rf_, _ = train.run(cfg, fitted)
        re_, _ = train.run(cfg, config.RULES)
        sh = evaluation.paired_difference_test(rf_ - train.rf.reindex(rf_.index), re_ - train.rf.reindex(re_.index), x.sharpe_fn)
        train_tests.append({"coefficients": name, "sharpe_diff": sh["diff"], "sharpe_low": sh["low"],
                            "sharpe_high": sh["high"], "sharpe_p_two_sided": sh["p_two_sided"]})
    train_tests = pd.DataFrame(train_tests)
    train_tests.to_csv(RES / "fit_rules_train_tests.csv", index=False)
    grid_d.to_csv(RES / "fit_rules_grid_d.csv", index=False)
    grid_h.to_csv(RES / "fit_rules_grid_h.csv", index=False)
    table.to_csv(RES / "fit_rules_test.csv")
    tests.to_csv(RES / "fit_rules_tests.csv", index=False)
    (ROOT / "docs" / "fit_rule_coefficients.md").write_text(markdown(grid_d, grid_h, fitted, table, tests, train_tests),
                                                              encoding="utf-8")
    print(table.round(3).to_string())
    print(tests.round(3).to_string())
    print(train_tests.round(3).to_string())
    print(f"done ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
