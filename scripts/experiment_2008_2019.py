"""Out-of-sample check on 2008-2019, as preregistered in docs/preregistration_2008_2019.md.

The period was not used when the rules and the November 2026 experiment were designed. Risky part:
equal weights of the three stocks with a price history from January 2008 (HSBK, CCBN, KZTK), so the
check concerns the macro rule, the expert currency rule and the conditional currency rule only.
Writes docs/results/exp_2008_2019_*.csv and docs/results_2008_2019.md.

Usage::

    python scripts/experiment_2008_2019.py
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

import end_to_end as e2e  # noqa: E402
from dss import evaluation, significance  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

KASE3 = ["HSBK", "CCBN", "KZTK"]
KASE5 = ["HSBK", "CCBN", "KZTK", "KCEL", "KZTO"]
START_C, END_C = "2008-01-07", "2019-12-31"
START_D = "2013-01-07"                      # robustness: five stocks
START_B, END_B = "2020-01-06", "2026-10-09"  # data already seen: descriptive only
TC = 10.0
RES = ROOT / "docs" / "results"
# devaluation date - 10 trading days ... + 30 trading days (dates fixed in the preregistration)
WINDOWS = {"2009 (04.02.2009)": ("2009-01-21", "2009-03-18"), "2014 (11.02.2014)": ("2014-01-28", "2014-03-25"),
           "2015 (20.08.2015)": ("2015-08-06", "2015-10-01")}
WINDOWS_B = {"2020": ("2020-02-20", "2020-03-31"), "2022": ("2022-02-15", "2022-03-31")}

EW100 = e2e.Config("EW 100%", False, False, False, 1.0)
EW = e2e.Config("EW + KZT", False, False, False)
MACRO = e2e.Config("Macro rule only", True, False, False)
CUR = e2e.Config("Currency rule only", False, True, False)
COND = e2e.Config("Conditional currency rule", False, True, False, fx_rule="conditional")
MC = e2e.Config("Macro and currency rules", True, True, False)
MCOND = e2e.Config("Macro and conditional currency rules", True, True, False, fx_rule="conditional")
CONFIGS = (EW100, EW, MACRO, CUR, COND, MC, MCOND)


def mdd_fn(r):
    return evaluation.max_drawdown_metric(np.asarray(r))


def sharpe_fn(r):
    return significance.sharpe(np.asarray(r), 0.0)


def window_return(r: pd.Series, a: str, b: str) -> float:
    seg = r.loc[a:b]
    return float((1 + seg).prod() - 1) if len(seg) else float("nan")


def run_period(tickers, start, end, windows, peer_window_days=None, configs=CONFIGS):
    """Daily returns, metric table and decision logs of the configurations on one universe and period."""
    inp = e2e.load_inputs(tickers)
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    eng = e2e.Engine(inp, "moderate", start, end, need_m4=False, log=lambda *a: None)
    if peer_window_days:
        eng.peer_window_days = peer_window_days
    daily, rows, logs = {}, {}, {}
    for cfg in configs:
        r, lg = e2e.run_config(eng, cfg, rets, TC)
        m = e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)
        m = {k: v for k, v in m.items() if not k.startswith("return ")}
        for name, (a, b) in windows.items():
            m[f"window {name}"] = window_return(r, a, b)
        if "m" in lg:
            m["share_condition_on"] = float((lg["h"] > 0).mean())
        daily[cfg.name], rows[cfg.name], logs[cfg.name] = r, m, lg
    return pd.DataFrame(daily), pd.DataFrame(rows).T, logs, rf


def hypotheses(daily: pd.DataFrame, table: pd.DataFrame, rf: pd.Series, windows) -> dict:
    ex = daily.sub(rf.reindex(daily.index), axis=0)
    h5 = evaluation.paired_difference_test(daily[MACRO.name], daily[EW.name], mdd_fn)
    h7 = evaluation.paired_difference_test(ex[COND.name], ex[CUR.name], sharpe_fn)
    p = {"H5": h5["p_greater"], "H7": h7["p_greater"]}
    adj = dict(zip(p, significance.holm(list(p.values()))))
    h6 = {w: float(table.loc[CUR.name, f"window {w}"] - table.loc[EW.name, f"window {w}"]) for w in windows}
    h8 = {w: float(table.loc[COND.name, f"window {w}"] - table.loc[EW.name, f"window {w}"]) for w in windows}
    return {"H5": h5 | {"p": p["H5"], "p_holm": adj["H5"], "confirmed": bool(h5["diff"] > 0 and adj["H5"] < 0.05)},
            "H6": {f"diff {w}": v for w, v in h6.items()} | {"confirmed": bool(all(v > 0 for v in h6.values()))},
            "H7": h7 | {"p": p["H7"], "p_holm": adj["H7"], "confirmed": bool(h7["diff"] > 0 and adj["H7"] < 0.05)},
            "H8": {f"diff {w}": v for w, v in h8.items()} | {"confirmed": bool(all(v > 0 for v in h8.values()))}}


def with_tests(daily: pd.DataFrame, table: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    t = significance.sharpe_tests(daily.drop(columns=[EW100.name]).sub(rf.reindex(daily.index), axis=0), 0.0,
                                  reference=EW.name)
    out = table.copy()
    for c in ("diff", "diff_low", "diff_high", "p_value", "p_holm"):
        out[c] = t[c]
    return out


def main() -> int:
    t0 = time.perf_counter()
    print("Period C: 2008-2019, three stocks (primary)")
    dC, tC, logs, rf = run_period(KASE3, START_C, END_C, WINDOWS)
    tests = hypotheses(dC, tC, rf, WINDOWS)
    tC = with_tests(dC, tC, rf)
    print("Secondary: window of the peer gap 91 and 365 days")
    sens = {}
    for wd in (91, 365):
        d, t, _, _ = run_period(KASE3, START_C, END_C, WINDOWS, wd, (EW, CUR, COND))
        sens[wd] = with_tests(pd.concat([d, dC[[EW100.name]]], axis=1), t, rf).loc[[COND.name]]
    print("Secondary: five stocks, 2013-2019")
    win_d = {k: v for k, v in WINDOWS.items() if not k.startswith("2009")}
    dD, tD, _, rfD = run_period(KASE5, START_D, END_C, win_d)
    tD = with_tests(dD, tD, rfD)
    print("Descriptive: 2020-2026, seven stocks (data already seen)")
    dB, tB, _, rfB = run_period(e2e.KASE7, START_B, END_B, WINDOWS_B)
    tB = with_tests(dB, tB, rfB)

    dC.to_csv(RES / "exp_2008_2019_daily.csv")
    tC.to_csv(RES / "exp_2008_2019_table.csv")
    logs[MCOND.name].to_csv(RES / "exp_2008_2019_decisions.csv", index=False)
    pd.DataFrame([{"hypothesis": h, **v} for h, v in tests.items()]).to_csv(RES / "exp_2008_2019_tests.csv", index=False)
    pd.concat({"C, window 91 d": sens[91], "C, window 365 d": sens[365], "D: 2013-2019, 5 stocks": tD,
               "B: 2020-2026, 7 stocks": tB}, names=["sample", "configuration"]).to_csv(RES / "exp_2008_2019_secondary.csv")
    dB.to_csv(RES / "exp_2008_2019_daily_2020_2026.csv")
    (ROOT / "docs" / "results_2008_2019.md").write_text(markdown(tC, tests, sens, tD, tB, logs, dC), encoding="utf-8")
    print(f"written docs/results_2008_2019.md ({time.perf_counter() - t0:.0f} s)")
    return 0


# ----------------------------------------------------------------------------- report
RU = {"EW 100%": "100% акций равными долями", "EW + KZT": "«EW + KZT»: 60% акций + тенге", "Macro rule only": "Макроправило",
      "Currency rule only": "Экспертное валютное правило", "Conditional currency rule": "Условное валютное правило",
      "Macro and currency rules": "Макро- и экспертное валютное правила",
      "Macro and conditional currency rules": "Макро- и условное валютное правила"}


def pp(v) -> str:
    return num(v * 100, 1, sign=True) + " п.п." if np.isfinite(v) else "—"


def yes(b) -> str:
    return "**подтверждена**" if b else "**не подтверждена**"


def table_md(t: pd.DataFrame, windows) -> list:
    head = ("| Конфигурация | Доходность, % год. | Волатильность, % | Шарп | Разность Шарпа к «EW + KZT» [95% ДИ] | "
            "p (Холм) | Макс. просадка, % | Средн. d, % | Средн. h, % | " + " | ".join(f"Окно {w}, %" for w in windows) + " |")
    out = [head, "|" + "---|" * (9 + len(windows))]
    for name, r in t.iterrows():
        diff = (f"{num(r['diff'], sign=True)} [{num(r['diff_low'])}; {num(r['diff_high'])}]"
                if np.isfinite(r.get("diff", np.nan)) else "—")
        ph = num(r["p_holm"], 3) if np.isfinite(r.get("p_holm", np.nan)) else "—"
        out.append(f"| {RU.get(name, name)} | {pct(r['annual_return'])} | {pct(r['volatility'])} | {num(r['sharpe'])} | {diff} | "
                   f"{ph} | {pct(r['max_drawdown'])} | {pct(r['mean_d'], 0)} | {pct(r['mean_h'], 0)} | "
                   + " | ".join(pct(r[f"window {w}"], 2) for w in windows) + " |")
    return out


def markdown(tC, tests, sens, tD, tB, logs, dC) -> str:
    h5, h6, h7, h8 = (tests[k] for k in ("H5", "H6", "H7", "H8"))
    lg = logs[COND.name]
    on = lg[lg["h"] > 0]
    spans = []
    if len(on):
        d = pd.to_datetime(lg["date"])
        flag = (lg["h"] > 0).to_numpy()
        start = None
        for i, f in enumerate(flag):
            if f and start is None:
                start = d.iloc[i]
            if (not f or i == len(flag) - 1) and start is not None:
                end = d.iloc[i] if f else d.iloc[i - 1]
                spans.append(f"{start:%d.%m.%Y}–{end:%d.%m.%Y}")
                start = None
    out = ["# Проверка на 2008–2019 гг.: макроправило и валютные правила", "",
           "План — `docs/preregistration_2008_2019.md` (зафиксирован коммитом до запуска). Результаты "
           "`python scripts/experiment_2008_2019.py`; таблицы `docs/results/exp_2008_2019_*.csv`. Умеренный профиль, "
           f"рисковая часть — равные доли HSBK, CCBN, KZTK, издержки 10 б.п., цены с дивидендами, учёт в тенге. "
           f"Период C: {dC.index[0]:%d.%m.%Y} – {dC.index[-1]:%d.%m.%Y}, {len(dC)} торговых дней.", "",
           "## Основные гипотезы", "",
           "| Гипотеза | Оценка [95% ДИ] | p (одностор.) | p (Холм) | Итог |", "|---|---|---|---|---|",
           f"| H5: макроправило − «EW + KZT», макс. просадка | {pp(h5['diff'])} [{pp(h5['low'])}; {pp(h5['high'])}] | "
           f"{num(h5['p'], 3)} | {num(h5['p_holm'], 3)} | {yes(h5['confirmed'])} |",
           "| H6: экспертное валютное правило − «EW + KZT», доходность в окнах девальваций | "
           + "; ".join(f"{w.split(' ')[0]}: {pct(h6[f'diff {w}'], 2, sign=True)}" for w in WINDOWS) + f" | — | — | {yes(h6['confirmed'])} |",
           f"| H7: условное − экспертное валютное правило, Шарп | {num(h7['diff'], sign=True)} [{num(h7['low'])}; "
           f"{num(h7['high'])}] | {num(h7['p'], 3)} | {num(h7['p_holm'], 3)} | {yes(h7['confirmed'])} |",
           "| H8: условное валютное правило − «EW + KZT», доходность в окнах девальваций | "
           + "; ".join(f"{w.split(' ')[0]}: {pct(h8[f'diff {w}'], 2, sign=True)}" for w in WINDOWS) + f" | — | — | {yes(h8['confirmed'])} |",
           "", "Положительная разность просадок в H5 означает менее глубокую просадку. Разности в H6 и H8 — в процентных "
           "пунктах доходности за окно.", "", "## Конфигурации, период C (2008–2019)", ""]
    out += table_md(tC, WINDOWS)
    out += ["", f"Условие условного правила (r_USD + E[δ] − r_KZT > 0) выполнялось на {pct(len(on) / len(lg), 0)} недельных дат "
            f"({len(on)} из {len(lg)}): " + ("; ".join(spans) if spans else "ни разу") + ".", "",
            "## Вторичные сравнения (разведочные)", "", "### Окно сравнения с сырьевыми валютами: 91 и 365 дней, период C", ""]
    for wd, t in sens.items():
        r = t.iloc[0]
        out.append(f"* {wd} дней: Шарп условного правила {num(r['sharpe'])}, макс. просадка {pct(r['max_drawdown'])}, средняя h "
                   f"{pct(r['mean_h'], 0)}, окна девальваций: " + "; ".join(pct(r[f'window {w}'], 2) for w in WINDOWS) + ".")
    win_d = [w for w in WINDOWS if not w.startswith("2009")]
    out += ["", "### Пять бумаг (HSBK, CCBN, KZTK, KCEL, KZTO), 2013–2019", ""] + table_md(tD, win_d)
    out += ["", "### 2020–2026, семь бумаг (данные уже использовались: только описание)", ""] + table_md(tB, WINDOWS_B)
    ew, cur, cond, mac = (tC.loc[c.name] for c in (EW, CUR, COND, MACRO))
    out += ["", "## Выводы", "",
            f"* **H5** {yes(h5['confirmed'])}: макроправило меняет максимальную просадку 2008–2019 гг. на {pp(h5['diff'])} "
            f"(с {pct(ew['max_drawdown'])} до {pct(mac['max_drawdown'])}) при средней доле акций {pct(mac['mean_d'], 0)}; "
            f"односторонний p = {num(h5['p'], 3)}, с поправкой Холма {num(h5['p_holm'], 3)}. Результат H2 на 2020–2026 гг. "
            "на независимом периоде " + ("повторился." if h5["confirmed"] else "статистически не повторился: направление то же, "
            "но эффект мал относительно просадки рынка в кризис 2008–2009 гг."),
            f"* **H6** {yes(h6['confirmed'])}: экспертное валютное правило в окнах трёх девальваций дало к «EW + KZT» "
            + ", ".join(pct(h6[f"diff {w}"], 2, sign=True) for w in WINDOWS) + ". За весь период его Шарп отличается от «EW + KZT» на "
            f"{num(cur['diff'], sign=True)} [{num(cur['diff_low'])}; {num(cur['diff_high'])}]: в периоде с тремя девальвациями "
            "валютная защита доходности не стоила.",
            f"* **H7** {yes(h7['confirmed'])}: Шарп условного правила отличается от экспертного на {num(h7['diff'], sign=True)} "
            f"[{num(h7['low'])}; {num(h7['high'])}] при средней доле доллара {pct(cond['mean_h'], 0)} против {pct(cur['mean_h'], 0)}.",
            f"* **H8** {yes(h8['confirmed'])}: условное правило в окнах девальваций дало к «EW + KZT» "
            + ", ".join(pct(h8[f"diff {w}"], 2, sign=True) for w in WINDOWS) + ". Окно, в котором условие не было выполнено, "
            "правило пропускает полностью.",
            "* Период 2008–2019 гг. содержит падение рынка акций KASE на " + pct(-tC.loc[EW100.name, "max_drawdown"], 0) +
            " (три бумаги, равные доли); абсолютные коэффициенты Шарпа всех конфигураций близки к нулю. Проверка касается только "
            "правил долей d и h: модули III и IV в ней не участвуют.",
            "", "## Отступления от плана", "",
            "* При первом запуске скрипт остановился до получения результатов: оценочные таблицы Модуля I не считались на "
            "панелях ВВП, заканчивающихся раньше 2015 г. (пустая тестовая выборка). Исправлено отдельным коммитом до повторного "
            "запуска; вероятность рецессии и прогноз роста, которые входят в правило, от этого разбиения не зависят. Гипотезы, "
            "периоды, окна, параметры и тесты — как в плане.",
            f"* Первая дневная доходность периода C приходится на {dC.index[0]:%d.%m.%Y} (первое решение 07.01.2008 или "
            "ближайший торговый день).", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
