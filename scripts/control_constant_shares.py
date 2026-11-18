"""Exploratory control: do the rules add anything over constant shares with the same average exposure?

For each period the expert currency rule is compared with a constant dollar share equal to the rule's
average dollar share, and the macro rule with a constant equity share equal to the rule's average
equity share. The constants are the averages over the same period (known only after it), so the
control answers one question only: is the gain due to the timing of the rule or to the exposure itself.
Both periods were already used; nothing here is a preregistered test.
Writes docs/results/control_constant_shares.csv and docs/control_constant_shares.md.

Usage::

    python scripts/control_constant_shares.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
import experiment_2008_2019 as x  # noqa: E402
from dss import evaluation  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

CONST_H, CONST_D = "Constant dollar share", "Constant equity share"
PERIODS = {"2008–2019, 3 бумаги": (x.KASE3, x.START_C, x.END_C, x.WINDOWS),
           "2020–2026, 7 бумаг": (e2e.KASE7, x.START_B, x.END_B, x.WINDOWS_B)}
RU = x.RU | {CONST_H: "Постоянная доля доллара", CONST_D: "Постоянная доля акций"}


def run(tickers, start, end, windows):
    d1, t1, _, rf = x.run_period(tickers, start, end, windows, configs=(x.EW, x.MACRO, x.CUR))
    h_bar, d_bar = float(t1.loc[x.CUR.name, "mean_h"]), float(t1.loc[x.MACRO.name, "mean_d"])
    controls = (e2e.Config(CONST_H, False, False, False, h_fixed=h_bar), e2e.Config(CONST_D, False, False, False, d_bar))
    d2, t2, _, _ = x.run_period(tickers, start, end, windows, configs=controls)
    daily, table = pd.concat([d1, d2], axis=1), pd.concat([t1, t2])
    ex = daily.sub(rf.reindex(daily.index), axis=0)
    tests = {"currency rule − constant dollar share, Sharpe":
             evaluation.paired_difference_test(ex[x.CUR.name], ex[CONST_H], x.sharpe_fn),
             "currency rule − constant dollar share, max drawdown":
             evaluation.paired_difference_test(daily[x.CUR.name], daily[CONST_H], x.mdd_fn),
             "macro rule − constant equity share, max drawdown":
             evaluation.paired_difference_test(daily[x.MACRO.name], daily[CONST_D], x.mdd_fn),
             "macro rule − constant equity share, Sharpe":
             evaluation.paired_difference_test(ex[x.MACRO.name], ex[CONST_D], x.sharpe_fn)}
    return table, tests, h_bar, d_bar


def markdown(res) -> str:
    out = ["# Контроль: правила против постоянных долей (разведочный)", "",
           "Вопрос: выигрыш правил получен за счёт момента изменения долей или за счёт самой средней доли? Экспертное "
           "валютное правило сравнивается с постоянной долей доллара, равной средней доле правила за период; макроправило — "
           "с постоянной долей акций, равной средней доле правила. Постоянные доли равны средним за тот же период (известны "
           "только после него), поэтому контроль отвечает только на этот вопрос. Оба периода уже использовались: сравнение "
           "разведочное, заранее не записано, гипотезы им не проверяются. `python scripts/control_constant_shares.py`; "
           "таблица `docs/results/control_constant_shares.csv`. Умеренный профиль, издержки 10 б.п., цены с дивидендами.", ""]
    for period, (table, tests, h_bar, d_bar, windows) in res.items():
        out += [f"## {period}", "",
                f"Средняя доля доллара экспертного правила {pct(h_bar, 0)} защитной части; средняя доля акций макроправила "
                f"{pct(d_bar, 0)}.", "",
                "| Конфигурация | Доходность, % год. | Шарп | Макс. просадка, % | Средн. d, % | Средн. h, % | "
                + " | ".join(f"Окно {w}, %" for w in windows) + " |", "|" + "---|" * (6 + len(windows))]
        for name, r in table.iterrows():
            out.append(f"| {RU.get(name, name)} | {pct(r['annual_return'])} | {num(r['sharpe'])} | {pct(r['max_drawdown'])} | "
                       f"{pct(r['mean_d'], 0)} | {pct(r['mean_h'], 0)} | "
                       + " | ".join(pct(r[f"window {w}"], 2) for w in windows) + " |")
        cur, ch = table.loc[x.CUR.name], table.loc[CONST_H]
        t = tests
        s, m = t["currency rule − constant dollar share, Sharpe"], t["currency rule − constant dollar share, max drawdown"]
        md, ms = t["macro rule − constant equity share, max drawdown"], t["macro rule − constant equity share, Sharpe"]
        out += ["", "| Сравнение | Оценка [95% ДИ] | p (двустор.) |", "|---|---|---|",
                f"| Валютное правило − постоянная доля доллара, Шарп | {num(s['diff'], sign=True)} [{num(s['low'])}; "
                f"{num(s['high'])}] | {num(s['p_two_sided'], 3)} |",
                f"| Валютное правило − постоянная доля доллара, макс. просадка | {x.pp(m['diff'])} [{x.pp(m['low'])}; "
                f"{x.pp(m['high'])}] | {num(m['p_two_sided'], 3)} |",
                "| Валютное правило − постоянная доля доллара, доходность в окнах | "
                + "; ".join(f"{w.split(' ')[0]}: {pct(cur[f'window {w}'] - ch[f'window {w}'], 2, sign=True)}" for w in windows)
                + " | — |",
                f"| Макроправило − постоянная доля акций, макс. просадка | {x.pp(md['diff'])} [{x.pp(md['low'])}; "
                f"{x.pp(md['high'])}] | {num(md['p_two_sided'], 3)} |",
                f"| Макроправило − постоянная доля акций, Шарп | {num(ms['diff'], sign=True)} [{num(ms['low'])}; "
                f"{num(ms['high'])}] | {num(ms['p_two_sided'], 3)} |", "",
                "Положительная разность просадок означает менее глубокую просадку у правила. Разности в окнах — в процентных "
                "пунктах доходности за окно.", ""]
    out += ["## Выводы", ""]
    for period, (table, tests, h_bar, d_bar, windows) in res.items():
        s = tests["currency rule − constant dollar share, Sharpe"]
        md = tests["macro rule − constant equity share, max drawdown"]
        cur, ch = table.loc[x.CUR.name], table.loc[CONST_H]
        out += [f"* **{period}.** Экспертное валютное правило от постоянной доли доллара {pct(h_bar, 0)} не отличается: "
                f"разность Шарпа {num(s['diff'], sign=True)} [{num(s['low'])}; {num(s['high'])}], в окнах "
                + "; ".join(pct(cur[f"window {w}"] - ch[f"window {w}"], 2, sign=True) for w in windows)
                + f". Макроправило от постоянной доли акций {pct(d_bar, 0)} по максимальной просадке отличается на "
                f"{x.pp(md['diff'])} [{x.pp(md['low'])}; {x.pp(md['high'])}], p = {num(md['p_two_sided'], 3)}."]
    out += ["* Выигрыш валютного правила в окнах девальваций даёт сама долларовая позиция, а не изменение её размера по "
            "индексу валютного риска: постоянная доля доллара того же среднего размера даёт тот же результат.",
            "* Статистически значимого преимущества макроправила перед постоянной долей акций того же среднего размера "
            "не найдено ни в одном периоде.", ""]
    return "\n".join(out) + "\n"


def main() -> int:
    t0 = time.perf_counter()
    res, rows = {}, []
    for period, (tickers, start, end, windows) in PERIODS.items():
        print(period)
        table, tests, h_bar, d_bar = run(tickers, start, end, windows)
        res[period] = (table, tests, h_bar, d_bar, windows)
        rows.append(table.assign(period=period, kind="configuration"))
        rows.append(pd.DataFrame(tests).T.assign(period=period, kind="test"))
    pd.concat(rows).to_csv(x.RES / "control_constant_shares.csv")
    (ROOT / "docs" / "control_constant_shares.md").write_text(markdown(res), encoding="utf-8")
    print(f"written docs/control_constant_shares.md ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
