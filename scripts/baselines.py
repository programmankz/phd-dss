"""Baseline portfolio rules in the same point-in-time backtest as the DSS (Table 4).

Every baseline replaces only the weights of the risky part; costs, accounting, the
defensive part and the rebalancing dates are those of scripts/end_to_end.py. Two
families: (a) d = d0, h = 0 (the baselines alone, "60/40" with the tenge deposit) and
(b) d and h from the integration rules I-II, i.e. the full DSS with Module III-IV
replaced by the baseline. Sharpe differences to the reference by the paired block
bootstrap with the Holm adjustment over all comparisons; deflated Sharpe ratio with
the number of trials = configurations of this run + the 16 of the ablation.

Samples: 2023-2026 (reference "Full DSS") and 2020-2026 without Black-Litterman and
Module IV (no ranks before 2023; reference "EW + KZT").

Outputs: docs/results/baselines_kase.csv, docs/baselines.md.

Usage::

    python scripts/baselines.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
from dss import baselines, significance  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

ABLATION_TRIALS = 16


def configs(with_signals: bool) -> list[e2e.Config]:
    rules = [r for r in baselines.BASELINES if with_signals or r != "black_litterman"]
    out = [e2e.Config("EW 100%", False, False, False, 1.0)]
    for r in rules:
        name = "EW + KZT" if r == "1/N" else f"{baselines.LABELS[r]} + KZT"
        out.append(e2e.Config(name, False, False, False, weights=r))
    for r in rules:
        out.append(e2e.Config(f"{baselines.LABELS[r]}, правила I–II", True, True, False, weights=r))
    if with_signals:
        out.append(e2e.Config("Full DSS", True, True, True))
    return out


def run(inp, start, end, with_signals, profile, tc, log=print):
    engine = e2e.Engine(inp, profile, start, end, need_m4=with_signals, log=log)
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    daily, rows = {}, {}
    for cfg in configs(with_signals):
        t0 = time.perf_counter()
        r, lg = e2e.run_config(engine, cfg, rets, tc)
        daily[cfg.name] = r
        rows[cfg.name] = e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)
        log(f"  {cfg.name}: Sharpe {rows[cfg.name]['sharpe']:.2f} ({time.perf_counter() - t0:.1f} s)")
    daily = pd.DataFrame(daily)
    excess = daily.sub(rf.reindex(daily.index), axis=0)
    ref = "Full DSS" if with_signals else "EW + KZT"
    tests = significance.sharpe_tests(excess, 0.0, reference=ref)
    table = pd.DataFrame(rows).T
    for c in ("ci_low", "ci_high", "diff", "diff_low", "diff_high", "p_value", "p_holm"):
        table[c] = tests[c]
    n_trials = len(daily.columns) + ABLATION_TRIALS
    sr_daily = (excess.mean() / excess.std(ddof=1)).to_numpy()
    for c in daily.columns:
        table.loc[c, "dsr"], table.loc[c, "sr0"] = significance.deflated_sharpe(excess[c], n_trials, sr_daily)
    table.attrs = {"reference": ref, "n_days": len(daily), "block": tests.attrs["block"], "n_trials": n_trials,
                   "start": str(daily.index.min().date()), "end": str(daily.index.max().date())}
    return table


def markdown(results: dict, profile: str, tc: float) -> str:
    out = ["# Базовые стратегии в общем бэктесте KASE", "",
           "Результаты `python scripts/baselines.py`; таблица `docs/results/baselines_kase.csv`. Каждая базовая "
           "стратегия заменяет только веса рисковой части: издержки, учёт, защитная часть, даты ребалансировки и "
           f"окно ковариации (252 дня, Ledoit–Wolf) — те же, что в таблице 4. Профиль {profile}, издержки "
           f"{num(tc, 0)} б.п., лимит на актив профиля применяется ко всем правилам. «+ KZT» — d = d0, h = 0; "
           "«правила I–II» — d и h по правилам интеграции, как в полной СППР. Константы Блэка–Литтермана "
           "(δ = 2,5, τ = 0,05, IC = 0,05) взяты из учебных значений и не подбирались; взгляды — по рангам "
           "Модуля IV (правило Гринолда α = IC·σ·z).", "",
           "DSR — дефлированный коэффициент Шарпа (Bailey, López de Prado, 2014): вероятность, что истинный Шарп "
           "выше максимума, ожидаемого среди N неэффективных стратегий; N = число конфигураций прогона + 16 "
           "конфигураций абляции. SR0 — этот порог (годовой).", ""]
    for key, t in results.items():
        a = t.attrs
        out += [f"## {key}", "",
                f"Период {a['start']} – {a['end']}, {a['n_days']} торговых дней, блок {a['block']} дн., "
                f"N = {a['n_trials']}, SR0 = {num(t['sr0'].iloc[0])}. Разности Шарпа — к «{a['reference']}».", "",
                "| Стратегия | Доходность, % год. | Волатильность, % | Шарп [95% ДИ] | Макс. просадка, % | Оборот, раз/год | "
                "Разность Шарпа [95% ДИ] | p | p (Холм) | DSR |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in t.iterrows():
            diff = (f"{num(r['diff'], sign=True)} [{num(r['diff_low'])}; {num(r['diff_high'])}]"
                    if np.isfinite(r["diff"]) else "—")
            out.append(f"| {name} | {pct(r['annual_return'])} | {pct(r['volatility'])} | {num(r['sharpe'])} "
                       f"[{num(r['ci_low'])}; {num(r['ci_high'])}] | {pct(r['max_drawdown'])} | "
                       f"{num(r['turnover_per_year'], 1)} | {diff} | {num(r['p_value'], 3)} | {num(r['p_holm'], 3)} | "
                       f"{num(r['dsr'], 2)} |")
        best = t["sharpe"].idxmax()
        sig = t[(t["p_holm"] < 0.05)].index.tolist()
        out += ["", f"Наибольший Шарп — «{best}». "
                + (f"После поправки Холма значимо отличаются от «{a['reference']}»: {', '.join(f'«{s}»' for s in sig)}."
                   if sig else f"После поправки Холма ни одна стратегия значимо не отличается от «{a['reference']}»."),
                ""]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", default=",".join(e2e.KASE7))
    p.add_argument("--end", default="2026-10-09")
    p.add_argument("--profile", default="moderate")
    p.add_argument("--tc", type=float, default=10.0)
    p.add_argument("--out", default=str(ROOT / "docs" / "results" / "baselines_kase.csv"))
    p.add_argument("--md", default=str(ROOT / "docs" / "baselines.md"))
    a = p.parse_args(argv)
    inp = e2e.load_inputs(a.tickers.split(","))
    results = {}
    for key, start, sig in (("2023–2026, с Модулем IV и Блэком–Литтерманом", "2023-01-16", True),
                            ("2020–2026, без сигналов", "2020-01-06", False)):
        print(key)
        results[key] = run(inp, start, a.end, sig, a.profile, a.tc)
    pd.concat(results, names=["sample", "strategy"]).to_csv(a.out)
    Path(a.md).write_text(markdown(results, a.profile, a.tc), encoding="utf-8")
    print(f"written {a.out}, {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
