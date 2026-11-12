"""Full factorial ablation of the DSS modules (2^4 = 16 configurations) and Shapley values.

Modules: I - macro rule of the risky share d; II - currency rule of the dollar share h;
III - cluster limits (with IV: cluster caps of the optimiser; without IV: equal weights per
cluster); IV - weights of the risky part from the signals. v(empty set) = "EW + KZT".
Same point-in-time engine, costs and accounting as scripts/end_to_end.py (Table 4).

Two samples:
* 2023-2026 (Table 4 period): all 16 configurations, Shapley over I-IV;
* 2020-2026 (Table 5 period): Module IV ranks are not available before 2023, so only the
  8 configurations of I-III; this sample contains the stress windows of 2020 and 2022.

Outputs: docs/results/abl_configs.csv, abl_shapley.csv, docs/ablation.md.

Usage::

    python scripts/ablation.py
    python scripts/ablation.py --end 2026-10-09 --n-boot 2000
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
from dss import evaluation  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

LABELS = {"I": "I — макроправило d", "II": "II — валютное правило h", "III": "III — кластерные лимиты",
          "IV": "IV — сигналы"}
METRIC_RU = {"sharpe": "Шарп", "max_drawdown": "Макс. просадка", "return 20.02-31.03.2020": "Доходность 20.02–31.03.2020",
             "return 15.02-31.03.2022": "Доходность 15.02–31.03.2022"}


def config_of(s: frozenset) -> e2e.Config:
    return e2e.Config(evaluation.coalition_name(s), macro="I" in s, fx="II" in s, m4="IV" in s, clusters="III" in s)


def run_sample(inp, start, end, players, profile, tc, n_boot, log=print):
    engine = e2e.Engine(inp, profile, start, end, need_m4="IV" in players, log=log)
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    daily, rows = {}, {}
    for s in evaluation.coalitions(players):
        cfg = config_of(s)
        t0 = time.perf_counter()
        r, lg = e2e.run_config(engine, cfg, rets, tc)
        daily[cfg.name] = r
        rows[cfg.name] = {"modules": cfg.name, **e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)}
        log(f"  {cfg.name:>14}: Sharpe {rows[cfg.name]['sharpe']:.2f}, MDD {rows[cfg.name]['max_drawdown']:.1%} "
            f"({time.perf_counter() - t0:.1f} s)")
    daily = pd.DataFrame(daily)
    shap = evaluation.shapley_table(daily, rf, e2e.SHOCKS, players, n_boot=n_boot)
    table = pd.DataFrame(rows).T
    meta = {"start": str(daily.index.min().date()), "end": str(daily.index.max().date()), "n_days": len(daily),
            "block": shap.attrs["block"], "n_boot": n_boot}
    return table, shap, meta


def markdown(results: dict, profile: str, tc: float) -> str:
    out = ["# Вклад модулей СППР: полный факторный эксперимент и значения Шепли", "",
           "Результаты `python scripts/ablation.py`; таблицы `docs/results/abl_configs.csv` и `abl_shapley.csv`. "
           f"Профиль {profile}, издержки {num(tc, 0)} б.п., та же point-in-time схема, что и в таблице 4 "
           "(`scripts/end_to_end.py`).", "",
           "## Схема", "",
           "Каждый модуль — «игрок»: I — макроправило доли рисковой части d (иначе d = d0), II — валютное правило "
           "доли доллара h (иначе h = 0), III — кластерные лимиты (вместе с IV — лимиты оптимизатора, без IV — "
           "равные веса по кластерам с теми же лимитами), IV — веса рисковой части по сигналам (иначе равные). "
           "v(∅) — портфель «EW + KZT». Значение Шепли модуля i — средний предельный вклад по всем порядкам "
           "включения модулей; сумма значений равна v(все) − v(∅). Интервалы — перцентильные 95% по "
           "циркулярному блочному бутстрепу дневных доходностей (одни и те же индексы для всех конфигураций). "
           "Для доходности в окне стресса интервал не строится: перестановка дней разрушает окно.", ""]
    for key, (table, shap, meta, players) in results.items():
        out += [f"## {key}", "",
                f"Период {meta['start']} – {meta['end']}, {meta['n_days']} торговых дней, блок бутстрепа "
                f"{meta['block']} дн., {meta['n_boot']} выборок; модули {', '.join(players)}.", "",
                "### Конфигурации", "",
                "| Модули | Доходность, % год. | Волатильность, % | Шарп | Макс. просадка, % | Оборот, раз/год | Средн. d, % | Средн. h, % |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in table.iterrows():
            out.append(f"| {r['modules']} | {pct(r['annual_return'])} | {pct(r['volatility'])} | {num(r['sharpe'])} | "
                       f"{pct(r['max_drawdown'])} | {num(r['turnover_per_year'], 1)} | {pct(r['mean_d'], 0)} | "
                       f"{pct(r['mean_h'], 0)} |")
        out += ["", "### Значения Шепли", "",
                "| Показатель | Модуль | φ | 95% ДИ |", "|---|---|---|---|"]
        for _, r in shap.iterrows():
            if r["module"] == "efficiency gap":
                continue
            pct_metric = r["metric"] != "sharpe"
            f = (lambda v: pct(v, 2, sign=True)) if pct_metric else (lambda v: num(v, 3, sign=True))
            ci = f"[{f(r['ci_low'])}; {f(r['ci_high'])}]" if np.isfinite(r["ci_low"]) else "—"
            mod = {"sum": "Σφ", "v(all)-v(none)": "v(все) − v(∅)"}.get(r["module"], LABELS.get(r["module"], r["module"]))
            out.append(f"| {METRIC_RU.get(r['metric'], r['metric'])} | {mod} | {f(r['phi'])} | {ci} |")
        gap = shap.loc[shap["module"] == "efficiency gap", "phi"].abs().max()
        out += ["", f"Проверка эффективности: max |Σφ − (v(все) − v(∅))| = {gap:.1e}.".replace(".", ",", 1), ""]
        sig = shap[shap["module"].isin(players) & ((shap["ci_low"] > 0) | (shap["ci_high"] < 0))]
        if len(sig):
            items = [f"{LABELS[r['module']]} — {METRIC_RU.get(r['metric'], r['metric'])} "
                     f"({'вклад положителен' if r['ci_low'] > 0 else 'вклад отрицателен'})" for _, r in sig.iterrows()]
            out += ["95% интервал не содержит нуля: " + "; ".join(items) + ". Для остальных модулей и показателей "
                    "вклад статистически не отличим от нуля.", ""]
        else:
            out += ["Ни один 95% интервал не исключает нуля: вклад модулей статистически не отличим от нуля.", ""]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", default=",".join(e2e.KASE7))
    p.add_argument("--start", default="2023-01-16", help="start of the 16-configuration sample")
    p.add_argument("--start-long", default="2020-01-06", help="start of the 8-configuration sample (I-III)")
    p.add_argument("--end", default="2026-10-09")
    p.add_argument("--profile", default="moderate")
    p.add_argument("--tc", type=float, default=10.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=str(ROOT / "docs" / "results"))
    p.add_argument("--md", default=str(ROOT / "docs" / "ablation.md"))
    a = p.parse_args(argv)
    inp = e2e.load_inputs(a.tickers.split(","))
    results = {}
    for key, start, players in (("2023–2026: модули I–IV, 16 конфигураций", a.start, evaluation.MODULES),
                                ("2020–2026: модули I–III, 8 конфигураций", a.start_long, ("I", "II", "III"))):
        print(key)
        table, shap, meta = run_sample(inp, start, a.end, players, a.profile, a.tc, a.n_boot)
        results[key] = (table, shap, meta, players)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat({k: v[0] for k, v in results.items()}, names=["sample", "config"]).to_csv(out / "abl_configs.csv")
    pd.concat({k: v[1] for k, v in results.items()}, names=["sample", "row"]).to_csv(out / "abl_shapley.csv")
    Path(a.md).write_text(markdown(results, a.profile, a.tc), encoding="utf-8")
    print(f"written {out / 'abl_configs.csv'}, {out / 'abl_shapley.csv'}, {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
