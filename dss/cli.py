"""Command-line interface.

    python -m dss.cli fetch gdp|fx|nse     download real data into ./data
    python -m dss.cli warmup               compute and cache all modules
    python -m dss.cli decide [options]     print the recommendation
    python -m dss.cli report -o out.html   write the HTML report
    python -m dss.cli sensitivity          sensitivity of the integration rules
    python -m dss.cli significance         bootstrap Sharpe CIs and Diebold-Mariano tests
    python -m dss.cli serve                start the web interface
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config, pipeline
from .data import loaders


def _settings(a) -> config.Settings:
    return config.Settings(profile=a.profile, scenario=a.scenario, universe=a.universe,
                           models=tuple(a.models.split(",")), rebalance=a.rebalance,
                           tc_bps=a.tc, cov_method=a.cov, use_fuzzy_caps=a.fuzzy,
                           protocol=a.protocol, backtest_caps=a.caps)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dss", description="СППР управления инвестиционным портфелем")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="загрузить реальные данные")
    f.add_argument("what", choices=["gdp", "fx", "nse"])
    f.add_argument("--start", default=None)
    for name in ("warmup", "decide", "report", "sensitivity", "significance"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="moderate", choices=list(config.PROFILES))
        s.add_argument("--scenario", default="base", choices=list(config.SCENARIOS))
        s.add_argument("--universe", default="kase")
        s.add_argument("--models", default="ridge,rf,gb")
        s.add_argument("--rebalance", default="W", choices=["D", "W", "M"])
        s.add_argument("--tc", type=float, default=10.0)
        s.add_argument("--cov", default="ledoit_wolf", choices=["ledoit_wolf", "sample"])
        s.add_argument("--fuzzy", action="store_true")
        s.add_argument("--protocol", default="walk_forward", choices=list(config.PROTOCOLS),
                       help="walk_forward или paper (одно разбиение 80/20, как в [A4])")
        s.add_argument("--no-caps", dest="caps", action="store_false",
                       help="бэктест без лимитов профиля и кластеров (0 ≤ w ≤ 1, как в [A4])")
        if name == "report":
            s.add_argument("-o", "--output", default="dss_report.html")
    sub.add_parser("serve")
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        out = loaders.data_dir()
        out.mkdir(parents=True, exist_ok=True)
        if a.what == "gdp":
            df, source = loaders.fetch_gdp()
            df.to_csv(out / "gdp_growth.csv")
            print(f"gdp_growth.csv ({source}): {df.shape[0]} стран, {df.columns.min()}–{df.columns.max()}")
        elif a.what == "fx":
            df = loaders.fetch_nbk(start=a.start or "2000-01-01")
            print(f"fx_rates.csv: {len(df)} дней, {df.index.min():%Y-%m-%d} – {df.index.max():%Y-%m-%d}")
        else:
            df = loaders.fetch_yahoo(loaders.NSE_TICKERS, start=a.start or "2020-01-01")
            df.to_csv(out / "prices_nse.csv")
            print(f"prices_nse.csv: {df.shape}")
        return 0

    if a.cmd == "serve":
        from webapp.app import main as serve
        serve()
        return 0

    r = pipeline.run(_settings(a))
    for e in r.data_errors:
        print("!! Ошибка данных: " + e, file=sys.stderr)
    if a.cmd == "warmup":
        print("Кэш готов.", "Источники:", *(f"  {k}: {v}" for k, v in r.sources.items()), sep="\n")
    elif a.cmd == "decide":
        d = r.decision
        if r.is_demo:
            print("!! ДЕМО-РЕЖИМ: часть данных синтетическая\n")
        for s in d.steps:
            print(f"[{s['module']}] {s['title']}")
            for x in s["facts"]:
                print(f"     - {x}")
            print(f"     => {s['result']}")
        print(f"\nДоля рисковых активов: {d.risky_share:.1%}; USD-хедж: {d.usd_hedge_share:.1%} безрисковой части")
        print(f"Ожидаемая волатильность: {d.exp_vol:.1%}; VaR95 1 мес.: {d.var95_1m:.1%}\n")
        with pd.option_context("display.width", 140):
            print(d.allocation[d.allocation.weight > 0.0005][["weight", "signal", "cluster"]].round(3))
        for w in d.warnings:
            print("! " + w)
    elif a.cmd == "sensitivity":
        sens = pipeline.rule_sensitivity(r)
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 70):
            print("Чувствительность к коэффициентам (выход — доля рисковых активов или USD-хедж):")
            print(sens["coefficients"].to_string(float_format=lambda v: f"{v:.3f}"), "\n")
            print("Монотонность правил:")
            print(sens["monotonicity"].to_string(float_format=lambda v: f"{v:.1f}"), "\n")
            print(f"Доля рисковых активов по P(рецессии) и сценарию (фаза «{r.macro.phase}»):")
            print(sens["risk_grid"].to_string(float_format=lambda v: f"{v:.3f}"))
    elif a.cmd == "significance":
        sharpe, dm = pipeline.significance_tables(r)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(f"Шарп стратегий, блочный бутстреп ({r.sharpe_tests.attrs['n_boot']} повторов, "
                  f"блоки {r.sharpe_tests.attrs['block']} дн.):")
            print(sharpe.to_string(float_format=lambda v: f"{v:.3f}"), "\n")
            print("Диболд–Мариано против наивного прогноза r̂ = 0 (DM < 0 — модель точнее):")
            print(dm.to_string(float_format=lambda v: f"{v:.3f}"))
    else:
        from .report import build_report
        with open(a.output, "w", encoding="utf-8") as fh:
            fh.write(build_report(r))
        print(f"Отчёт: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
