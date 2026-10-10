"""Command-line interface.

    python -m dss.cli fetch gdp|fx|nse     download real data into ./data
    python -m dss.cli warmup               compute and cache all modules
    python -m dss.cli decide [options]     print the recommendation
    python -m dss.cli report -o out.html   write the HTML report
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
                           tc_bps=a.tc, cov_method=a.cov, use_fuzzy_caps=a.fuzzy)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dss", description="СППР управления инвестиционным портфелем")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="загрузить реальные данные")
    f.add_argument("what", choices=["gdp", "fx", "nse"])
    f.add_argument("--start", default=None)
    for name in ("warmup", "decide", "report"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="moderate", choices=list(config.PROFILES))
        s.add_argument("--scenario", default="base", choices=list(config.SCENARIOS))
        s.add_argument("--universe", default="kase")
        s.add_argument("--models", default="ridge,rf,gb")
        s.add_argument("--rebalance", default="W", choices=["D", "W", "M"])
        s.add_argument("--tc", type=float, default=10.0)
        s.add_argument("--cov", default="ledoit_wolf", choices=["ledoit_wolf", "sample"])
        s.add_argument("--fuzzy", action="store_true")
        if name == "report":
            s.add_argument("-o", "--output", default="dss_report.html")
    sub.add_parser("serve")
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        out = loaders.data_dir()
        out.mkdir(parents=True, exist_ok=True)
        if a.what == "gdp":
            df = loaders.fetch_imf_gdp()
            df.to_csv(out / "gdp_growth.csv")
            print(f"gdp_growth.csv: {df.shape[0]} стран, {df.columns.min()}–{df.columns.max()}")
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
    else:
        from .report import build_report
        with open(a.output, "w", encoding="utf-8") as fh:
            fh.write(build_report(r))
        print(f"Отчёт: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
