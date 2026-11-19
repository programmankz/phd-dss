"""Command-line interface.

    python -m dss.cli fetch gdp|fx|nse|kase|rates|...  download real data into ./data
    python -m dss.cli warmup               compute and cache all modules
    python -m dss.cli decide [options]     print the recommendation
    python -m dss.cli report -o out.html   write the HTML report
    python -m dss.cli sensitivity          sensitivity of the integration rules
    python -m dss.cli significance         bootstrap Sharpe CIs and Diebold-Mariano tests
    python -m dss.cli export -o t.xlsx     all result tables (.xlsx needs openpyxl, else .zip of CSV)
    python -m dss.cli serve                start the web interface
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config, pipeline
from .numfmt import num, pct
from .data import loaders


def _settings(a) -> config.Settings:
    return config.Settings(profile=a.profile, scenario=a.scenario, universe=a.universe,
                           models=tuple(a.models.split(",")), rebalance=a.rebalance,
                           tc_bps=a.tc, cov_method=a.cov, use_fuzzy_caps=a.fuzzy,
                           protocol=a.protocol, backtest_caps=a.caps, limits_policy=a.limits)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dss", description="СППР управления инвестиционным портфелем")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="загрузить реальные данные")
    f.add_argument("what", choices=["gdp", "fx", "nse", "kase", "rates", "kase-dividends", "kase-extra", "extended",
                                         "kase-index", "bonds", "brent", "macro", "fx-panel", "gdp-quarterly"])
    f.add_argument("--start", default=None)
    f.add_argument("--tickers", default=None, help="kase/nse: тикеры через запятую вместо набора по умолчанию")
    for name in ("warmup", "decide", "report", "sensitivity", "significance", "export"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="moderate", choices=list(config.PROFILES))
        s.add_argument("--scenario", default="base", choices=list(config.SCENARIOS))
        s.add_argument("--universe", default="kase")
        s.add_argument("--models", default="ridge,rf,gb")
        s.add_argument("--rebalance", default="W", choices=["D", "W", "M"])
        s.add_argument("--tc", type=float, default=10.0)
        s.add_argument("--cov", default="ledoit_wolf", choices=["ledoit_wolf", "sample"])
        s.add_argument("--fuzzy", action="store_true")
        s.add_argument("--limits", default="relax", choices=list(config.LIMITS_POLICIES),
                       help="несовместимые лимиты профиля: relax — ослабить, strict — отказ от рисковой части")
        s.add_argument("--protocol", default="walk_forward", choices=list(config.PROTOCOLS),
                       help="walk_forward или paper (одно разбиение 80/20, как в [A4])")
        s.add_argument("--no-caps", dest="caps", action="store_false",
                       help="бэктест без лимитов профиля и кластеров (0 ≤ w ≤ 1, как в [A4])")
        if name == "report":
            s.add_argument("-o", "--output", default="dss_report.html")
        if name == "export":
            s.add_argument("-o", "--output", default=None,
                           help="файл .xlsx (нужен openpyxl) или .zip (CSV); по умолчанию dss_tables.xlsx/.zip")
    sub.add_parser("serve")
    at = sub.add_parser("attract", help="инвестиционная привлекательность: индекс KASE и рост ВВП")
    at.add_argument("--definition", default="sign", choices=["sign", "accel", "panel"],
                    help="sign: рост > 0; accel: ускорение роста; panel: рост > 0, обучение на панели стран")
    at.add_argument("--freq", default=None, choices=["annual", "quarterly"])
    at.add_argument("--window", type=int, default=6, help="окно доходности индекса KASE, мес.")
    ar = sub.add_parser("article", help="эксперимент статьи [A4] на KASE (и на выборке NSE статьи)")
    ar.add_argument("--universe", default="kase", choices=["kase", "nse"])
    ar.add_argument("--set", dest="kase_set", default="nine", choices=["nine", "seven"],
                    help="nine: 9 бумаг с 12.2022; seven: 7 бумаг с 12.2018 (без KSPI и KMGZ)")
    a = p.parse_args(argv)

    if a.cmd == "article":
        from . import replication
        r = replication.run(a.universe, kase_set=a.kase_set)
        i = r.info
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(f"Протокол [A4], {replication.UNIVERSES[r.universe]}: {i['assets']} бумаг, данные "
                  f"{i['data_start']:%d.%m.%Y} – {i['data_end']:%d.%m.%Y}, тест с {i['test_start']:%d.%m.%Y}, {i['bt_days']} дн.")
            for note in r.notes:
                print("Внимание:", note)
            print(r.strategies.to_string(float_format=lambda v: num(v, 3), na_rep="—"), "\n")
            print(r.models.to_string(float_format=lambda v: num(v, 4), na_rep="—"), "\n")
            print(r.sharpe_tests[["sharpe", "diff", "p_value", "p_holm"]].to_string(float_format=lambda v: num(v, 3), na_rep="—"))
        return 0

    if a.cmd == "attract":
        from . import attractiveness
        r = attractiveness.run(a.definition, a.freq, a.window)
        s = r.signal
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(f"{r.label}; данные {attractiveness.FREQS[r.freq]}, {r.data.index[0]} – {r.data.index[-1]}, "
                  f"окно KASE {r.window_months} мес.")
            for note in r.notes:
                print("Внимание:", note)
            print(r.metrics.to_string(float_format=lambda v: num(v, 3), na_rep="—"), "\n")
            print(r.coefficients.to_string(float_format=lambda v: num(v, 4), na_rep="—"), "\n")
            if s["available"]:
                print(f"Показатель на {s['period']}: A = {s['value']}, P(A = 1) = {num(s['probs']['ARIMA + KASE'], 2)}, "
                      f"прогноз роста {num(s['growth']['ARIMA + KASE'], 1)}% (ARIMA без индекса {num(s['growth']['ARIMA'], 1)}%)")
        return 0

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
        elif a.what == "rates":
            nbk = loaders.fetch_nbk_base_rate()
            tb = loaders.fetch_tb3ms()
            print(f"nbk_base_rate.csv: {len(nbk)} изменений ставки, последнее {nbk['effective_from'].max():%Y-%m-%d}; "
                  f"tb3ms.csv: {len(tb)} месяцев, последний {tb['Date'].max()}")
        elif a.what == "brent":
            df = loaders.fetch_brent(start=a.start or "2000-01-01")
            print(f"brent.csv: {len(df)} дней, {df.index.min():%Y-%m-%d} – {df.index.max():%Y-%m-%d}")
        elif a.what == "macro":
            df = loaders.fetch_macro_kz(start=a.start or "2008-01")
            print(f"macro_monthly_kz.csv: {len(df)} месяцев {df.index.min()} – {df.index.max()}; "
                  + ", ".join(f"{c}: {df[c].notna().sum()}" for c in ("kei", "kei_nominal_bn_kzt", "cpi_yoy")))
        elif a.what == "fx-panel":
            df = loaders.fetch_fx_panel(start=a.start or "2000-01-01")
            print("fx_commodity_panel.csv: " + ", ".join(
                f"{c} {df[c].first_valid_index():%Y-%m-%d} – {df[c].last_valid_index():%Y-%m-%d}" for c in df))
        elif a.what == "kase-dividends":
            tickers = a.tickers.split(",") if a.tickers else loaders.KASE_TICKERS
            df = loaders.fetch_kase_dividends(tickers)
            print(f"dividends_kase.csv: {len(df)} выплат по {df['ticker'].nunique()} бумагам, "
                  f"даты фиксации {df['record_date'].min()} – {df['record_date'].max()}")
        elif a.what == "kase-extra":
            tickers = a.tickers.split(",") if a.tickers else loaders.KASE_EXTRA_TICKERS
            h = loaders.fetch_kase_history(tickers, start=a.start or "2008-01-01", fill="none")
            h["close"].to_csv(out / "prices_kase_extra.csv")
            h["volume"].to_csv(out / "volumes_kase_extra.csv")
            print(f"prices_kase_extra.csv: {h['close'].shape[1]} бумаг, {h['close'].index.min():%Y-%m-%d} – "
                  f"{h['close'].index.max():%Y-%m-%d}")
        elif a.what == "extended":
            h = loaders.fetch_kase_history(loaders.KASE_NEW_TICKERS, start=a.start or "2016-01-01", fill="none")
            h["close"].to_csv(out / "prices_kase_new.csv")
            h["volume"].to_csv(out / "volumes_kase_new.csv")
            x = loaders.fetch_aix_history(start=a.start or "2017-01-01")
            x["close"].to_csv(out / "prices_aix.csv")
            x["volume"].to_csv(out / "volumes_aix.csv")
            div = loaders.fetch_kase_dividends(loaders.KASE_NEW_TICKERS, cache=out / "dividends_kase_new.csv")
            print(f"prices_kase_new.csv: {', '.join(h['close'].columns)}; prices_aix.csv: {', '.join(x['close'].columns)} "
                  f"(USD); dividends_kase_new.csv: {len(div)} выплат")
        elif a.what == "gdp-quarterly":
            s = loaders.fetch_gdp_quarterly_kz()
            print(f"gdp_quarterly_kz.csv: {len(s)} кварталов, {s.index.min()} – {s.index.max()}")
        elif a.what == "kase-index":
            df = loaders.fetch_kase_index(start=a.start or "2008-01-01")
            bi = loaders.fetch_kase_bond_indices(start=a.start or "2008-01-01")
            print(f"index_kase.csv: {len(df)} дней, {df.index.min():%Y-%m-%d} – {df.index.max():%Y-%m-%d}; "
                  f"bond_indices_kase.csv: {bi.shape[1]} индексов с {bi.index.min():%Y-%m-%d}")
        elif a.what == "bonds":
            codes = a.tickers.split(",") if a.tickers else None
            px, meta = loaders.fetch_kase_bonds(codes, start=a.start or "2008-01-01")
            print(f"bonds_kase.csv: {px.shape[1]} выпусков, {int(px.notna().sum().sum())} цен; bonds_kase_meta.csv")
        elif a.what == "kase":
            tickers = a.tickers.split(",") if a.tickers else loaders.KASE_TICKERS
            # closes of trading days only (NaN otherwise): dividends are then removed from real
            # trades, and validate_prices forward-fills the adjusted series
            h = loaders.fetch_kase_history(tickers, start=a.start or "2008-01-01", fill="none")
            df = h["close"]
            df.to_csv(out / "prices_kase.csv")
            h["volume"].to_csv(out / "volumes_kase.csv")
            print(f"prices_kase.csv: {df.shape[1]} бумаг, {df.index.min():%Y-%m-%d} – {df.index.max():%Y-%m-%d}; "
                  f"общая история с {df.ffill().dropna().index.min():%Y-%m-%d}")
            fund = out / "fundamentals_kase.csv"
            if not fund.exists():  # sectors only: kase.kz has no P/B, P/E feed; fill them in by hand
                pd.DataFrame({"ticker": df.columns, "sector": [loaders.KASE_SECTORS.get(t, "") for t in df.columns],
                              "PB": None, "PE": None}).to_csv(fund, index=False)
                print("fundamentals_kase.csv: секторы записаны, P/B и P/E пусты — заполните вручную")
        else:
            tickers = a.tickers.split(",") if a.tickers else loaders.NSE_TICKERS
            df = loaders.fetch_yahoo(tickers, start=a.start or "2020-01-01")
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
        print(f"\nДоля рисковых активов: {pct(d.risky_share, 1)}; USD-хедж: {pct(d.usd_hedge_share, 1)} безрисковой части")
        print(f"Ожидаемая волатильность: {pct(d.exp_vol, 1)}; VaR95 1 мес.: {pct(d.var95_1m, 1)}\n")
        with pd.option_context("display.width", 140):
            print(d.allocation[d.allocation.weight > 0.0005][["weight", "signal", "cluster"]]
                  .to_string(float_format=lambda v: num(v, 3), na_rep="—"))
        for w in d.warnings:
            print("! " + w)
    elif a.cmd == "sensitivity":
        sens = pipeline.rule_sensitivity(r)
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 70):
            print("Чувствительность к коэффициентам (выход — доля рисковых активов или USD-хедж):")
            print(sens["coefficients"].to_string(float_format=lambda v: num(v, 3), na_rep="—"), "\n")
            print("Монотонность правил:")
            print(sens["monotonicity"].to_string(float_format=lambda v: num(v, 1), na_rep="—"), "\n")
            print(f"Доля рисковых активов по P(рецессии) и сценарию (фаза «{r.macro.phase}»):")
            print(sens["risk_grid"].rename(index=lambda v: num(v, 1)).to_string(float_format=lambda v: num(v, 3), na_rep="—"))
    elif a.cmd == "significance":
        sharpe, dm = pipeline.significance_tables(r)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(f"Безрисковая ставка {pct(r.settings.risk_free, 1)} ({r.settings.risk_free_note}).")
            print(f"Шарп стратегий, блочный бутстреп ({r.sharpe_tests.attrs['n_boot']} повторов, "
                  f"блоки {r.sharpe_tests.attrs['block']} дн.):")
            print(sharpe.to_string(float_format=lambda v: num(v, 3), na_rep="—"), "\n")
            print("Диболд–Мариано против наивного прогноза r̂ = 0 (DM < 0 — модель точнее):")
            print(dm.to_string(float_format=lambda v: num(v, 3), na_rep="—"))
    elif a.cmd == "export":
        from .export import excel_available, export
        out = a.output or ("dss_tables.xlsx" if excel_available() else "dss_tables.zip")
        if out.endswith(".xlsx") and not excel_available():
            print("Для .xlsx нужен пакет openpyxl: python -m pip install openpyxl "
                  "(или укажите файл .zip — таблицы будут в CSV)", file=sys.stderr)
            return 2
        data, ext, _ = export(r, "xlsx" if out.endswith(".xlsx") else "zip")
        with open(out, "wb") as fh:
            fh.write(data)
        print(f"Таблицы: {out}")
    else:
        from .report import build_report
        with open(a.output, "w", encoding="utf-8") as fh:
            fh.write(build_report(r))
        print(f"Отчёт: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
