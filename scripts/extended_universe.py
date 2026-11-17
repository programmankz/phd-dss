"""Exploratory check: do the conclusions hold on a wider set of shares, including the second exchange?

The main set is seven KASE shares. Here the set is widened with KSPI and KMGZ (already in
data/prices_kase.csv) and with four shares loaded on 17.11.2026: AIRA, ASBN and FRHC_KZ from KASE
and CORE (Solidcore Resources) from AIX, quoted in US dollars and converted to tenge at the
official NBK rate of the day. The share part is equally weighted, the rules are the expert ones,
nothing is tuned. Both periods overlap data that were already used, the set of added shares was
chosen after the results were known: nothing here is a preregistered test.
Writes docs/results/extended_universe.csv and docs/extended_universe.md.

Usage::

    python scripts/extended_universe.py            # run the check
    python scripts/extended_universe.py --actions  # rebuild data/corporate_actions_new.csv first
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_corporate_actions as cca  # noqa: E402
import end_to_end as e2e  # noqa: E402
import experiment_2008_2019 as x  # noqa: E402
from dss import evaluation  # noqa: E402
from dss.data import loaders  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

KASE8 = e2e.KASE7 + ["KSPI"]
KASE9 = KASE8 + ["KMGZ"]
NEW3 = ["ASBN", "FRHC_KZ", "CORE"]
END = x.END_B
# period -> (base set, added shares, start, shock windows)
PERIODS = {"2022–2026": (KASE8, NEW3, "2022-01-10", {"2022": x.WINDOWS_B["2022"]}),
           "2024–2026": (KASE9, NEW3 + ["AIRA"], "2024-02-19", {})}
CONFIGS = (x.EW100, x.EW, x.MACRO, x.CUR, x.MC)
NAMES = {"AIRA": "Air Astana, KASE", "ASBN": "ForteBank, KASE", "FRHC_KZ": "Freedom Holding, KASE",
         "CORE": "Solidcore Resources, AIX, USD"}


def write_actions() -> pd.DataFrame:
    """data/corporate_actions_new.csv: dividend rows of the new KASE shares (ex-dates by the KASE rule)."""
    d = loaders.data_dir()
    raw = pd.concat([cca.load_raw(), loaders._dated_frame(d / "prices_kase_new.csv", "Date,<тикер1>,…")], axis=1, sort=True)
    div = pd.read_csv(d / "dividends_kase_new.csv")
    actions = cca.build_dividend_actions(div, raw[loaders.KASE_NEW_TICKERS].join(raw[e2e.KASE7], how="outer"))
    actions = actions[actions["ticker"].isin(loaders.KASE_NEW_TICKERS)]
    actions.to_csv(d / "corporate_actions_new.csv", index=False)
    return actions


def extended_inputs(base: list, added: list) -> e2e.Inputs:
    """Inputs of the base KASE set with the added shares joined on KASE trading days."""
    inp = e2e.load_inputs(base)
    new = loaders.load_extended_prices(inp.fx_rates["USD"])[added]
    days = inp.prices.index
    new = new.reindex(days.union(new.index)).ffill().reindex(days)
    prices = inp.prices.join(new).dropna()
    return dataclasses.replace(inp, prices=prices, usd=inp.usd.reindex(prices.index), source=inp.source + "; extended")


def run(inp: e2e.Inputs, start: str, windows: dict):
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    eng = e2e.Engine(inp, "moderate", start, END, need_m4=False, log=lambda *a: None)
    daily, rows = {}, {}
    for cfg in CONFIGS:
        r, lg = e2e.run_config(eng, cfg, rets, x.TC)
        m = e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)
        m = {k: v for k, v in m.items() if not k.startswith("return ")}
        for name, (a, b) in windows.items():
            m[f"window {name}"] = x.window_return(r, a, b)
        daily[cfg.name], rows[cfg.name] = r, m
    return pd.DataFrame(daily), pd.DataFrame(rows).T, rf


def tests_of(daily: pd.DataFrame, rf: pd.Series) -> dict:
    ex = daily.sub(rf.reindex(daily.index), axis=0)
    return {"currency rule − EW + KZT, Sharpe": evaluation.paired_difference_test(ex[x.CUR.name], ex[x.EW.name], x.sharpe_fn),
            "currency rule − EW + KZT, max drawdown":
            evaluation.paired_difference_test(daily[x.CUR.name], daily[x.EW.name], x.mdd_fn),
            "macro rule − EW + KZT, max drawdown":
            evaluation.paired_difference_test(daily[x.MACRO.name], daily[x.EW.name], x.mdd_fn)}


def liquidity(start: str) -> pd.DataFrame:
    """Share of KASE trading days with deals and median daily turnover of the added shares from ``start``."""
    d = loaders.data_dir()
    days = loaders._dated_frame(d / "prices_kase.csv", "Date,…").loc[start:END].index
    usd = loaders.load_fx().frame["USD"]
    vol = loaders._dated_frame(d / "volumes_kase_new.csv", "Date,…")
    aix = loaders._dated_frame(d / "volumes_aix.csv", "Date,…")
    full = pd.concat([vol, aix.mul(usd.reindex(aix.index, method="ffill"), axis=0)], axis=1, sort=True)
    vol, rows = full.reindex(days), {}
    for t in vol:
        s = vol[t].loc[vol[t].first_valid_index():]
        rows[t] = {"first": full[t].first_valid_index(), "traded_share": float((s > 0).mean()),
                   "median_turnover_kzt": float(s[s > 0].median())}
    return pd.DataFrame(rows).T


def usd_link(inp: e2e.Inputs, start: str) -> pd.Series:
    """Correlation of weekly share returns in tenge with the weekly change of USD/KZT."""
    w = inp.prices.loc[start:END].resample("W-FRI").last().pct_change().dropna()
    u = inp.usd.loc[start:END].resample("W-FRI").last().pct_change().reindex(w.index)
    return w.corrwith(u)


def markdown(res: dict, liq: pd.DataFrame, link: pd.Series) -> str:
    out = ["# Расширенный набор бумаг: KASE и AIX (разведочная проверка)", "",
           "Вопрос: сохраняются ли выводы, если к основному набору из семи акций KASE добавить другие ликвидные бумаги "
           "казахстанского рынка, включая вторую биржу — AIX? Добавлены KSPI и KMGZ (уже были в `data/prices_kase.csv`) и "
           "четыре бумаги, загруженные 17.11.2026: AIRA (Air Astana), ASBN (ForteBank), FRHC_KZ (Freedom Holding) с KASE и "
           "CORE (Solidcore Resources, бывший Polymetal) с AIX. CORE торгуется в долларах США, цена пересчитана в тенге по "
           "официальному курсу НБ РК на день торгов. Долевая часть — равные веса, правила экспертные, параметры не менялись. "
           "Оба периода пересекаются с уже использованными данными, состав добавленных бумаг выбран после получения основных "
           "результатов: проверка разведочная, гипотезы ею не проверяются. `python scripts/extended_universe.py`; таблица "
           "`docs/results/extended_universe.csv`. Умеренный профиль, издержки 10 б.п., конец периода 09.10.2026.", "",
           "## Добавленные бумаги", "",
           "| Бумага | Первая сделка в файле | Дней со сделками с 10.01.2022, % | Медианный дневной оборот, млн ₸ | "
           "Корреляция недельной доходности в тенге с изменением курса USD/KZT |", "|---|---|---|---|---|"]
    for t, r in liq.iterrows():
        out.append(f"| {t} ({NAMES[t]}) | {r['first']:%d.%m.%Y} | {pct(r['traded_share'], 0)} | "
                   f"{num(r['median_turnover_kzt'] / 1e6, 1)} | {num(link[t], 2, sign=True)} |")
    base = link.drop(index=[t for t in liq.index if t in link])
    out += ["", f"Доля дней и оборот считаются с первой сделки бумаги, но не раньше 10.01.2022. Для сравнения: у бумаг "
            f"основного набора и KSPI, KMGZ та же корреляция с курсом лежит в пределах от {num(base.min(), 2, sign=True)} до "
            f"{num(base.max(), 2, sign=True)} (2024–2026). Дивиденды AIRA и ASBN учтены по данным KASE "
            "(`data/dividends_kase_new.csv`, `data/corporate_actions_new.csv`); у FRHC_KZ выплат на KASE нет; "
            "дивиденды CORE не учтены: в выгрузке AIX их нет, отдельный источник не проверялся.", ""]
    for period, v in res.items():
        base_set, added, start, windows = PERIODS[period]
        out += [f"## {period}: {len(base_set)} бумаг KASE и {len(base_set) + len(added)} бумаг с добавленными", "",
                f"Начало {pd.Timestamp(start):%d.%m.%Y}. Добавлены: {', '.join(added)}.", ""]
        for label in ("base", "ext"):
            table, tests = v[label]
            n = len(base_set) + (len(added) if label == "ext" else 0)
            out += [f"**{n} бумаг{' (с добавленными)' if label == 'ext' else ' (только KASE)'}**", "",
                    "| Конфигурация | Доходность, % год. | Волатильность, % | Шарп | Макс. просадка, % | "
                    + "".join(f"Окно {w}, % | " for w in windows), "|" + "---|" * (5 + len(windows))]
            for name, r in table.iterrows():
                out.append(f"| {x.RU.get(name, name)} | {pct(r['annual_return'])} | {pct(r['volatility'])} | {num(r['sharpe'])} | "
                           f"{pct(r['max_drawdown'])} | " + "".join(f"{pct(r[f'window {w}'], 2)} | " for w in windows))
            s, m, md = (tests[k] for k in tests)
            out += ["", "| Сравнение | Оценка [95% ДИ] | p (двустор.) |", "|---|---|---|",
                    f"| Валютное правило − равные веса с тенге, Шарп | {num(s['diff'], sign=True)} [{num(s['low'])}; "
                    f"{num(s['high'])}] | {num(s['p_two_sided'], 3)} |",
                    f"| Валютное правило − равные веса с тенге, макс. просадка | {x.pp(m['diff'])} [{x.pp(m['low'])}; "
                    f"{x.pp(m['high'])}] | {num(m['p_two_sided'], 3)} |",
                    f"| Макроправило − равные веса с тенге, макс. просадка | {x.pp(md['diff'])} [{x.pp(md['low'])}; "
                    f"{x.pp(md['high'])}] | {num(md['p_two_sided'], 3)} |", ""]
        e = v["universe"]
        out += [f"Сам состав: равные веса с тенге на расширенном наборе против того же на наборе KASE — разность Шарпа "
                f"{num(e['diff'], sign=True)} [{num(e['low'])}; {num(e['high'])}], p = {num(e['p_two_sided'], 3)}.", ""]
    out += ["Положительная разность просадок означает менее глубокую просадку у правила.", "", "## Выводы", ""]
    for period, v in res.items():
        (tb, sb), (te, se) = v["base"], v["ext"]
        k1, k3 = "currency rule − EW + KZT, Sharpe", "macro rule − EW + KZT, max drawdown"
        out.append(f"* **{period}.** Валютное правило против равных весов с тенге, разность Шарпа: {num(sb[k1]['diff'], sign=True)} "
                   f"на наборе KASE и {num(se[k1]['diff'], sign=True)} [{num(se[k1]['low'])}; {num(se[k1]['high'])}] на расширенном. "
                   f"Макроправило, разность максимальной просадки: {x.pp(sb[k3]['diff'])} и {x.pp(se[k3]['diff'])} "
                   f"[{x.pp(se[k3]['low'])}; {x.pp(se[k3]['high'])}]. Равные веса с тенге: Шарп {num(tb.loc[x.EW.name, 'sharpe'])} "
                   f"и {num(te.loc[x.EW.name, 'sharpe'])}, просадка {pct(tb.loc[x.EW.name, 'max_drawdown'])} и "
                   f"{pct(te.loc[x.EW.name, 'max_drawdown'])}.")
    out += ["* Добавление бумаг, включая CORE с AIX, знаков и порядка величин не меняет: в оба периода долларовая "
            "позиция снижает Шарп (в окне шока 2022 г. она уменьшает потери меньше чем на 1 п.п.), макроправило уменьшает просадку 2022 г. и ничего не даёт в 2024–2026 гг.",
            "* Расширенный набор сам по себе рискованнее: в окне 2022 г. он теряет больше (CORE за окно потерял 64%), "
            "просадка глубже, Шарп тот же.",
            "* CORE котируется в долларах, но защитой от ослабления тенге не служит: связь его недельной доходности в тенге "
            "с курсом такая же слабая, как у акций KASE.",
            "* Девальваций масштаба 2009, 2014 и 2015 гг. в этих периодах не было, поэтому главный результат работы (защита в окнах девальваций 2009, 2014, "
            "2015 гг.) на расширенном наборе проверить нельзя: AIX работает с 2017 г.", ""]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    t0 = time.perf_counter()
    if "--actions" in (argv if argv is not None else sys.argv[1:]):
        print(f"corporate_actions_new.csv: {len(write_actions())} событий")
    res, rows = {}, []
    for period, (base, added, start, windows) in PERIODS.items():
        print(period)
        v, ew = {}, {}
        for label, inp in (("base", e2e.load_inputs(base)), ("ext", extended_inputs(base, added))):
            daily, table, rf = run(inp, start, windows)
            tests = tests_of(daily, rf)
            v[label], ew[label] = (table, tests), daily[x.EW.name] - rf.reindex(daily.index)
            rows.append(table.assign(period=period, universe=label, kind="configuration"))
            rows.append(pd.DataFrame(tests).T.assign(period=period, universe=label, kind="test"))
            if label == "ext" and period == "2024–2026":
                link = usd_link(inp, start)
        v["universe"] = evaluation.paired_difference_test(ew["ext"], ew["base"], x.sharpe_fn)
        res[period] = v
    pd.concat(rows).to_csv(x.RES / "extended_universe.csv")
    (ROOT / "docs" / "extended_universe.md").write_text(markdown(res, liquidity("2022-01-10"), link), encoding="utf-8")
    print(f"written docs/extended_universe.md ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
