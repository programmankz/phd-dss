"""Return - risk - liquidity profile of the KASE market.

Stocks of data/prices_kase.csv and the KASE index (data/index_kase.csv), each over its own history
and over the common period of the seven stocks of Table 4: annual return, volatility, maximum
drawdown, Sharpe ratio over the NBK base rate (daily excess returns, as in the article), share of
days without a price change, Amihud illiquidity (data/volumes_kase.csv) and the correlation of
weekly returns with USD/KZT - with dividends (corporate_actions.csv) and without (price only).
Bonds: the KASE bond indices (data/bond_indices_kase.csv, price-type series) and, for the issues of
data/bonds_kase.csv, the number of trading days and the share of exchange days without trades.

Outputs: docs/results/kase_profile.csv, kase_profile_bonds.csv, kase_profile.png, docs/kase_profile.md.

Usage::

    python scripts/kase_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
from dss import significance  # noqa: E402
from dss.data import loaders  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

RES = ROOT / "docs" / "results"
BOND_INDEX_PRICE = ("KZGB_DP", "KZGB_CP", "KASE_BMC")


def rf_daily(days: pd.DatetimeIndex) -> pd.Series:
    rates, _ = loaders.load_rates()
    dt = pd.Series(days, index=days).diff().dt.days.fillna(0) / 365.0
    return rates["kzt"].reindex(days, method="ffill").shift(1) * dt


def profile(px: pd.Series, vol: pd.Series | None, usd: pd.Series, exchange_days=None) -> dict:
    """Metrics of one price series over its non-missing history; ``exchange_days`` - all trading days
    of the exchange, for the share of days without trades in this security."""
    p = px.dropna()
    r = p.pct_change().dropna()
    if len(r) < 60:
        return {}
    ex = r - rf_daily(p.index).reindex(r.index).fillna(0)
    curve = (1 + r).cumprod()
    out = {"start": p.index[0].date(), "end": p.index[-1].date(), "n_days": len(r),
           "annual_return": float(np.expm1(np.log1p(r).sum() * 252 / len(r))),
           "volatility": float(r.std() * np.sqrt(252)), "max_drawdown": float((curve / curve.cummax() - 1).min()),
           "sharpe": float(significance.sharpe(ex.to_numpy(), 0.0)), "zero_share": float((r.abs() < 1e-12).mean())}
    if vol is not None:
        v = vol.reindex(r.index)
        traded = v.where(v > 0)
        out["amihud"] = float((r.abs() / traded * 1e9).mean())
    if exchange_days is not None:
        ex_days = exchange_days[(exchange_days >= p.index[0]) & (exchange_days <= p.index[-1])]
        out["no_trade_share"] = float(1 - len(p) / max(len(ex_days), 1))
    wk_p = p.resample("W-FRI").last()
    wk_u = usd.reindex(p.index, method="ffill").resample("W-FRI").last()
    out["corr_usd_weekly"] = float(wk_p.pct_change().corr(wk_u.pct_change()))
    return out


def main() -> int:
    adj, _ = loaders.raw_prices("kase")
    raw = loaders._dated_frame(loaders.data_dir() / "prices_kase.csv", "Date,<тикер1>,…")
    vols = pd.read_csv(loaders.data_dir() / "volumes_kase.csv", index_col=0, parse_dates=True)
    usd = loaders.load_fx().frame["USD"]
    idx = pd.read_csv(loaders.data_dir() / "index_kase.csv", index_col=0, parse_dates=True)
    common_start = adj[e2e.KASE7].dropna().index[0]
    exch = idx.index.union(adj.dropna(how="all").index)
    rows = []
    for period, start in (("вся история", None), (f"общий период KASE7 (с {common_start:%d.%m.%Y})", common_start)):
        for div, frame in (("с дивидендами", adj), ("без дивидендов", raw)):
            for t in frame.columns:
                s = frame[t].loc[start:] if start is not None else frame[t]
                m = profile(s, vols[t].reindex(s.index) if t in vols else None, usd, exch)
                if m:
                    rows.append({"period": period, "dividends": div, "asset": t, **m})
        s = idx["close"].loc[start:] if start is not None else idx["close"]
        rows.append({"period": period, "dividends": "индекс (ценовой)", "asset": "Индекс KASE",
                     **profile(s, idx["volume_mln_kzt"].reindex(s.index) * 1e6, usd)})
    stocks = pd.DataFrame(rows)
    stocks.to_csv(RES / "kase_profile.csv", index=False)

    bi = pd.read_csv(loaders.data_dir() / "bond_indices_kase.csv", index_col=0, parse_dates=True)
    brows = []
    for c in BOND_INDEX_PRICE:
        if c in bi:
            m = profile(bi[c], None, usd)
            if m:
                brows.append({"kind": "индекс", "asset": c, **m})
    meta = pd.read_csv(loaders.data_dir() / "bonds_kase_meta.csv").set_index("code")
    bonds = pd.read_csv(loaders.data_dir() / "bonds_kase.csv", index_col=0, parse_dates=True)
    exch_days = adj.dropna(how="all").index
    for c in bonds.columns:
        s = bonds[c].dropna()
        if s.empty:
            continue
        window = exch_days[(exch_days >= s.index[0]) & (exch_days <= s.index[-1])]
        brows.append({"kind": "выпуск", "asset": c, "start": s.index[0].date(), "end": s.index[-1].date(),
                      "n_days": int(len(s)), "no_trade_share": float(1 - len(s) / max(len(window), 1)),
                      "currency": meta.loc[c, "currency"] if c in meta.index else "",
                      "type": meta.loc[c, "kind"] if c in meta.index else ""})
    bonds_t = pd.DataFrame(brows)
    bonds_t.to_csv(RES / "kase_profile_bonds.csv", index=False)
    plot(stocks)
    (ROOT / "docs" / "kase_profile.md").write_text(markdown(stocks, bonds_t, common_start), encoding="utf-8")
    print("written docs/kase_profile.md")
    return 0


def plot(stocks: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = stocks[stocks["period"].str.startswith("общий")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for div, mk, col in (("с дивидендами", "o", "#1a1a1a"), ("без дивидендов", "o", "#aaaaaa"),
                         ("индекс (ценовой)", "s", "#555555")):
        d = s[s["dividends"] == div]
        ax.scatter(d["volatility"] * 100, d["annual_return"] * 100, marker=mk, c=col, label=div, s=30)
        for _, r in d.iterrows():
            if div != "без дивидендов":
                ax.annotate(r["asset"], (r["volatility"] * 100, r["annual_return"] * 100), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Волатильность, % годовых")
    ax.set_ylabel("Доходность, % годовых")
    ax.set_title("Акции KASE: доходность и риск, общий период")
    ax.grid(c="#e5e5e5")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "kase_profile.png", dpi=200)
    plt.close(fig)


def markdown(stocks, bonds, common_start) -> str:
    out = ["# Профиль рынка KASE: доходность – риск – ликвидность", "",
           "Результаты `python scripts/kase_profile.py`; таблицы `docs/results/kase_profile.csv`, "
           "`kase_profile_bonds.csv`, график `kase_profile.png`. Шарп — по дневным доходностям сверх базовой ставки "
           "НБ РК, как в статье; «без сделок» — доля торговых дней биржи (дни индекса KASE или любой бумаги) без цены бумаги, "
           "остальные показатели — по дням со сделками; Амихуд — mean(|r|/оборот)·10⁹ по дням со сделками (оборот в тенге); «без "
           "изменения цены» — доля дней с нулевой доходностью; корреляция — недельных доходностей с курсом USD/KZT "
           "(положительная: бумага дорожает в тенге при ослаблении тенге).", ""]
    for period, g in stocks.groupby("period", sort=False):
        out += [f"## Акции, {period}", "",
                "| Бумага | Дивиденды | Период | Доходность, % | Волатильность, % | Макс. просадка, % | Шарп | "
                "Без изменения цены, % | Без сделок, % | Амихуд | Корр. с USD/KZT |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values(["asset", "dividends"]).iterrows():
            out.append(f"| {r['asset']} | {r['dividends']} | {r['start']} – {r['end']} | {pct(r['annual_return'])} | "
                       f"{pct(r['volatility'])} | {pct(r['max_drawdown'])} | {num(r['sharpe'])} | {pct(r['zero_share'])} | "
                       f"{pct(r.get('no_trade_share'))} | {num(r.get('amihud'), 3)} | {num(r['corr_usd_weekly'])} |")
        out.append("")
    out += ["![](results/kase_profile.png)", "", "## Облигации", "",
            "Индексы (KZGB_DP — «грязная» цена ГЦБ, KZGB_CP — чистая цена ГЦБ, KASE_BMC — корпоративные облигации). "
            "Это ценовые индексы без реинвестирования купонов, поэтому их доходность и Шарп занижены и с акциями "
            "не сравнимы; они показывают только колебания цен:", "",
            "| Индекс | Период | Доходность, % | Волатильность, % | Макс. просадка, % | Шарп | Корр. с USD/KZT |",
            "|---|---|---|---|---|---|---|"]
    for _, r in bonds[bonds["kind"] == "индекс"].iterrows():
        out.append(f"| {r['asset']} | {r['start']} – {r['end']} | {pct(r['annual_return'])} | {pct(r['volatility'])} | "
                   f"{pct(r['max_drawdown'])} | {num(r['sharpe'])} | {num(r['corr_usd_weekly'])} |")
    out += ["", "Отдельные выпуски: по ним нет непрерывного ряда цен, поэтому показаны только число дней со сделками "
            "и доля торговых дней биржи без сделок между первой и последней сделкой.", "",
            "| Выпуск | Тип | Валюта | Период | Дней со сделками | Без сделок, % |", "|---|---|---|---|---|---|"]
    for _, r in bonds[bonds["kind"] == "выпуск"].iterrows():
        out.append(f"| {r['asset']} | {r['type']} | {r['currency']} | {r['start']} – {r['end']} | {r['n_days']} | "
                   f"{pct(r['no_trade_share'])} |")
    s = stocks[(stocks["period"].str.startswith("общий")) & (stocks["dividends"] == "с дивидендами")]
    out += ["", "## Выводы", "",
            f"* В общем периоде (с {common_start:%d.%m.%Y}) Шарп акций с дивидендами — от {num(s['sharpe'].min())} до "
            f"{num(s['sharpe'].max())}; доля дней без изменения цены — от {pct(s['zero_share'].min())} до "
            f"{pct(s['zero_share'].max())}.",
            "* Облигационный рынок KASE для бэктеста защитной части по отдельным выпускам непригоден: у выпусков "
            "десятки–сотни дней со сделками за всю историю. Поэтому тенговая часть в СППР моделируется базовой "
            "ставкой НБ РК (с проверкой чувствительности к её снижению), а индексы ГЦБ — только для сравнения.", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
