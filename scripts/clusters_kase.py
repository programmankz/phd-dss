"""Module III on KASE: similarity measures, liquidity features, stability and the effect on the backtest.

Variants: dissimilarity ``features`` (Euclidean on standardised features, default), ``correlation``
(1 - rho of returns) and ``blend`` (50/50), each with the default features and with the liquidity
features (zero-return share, log Amihud; data/volumes_kase.csv). For every variant on the last
252 trading days before --asof:

* k chosen by the silhouette (as in the DSS), silhouette on the dissimilarity, cluster sizes;
* stability - adjusted Rand index between the labels on the original window and on 200 circular
  block bootstrap resamples of its days (block 10), for the same k;
* the full DSS of Table 4 (2023-2026) with this Module III variant: Sharpe, maximum drawdown,
  share of decisions with relaxed limits.

Also the slide table: ticker -> cluster -> medoid -> features in original units
(docs/results/clusters_kase_table.csv). Outputs: docs/results/clusters_kase_compare.csv,
clusters_kase_table.csv, docs/clusters_kase.md.

Usage::

    python scripts/clusters_kase.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
from dss import clustering, config, significance  # noqa: E402
from dss.data import loaders  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

N_BOOT = 200
FEATURE_RU = {"ann_return": "Доходность, % год.", "ann_vol": "Волатильность, %", "sharpe": "Доходность/риск",
              "max_dd": "Макс. просадка, %", "beta": "Бета", "mom_6m": "Моментум 6 мес., %",
              "zero_ret_share": "Доля дней без изменения цены, %", "log_amihud": "ln Амихуда"}


def labels_for(window, distance, volumes, k=None):
    feats = clustering.asset_features(window, None, volumes)
    X = StandardScaler().fit_transform(feats)
    r = np.log(window).diff().dropna()
    D = clustering.dissimilarity(X, r, distance)
    if k is None:
        kmax = max(2, min(6, len(feats) // 3))
        k = clustering.choose_k(X, range(2, kmax + 1), D=D)["best_k"]
    lab, med, _ = clustering.kmedoids_pam(D, k)
    sil = silhouette_score(D, lab, metric="precomputed") if len(set(lab)) > 1 else np.nan
    return lab, med, k, sil, feats


def stability(window, distance, volumes, k, base, n_boot=N_BOOT):
    idx = significance.block_bootstrap_indices(len(window) - 1, n_boot, block=10)
    r = window.pct_change().iloc[1:].to_numpy()
    vol = volumes.reindex(window.index).iloc[1:].to_numpy() if volumes is not None else None
    aris = []
    for b in idx:
        rb = r[b]
        pb = pd.DataFrame(np.vstack([np.ones(r.shape[1]), np.cumprod(1 + rb, axis=0)]) * window.iloc[0].to_numpy(),
                          index=window.index, columns=window.columns)
        vb = (pd.DataFrame(np.vstack([np.full(r.shape[1], np.nan), vol[b]]), index=window.index, columns=window.columns)
              if vol is not None else None)
        lab, *_ = labels_for(pb, distance, vb, k)
        aris.append(adjusted_rand_score(base, lab))
    return float(np.mean(aris)), float(np.quantile(aris, 0.05))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--asof", default="2026-10-09")
    p.add_argument("--end", default="2026-10-09")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    a = p.parse_args(argv)
    inp = e2e.load_inputs(e2e.KASE7)
    volumes = pd.read_csv(loaders.data_dir() / "volumes_kase.csv", index_col=0, parse_dates=True)[e2e.KASE7]
    window = inp.prices.loc[:a.asof].iloc[-config.COV_WINDOW:]
    eng = e2e.Engine(inp, "moderate", "2023-01-16", a.end, log=lambda *x: None)
    rets = e2e.asset_returns(inp)
    rf = rets[e2e.KZT]
    full = e2e.Config("Full DSS", True, True, True)
    rows, tables = [], {}
    for distance in ("features", "correlation", "blend"):
        for liq in (False, True):
            vol = volumes if liq else None
            lab, med, k, sil, feats = labels_for(window, distance, vol)
            ari, ari5 = stability(window, distance, vol, k, lab, a.n_boot)
            eng.cluster_distance, eng.cluster_volumes, eng._clusters = distance, vol, {}
            r, lg = e2e.run_config(eng, full, rets, 10.0)
            m = e2e.metrics(r, rf, float(lg["turnover"].sum() / (len(r) / 252)), lg)
            sizes = sorted(np.bincount(lab).tolist(), reverse=True)
            name = f"{distance}{' + liquidity' if liq else ''}"
            rows.append({"variant": name, "k": k, "sizes": "/".join(map(str, sizes)),
                         "singletons": int(sum(s == 1 for s in sizes)), "silhouette": sil, "ari_mean": ari,
                         "ari_p05": ari5, "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                         "relaxed_share": m.get("relaxed_share", np.nan), "turnover": m["turnover_per_year"]})
            print(rows[-1])
            if distance == "features":
                t = feats.copy()
                t.insert(0, "medoid", [window.columns[med[c]] for c in lab])
                t.insert(0, "cluster", [f"К{c + 1}" for c in lab])
                tables[name] = t
    cmp = pd.DataFrame(rows).set_index("variant")
    out = ROOT / "docs" / "results"
    cmp.to_csv(out / "clusters_kase_compare.csv")
    table = pd.concat(tables, names=["variant", "ticker"])
    table.to_csv(out / "clusters_kase_table.csv")
    (ROOT / "docs" / "clusters_kase.md").write_text(markdown(cmp, tables, a, window), encoding="utf-8")
    print("written docs/clusters_kase.md")
    return 0


def markdown(cmp, tables, a, window) -> str:
    out = ["# Модуль III на данных KASE: меры сходства, ликвидность, устойчивость", "",
           "Результаты `python scripts/clusters_kase.py`; таблицы `docs/results/clusters_kase_compare.csv` и "
           "`clusters_kase_table.csv`. Окно — 252 торговых дня до "
           f"{pd.Timestamp(a.asof):%d.%m.%Y} ({window.index[0]:%d.%m.%Y} – {window.index[-1]:%d.%m.%Y}), 7 бумаг "
           "таблицы 4, K-medoids (PAM), k по силуэту. Признаки ликвидности: доля дней без изменения цены "
           "(Lesmond и др., 1999) и логарифм меры Амихуда по дневному обороту в тенге. Устойчивость — "
           f"скорректированный индекс Рэнда (ARI) между разбиением исходного окна и разбиениями {a.n_boot} "
           "циркулярных блочных бутстреп-выборок дней (блок 10 дн., то же k). Бэктест — полная СППР таблицы 4 "
           "(2023–2026) с этим вариантом Модуля III, остальное без изменений.", "",
           "| Вариант | k | Размеры кластеров | Одиночек | Силуэт | ARI, среднее | ARI, 5% квантиль | Шарп СППР | "
           "Макс. просадка, % | Доля ослаблений лимитов, % | Оборот, раз/год |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in cmp.iterrows():
        out.append(f"| {v} | {r['k']} | {r['sizes']} | {r['singletons']} | {num(r['silhouette'])} | {num(r['ari_mean'])} | "
                   f"{num(r['ari_p05'])} | {num(r['sharpe'])} | {pct(r['max_drawdown'])} | {pct(r['relaxed_share'], 0)} | "
                   f"{num(r['turnover'], 1)} |")
    single = cmp[cmp["singletons"] > 0].index.tolist()
    best_ari = cmp["ari_mean"].idxmax()
    out += ["", f"Кластеры-одиночки дают варианты: {', '.join(single) if single else 'нет'}. Наиболее устойчив "
            f"вариант «{best_ari}» (ARI {num(cmp.loc[best_ari, 'ari_mean'])}). Разброс Шарпа СППР между вариантами "
            f"{num(cmp['sharpe'].min())} – {num(cmp['sharpe'].max())} меньше ширины бутстреп-интервалов Шарпа "
            "таблицы 4 (около ±1,3), поэтому выбор меры сходства по итоговой доходности не обоснован; выбирать её "
            "нужно по устойчивости и интерпретируемости, до теста.", ""]
    for name, t in tables.items():
        out += [f"## Таблица для слайда: {name}", "", "| Бумага | Кластер | Медоид | " +
                " | ".join(FEATURE_RU.get(c, c) for c in t.columns[2:]) + " |",
                "|---|---|---|" + "---|" * (len(t.columns) - 2)]
        for tk, r in t.sort_values("cluster").iterrows():
            vals = []
            for c in t.columns[2:]:
                v = r[c]
                vals.append(pct(v) if c in ("ann_return", "ann_vol", "max_dd", "mom_6m", "zero_ret_share") else num(v))
            out.append(f"| {tk} | {r['cluster']} | {r['medoid']} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
