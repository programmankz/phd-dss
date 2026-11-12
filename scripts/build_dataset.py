"""Build the KZ-MacroFin dataset and evaluate the regime classifiers.

Outputs: data_build/kz_macrofin.csv, docs/results/kz_macrofin_classifiers.csv,
docs/results/kz_macrofin_predictions.csv, docs/dataset.md.

Usage::

    python scripts/build_dataset.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dss import dataset  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

FEATURE_DOC = {
    "usd_ret_4w": "ln(USD/KZT_t / USD/KZT_{t−4 нед.}), официальный курс НБ РК",
    "usd_ret_13w": "то же за 13 недель",
    "usd_vol_13w": "ст. откл. дневных лог-изменений USD/KZT за 13 недель · √252",
    "brent_ret_4w": "лог-изменение цены Brent за 4 недели (FRED DCOILBRENTEU)",
    "brent_ret_13w": "то же за 13 недель",
    "commodity_fx_4w": "среднее лог-изменение NOK, BRL, RUB, CLP к USD за 4 недели (AZN привязан и исключён)",
    "base_rate": "базовая ставка НБ РК на t, доля",
    "base_rate_chg_13w": "изменение базовой ставки за 13 недель",
    "rate_spread": "базовая ставка − ставка 3-мес. T-bill",
    "kase_ret_4w": "лог-доходность индекса KASE за 4 недели",
    "kase_ret_13w": "то же за 13 недель",
    "kase_vol_13w": "ст. откл. дневных лог-доходностей индекса KASE за 13 недель · √252",
    "kase_dd_52w": "индекс KASE к максимуму за 52 недели − 1",
    "cpi_yoy": "ИПЦ, % к соответствующему месяцу прошлого года, последний опубликованный к t (дата — cpi_yoy_asof)",
    "kei": "КЭИ нарастающим итогом, % к прошлому году, последний опубликованный (с 2021 г.; в классификаторах не используется)",
    "kei_nominal_yoy": "номинальный КЭИ месяца к тому же месяцу прошлого года − 1, последний опубликованный",
    "gdp_growth_last": "рост ВВП последнего опубликованного года (год Y известен с 1 апреля Y + 1)",
}
LABEL_RU = {"y_dev": "девальвация в ближайшие 13 недель", "y_mstress": "стресс рынка акций в ближайшие 13 недель",
            "y_rec": "рецессия (текущее состояние)"}


def main() -> int:
    inp = dataset.load_inputs()
    ds = dataset.build(inp)
    tab = ds.table
    out = ROOT / "data_build"
    out.mkdir(exist_ok=True)
    tab.to_csv(out / "kz_macrofin.csv", float_format="%.6g")
    print(f"dataset: {len(tab)} weeks, {tab.index.min():%Y-%m-%d} – {tab.index.max():%Y-%m-%d}")
    rows, preds = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lab in dataset.LABELS:
            p = dataset.walk_forward_classifiers(ds, lab)
            preds[lab] = p
            rows.append({"label": lab, "first_week": p.index.min().date() if len(p) else None,
                         **dataset.classifier_metrics(p)})
            print(rows[-1])
    met = pd.DataFrame(rows).set_index("label")
    res = ROOT / "docs" / "results"
    met.to_csv(res / "kz_macrofin_classifiers.csv")
    pd.concat(preds, names=["label", "date"]).to_csv(res / "kz_macrofin_predictions.csv", float_format="%.5g")
    (ROOT / "docs" / "dataset.md").write_text(markdown(ds, met), encoding="utf-8")
    print("written docs/dataset.md")
    return 0


def episodes(s: pd.Series) -> int:
    y = s.dropna().astype(int)
    return int(((y.diff().fillna(y.iloc[0]) == 1)).sum()) if len(y) else 0


def markdown(ds, met) -> str:
    tab = ds.table
    out = ["# Набор данных KZ-MacroFin", "",
           "Недельный размеченный набор макрофинансовой среды Казахстана для режимных классификаторов. Строится "
           "`python scripts/build_dataset.py` (код `dss/dataset.py`) из файлов `data/`; результат — "
           f"`data_build/kz_macrofin.csv`, {len(tab)} недель, {tab.index.min():%d.%m.%Y} – {tab.index.max():%d.%m.%Y}. "
           "Строка — последняя торговая дата индекса KASE в неделе t.", "",
           "## Признаки", "",
           "Все признаки строятся только по данным, доступным на t: рыночные ряды — на t, месячные — по дате "
           "публикации (столбцы `*_asof`), годовой ВВП — с 1 апреля следующего года. Даты публикации КЭИ и ИПЦ в "
           "`data/macro_monthly_kz.csv` — **допущение** о графике БНС (КЭИ — 15-го числа следующего месяца, ИПЦ — "
           "первый рабочий день), а не фактические даты релизов. Отсутствие утечки проверяет "
           "`tests/test_dataset.py` (строки до даты отсечки не меняются, если данные после неё удалить).", "",
           "| Признак | Определение | Заполнено недель |", "|---|---|---|"]
    for f in ds.features:
        out.append(f"| `{f}` | {FEATURE_DOC.get(f, '')} | {int(tab[f].notna().sum())} |")
    out += ["", "## Метки", "",
            "| Метка | Определение | Недель с меткой | Доля 1 | Эпизодов |", "|---|---|---|---|---|",
            f"| `y_dev` | 1, если в (t, t + 13 нед.] детектор Модуля II срабатывает: рост USD/KZT > 4% за день или "
            f"> 8% за 5 торговых дней | {int(tab['y_dev'].notna().sum())} | {pct(tab['y_dev'].mean())} | {episodes(tab['y_dev'])} |",
            f"| `y_mstress` | 1, если min_(t, t+13 нед.] KASE / KASE_t − 1 < q₁₀ этой величины на обучающей части "
            f"(до {dataset.TRAIN_END}; q₁₀ = {pct(ds.mstress_threshold)}) | {int(tab['y_mstress'].notna().sum())} | "
            f"{pct(tab['y_mstress'].mean())} | {episodes(tab['y_mstress'])} |",
            f"| `y_rec` | 1, если рост ВВП года t < 0 или (с 2021 г.) КЭИ месяца t нарастающим итогом < 100 | "
            f"{int(tab['y_rec'].notna().sum())} | {pct(tab['y_rec'].mean())} | {episodes(tab['y_rec'])} |",
            "| `y_rank_<бумага>` | терциль (1 — худшая треть … 3 — лучшая) доходности бумаги за следующие 4 недели "
            "среди бумаг, торгующихся на t | по бумагам | — | — |", "",
            "**Оговорка о малом числе событий.** Девальвационных эпизодов и рецессий за 2008–2026 гг. единицы "
            "(рецессия по ВВП — только 2020 г.; крупные девальвации — 2009, 2014, 2015 гг. и отдельные скачки "
            "позже). Метрики классификаторов на таком числе эпизодов очень неустойчивы; их нельзя читать как "
            "оценку качества на будущих кризисах.", "",
            "## Классификаторы режимов", "",
            f"Walk-forward: первое обучение на неделях до {dataset.TRAIN_END}, переобучение раз в 52 недели, в обучение "
            "входят только недели, у которых окно метки закрылось до первой прогнозной недели. Логистическая "
            "регрессия (стандартизация, C = 1) и градиентный бустинг (150 деревьев глубины 2) с сигмоидной "
            "калибровкой вероятностей (3 блока); пропуски — медианой обучающей части. Задержка — число недель от "
            "начала эпизода до первой недели с вероятностью > 0,5 внутри эпизода.", "",
            "| Метка | Недель | С меткой 1 | Эпизодов | ROC-AUC логит | ROC-AUC бустинг | Brier логит | Brier бустинг | "
            "Brier климатологии | Обнаружено эпизодов (логит / бустинг) | Задержка, нед. (логит / бустинг) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in met.iterrows():
        out.append(f"| `{lab}` ({LABEL_RU[lab]}) | {int(r['n_weeks'])} | {int(r['n_positive'])} | {int(r['episodes'])} | "
                   f"{num(r['auc_p_logit'])} | {num(r['auc_p_gb'])} | {num(r['brier_p_logit'], 3)} | {num(r['brier_p_gb'], 3)} | "
                   f"{num(r['brier_climatology'], 3)} | {int(r['detected_p_logit'])} / {int(r['detected_p_gb'])} | "
                   f"{num(r['lag_p_logit'], 1)} / {num(r['lag_p_gb'], 1)} |")
    out += ["", "Первая прогнозная неделя: " + "; ".join(f"`{lab}` — {r['first_week']}" for lab, r in met.iterrows())
            + ". Блок переобучения пропускается, если в обучающей части нет обоих классов: до 2020 г. в данных нет "
            "ни одной недели рецессии, а метка года Y известна только с 1 апреля Y + 1 (рост ВВП), поэтому прогнозы "
            "`y_rec` начинаются только с конца 2021 г. и эпизоды 2020–2021 гг. в оценку не входят; в тестовой части "
            "остаётся один короткий эпизод (январь 2026 г., КЭИ < 100). Для `y_mstress` порог q₁₀ обучающей части (с кризисом 2008–2009 гг.) настолько глубок, "
            "что после 2018 г. таких падений индекса не было: меток 1 в тестовой части нет, ROC-AUC не определён."]
    better = [lab for lab, r in met.iterrows()
              if r["n_positive"] > 0 and min(r["brier_p_logit"], r["brier_p_gb"]) < r["brier_climatology"]]
    out += ["", ("Brier ниже климатологии (постоянной доли событий) у меток: " + ", ".join(f"`{b}`" for b in better) + "."
                 if better else "Ни один классификатор не обыгрывает климатологию по Brier: вероятностные прогнозы "
                 "режимов на этих данных не лучше постоянной частоты событий."),
            "", "Вероятности π_t и p_dev этих классификаторов в правила раздела 4 пока не подключены: при таком числе "
            "эпизодов замена P(рецессии) Модуля I на них не обоснована (см. `docs/progress.md`).", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
