"""Authors' notebook of article [A1] (dss/original_a1.py) on the article's data file.

Writes docs/original_a1.md and docs/results/original_a1_*.csv.
Run: python scripts/original_a1.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dss import original_a1 as oa  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402


def md(df: pd.DataFrame, index: str, d: int = 3, pct_cols=()) -> str:
    head = "| " + " | ".join([index, *map(str, df.columns)]) + " |\n|" + "---|" * (len(df.columns) + 1)
    rows = []
    for i, r in df.iterrows():
        cells = [v if isinstance(v, str) else "—" if pd.isna(v) else pct(v, 1) if c in pct_cols else num(v, d)
                 for c, v in r.items()]
        rows.append("| " + " | ".join([str(i), *cells]) + " |")
    return "\n".join([head, *rows])


def main() -> None:
    warnings.filterwarnings("ignore")
    wide = oa.load_article_data()
    res = Path("docs/results")
    nb = oa.NOTEBOOK
    leak, clean = oa.same_year_classification(wide), oa.same_year_classification(wide, past_only=True)
    same = pd.DataFrame({"Ноутбук, точность": pd.Series(nb["same_year"]), "Перенос, точность": leak["accuracy"],
                         "Перенос, ROC-AUC": leak["roc_auc"], "Без года t, точность": clean["accuracy"],
                         "Без года t, ROC-AUC": clean["roc_auc"]})
    reg, clf = oa.next_year_models(wide)
    a = clf.attrs
    nclf = pd.DataFrame({"Ноутбук, точность": {k: v[0] for k, v in nb["next_year_clf"].items()},
                         "Ноутбук, ROC-AUC": {k: v[1] for k, v in nb["next_year_clf"].items()},
                         "Перенос, точность": clf["accuracy"], "Перенос, ROC-AUC": clf["roc_auc"],
                         "Доля прогнозов «спад»": clf["predicted_recessions"]})
    nreg = pd.DataFrame({"Ноутбук, RMSE": {k: v[1] for k, v in nb["next_year_reg"].items()},
                         "Ноутбук, R²": {k: v[2] for k, v in nb["next_year_reg"].items()},
                         "Перенос, RMSE": reg["RMSE"], "Перенос, R²": reg["R2"]})
    kz = oa.country_forecasts(wide)
    kzt = pd.concat([kz.add_prefix("Перенос: "), pd.DataFrame(nb["kz"], index=kz.index).add_prefix("Ноутбук: ")], axis=1)
    st, sil, outliers = oa.clusters(wide)
    cl = pd.DataFrame({"Ноутбук": pd.Series(nb["clusters"]), "Перенос": st["cluster"].value_counts()}).astype(float)
    for name, t in (("same_year", same), ("next_year_clf", nclf), ("next_year_reg", nreg), ("kazakhstan", kzt), ("clusters", cl)):
        t.to_csv(res / f"original_a1_{name}.csv")
    always = 1 - leak.attrs["crisis_test"]
    out = ["# Исходный код статьи [A1] по росту ВВП", "",
           "Файл создан скриптом `scripts/original_a1.py`. Расчёт повторяет ноутбук авторов `GDP.ipynb` по ячейкам "
           "(`dss/original_a1.py`) на файле данных статьи `data/world_gdp_data.csv` (IMF WEO, 196 стран, 1980–2024). "
           "Модели XGBoost и LightGBM из ноутбука здесь не считались: это необязательные библиотеки, а лучшими в ноутбуке "
           "оказались линейные модели.", "",
           "## 1. Классификация спада в году t (ячейки 5, 17–20)", "",
           f"Обучение до 2015 года ({leak.attrs['train']} наблюдений), тест 2015–2024 ({leak.attrs['test']}), доля лет со "
           f"спадом в тесте {pct(leak.attrs['crisis_test'], 1)}. Правило «спада не будет» даёт точность {pct(always, 1)}.", "",
           md(same, "Модель", 3, pct_cols=("Ноутбук, точность", "Перенос, точность", "Без года t, точность")), "",
           "В ноутбуке признаки `rolling_mean3`, `rolling_std3`, `rolling_mean5` и `cumulative_mean` считаются по ряду, "
           "который включает рост года t, а целевой класс: знак роста того же года t. Столбцы «Без года t» получены тем же "
           "кодом, но скользящие статистики сдвинуты на год назад.", "",
           "## 2. Прогноз на следующий год (ячейки 26–32)", "",
           f"24 признака, обучение 1980–2019 ({a['train']}), тест 2020–2023 ({a['test']}), доля спадов в тесте "
           f"{pct(a['recession_test'], 1)}.", "",
           md(nclf, "Модель", 3, pct_cols=("Доля прогнозов «спад»",)), "", md(nreg, "Модель", 3), "",
           f"У строк 2019 года целевой год 2020, и спад в нём у {pct(a['recession_last_train_year'], 0)} стран. Деревья "
           "запоминают правило «последний год обучения означает спад» и переносят его на все годы теста, отсюда точность "
           "около 0,19 и R² около −4. Признаки `country_mean`, `country_std`, `country_min`, `country_max`, "
           "`country_median` и `gdp_dev_from_mean` считаются по всей выборке, включая годы теста.", "",
           "## 3. Казахстан, прогноз на 2025–2029 (ячейки 13–15)", "", md(kzt, "Год", 2), "",
           "AR(3) со свободным членом на первых разностях, оценка МНК; порядок по AIC в ноутбуке не выбирается. "
           "Коэффициенты: " + "; ".join(num(float(c), 3) for c in kz.attrs["ar_coef"]) + ". «Prophet-style»: квадратичный "
           "тренд и средний остаток каждого пятого года. «LSTM-style»: линейная регрессия на окне 5 лет.", "",
           "## 4. Кластеры стран (ячейки 22–23)", "", md(cl, "Кластер", 0), "",
           f"K-means по шести показателям страны, K = 4; лучший силуэт при K = {int(sil.idxmax())} (в ноутбуке "
           f"{nb['best_k']}). DBSCAN: {outliers} стран-выбросов (в ноутбуке {nb['dbscan_outliers']}). Казахстан в кластере "
           f"«{st.loc['Kazakhstan', 'cluster']}».", ""]
    path = Path("docs/original_a1.md")
    tail = ""
    if path.exists() and "## Выводы" in path.read_text(encoding="utf-8"):
        tail = "## Выводы" + path.read_text(encoding="utf-8").split("## Выводы", 1)[1]
    path.write_text("\n".join(out) + ("\n" + tail if tail else ""), encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
