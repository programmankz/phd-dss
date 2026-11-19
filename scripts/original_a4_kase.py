"""Authors' code of article [A4] (dss/original_a4.py) run on KASE.

Stage A (per-share models) is computed here and stored in docs/results/original_a4_kase_stage_a_{11,7}.csv;
the four neural networks need TensorFlow (pip install tensorflow). Then docs/original_a4_kase.md is written.
Run: python scripts/original_a4_kase.py [--no-dl] [--report-only]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dss import original_a4 as oa  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

PCT = ("total_return", "annual_return", "annual_vol", "max_drawdown")
RU = {"total_return": "Доходность за период", "annual_return": "Годовая доходность", "annual_vol": "Волатильность",
      "sharpe": "Шарп (ставка 5%)", "max_drawdown": "Макс. просадка"}
SETS = {"seven": "7 бумаг", "nine": "9 бумаг"}


def md(df: pd.DataFrame, index: str, d: int = 2, pct_cols=PCT, pd_: int = 1) -> str:
    head = "| " + " | ".join([index, *[RU.get(c, c) for c in df.columns]]) + " |\n|" + "---|" * (len(df.columns) + 1)
    rows = []
    for i, r in df.iterrows():
        cells = [v if isinstance(v, str) else "—" if pd.isna(v) else pct(v, pd_) if c in pct_cols else num(v, d)
                 for c, v in r.items()]
        rows.append("| " + " | ".join([str(i), *cells]) + " |")
    return "\n".join([head, *rows])


def stage_a(with_dl: bool) -> None:
    processed = oa.kase_processed()
    summary, metrics = oa.stage_a(processed, with_dl, log=print)
    n = 11 if with_dl else 7
    res = oa.results_dir()
    summary.to_csv(res / oa.STAGE_A_FILES[n], index=False)
    metrics.to_csv(res / f"original_a4_kase_models_{n}.csv", index=False)


def report() -> None:
    res = oa.results_dir()
    out = ["# Исходный код статьи [A4] на данных KASE", "",
           "Файл создан скриптом `scripts/original_a4_kase.py`. Расчёт повторяет ноутбуки `Data collection.ipynb` и "
           "`Final_calc.ipynb` репозитория статьи (github.com/Yerassyl04/signal-based-portopt) шаг за шагом, заменены "
           "только данные: дневные свечи KASE из `data/ohlcv_kase.csv` вместо 20 акций NSE. Гиперпараметры взяты из "
           "ноутбука и по выборке не подбирались.", "",
           "**Шаги исходного кода.** Для каждой бумаги отдельно: признаки Log_Ret, RSI(14), MACD, ATR(14), ширина полос "
           "Боллинджера, OBV, SMA(14), объём; окно 60 дней; цель: доходность следующего дня; хронологическое разбиение "
           "80/20; 11 моделей (Linear, Ridge, Lasso, SVR, Decision Tree, Random Forest, Gradient Boosting, RNN, LSTM, "
           "CNN-LSTM, LSTM-MLP). Лучшая модель бумаги выбирается по 0,6·точность направления + 0,4·(R² + 1)·50 на тестовой "
           "части, сигнал бумаги: среднее 10 последних прогнозов этой модели, затем ранг/N. Портфели: Signal Weighted, "
           "Equal Weighted, Min Variance, Max Signal-to-Risk; ковариация по всей выборке.", "",
           "**Как получена таблица IV статьи.** Ячейки, которая строит доходности стратегий, в репозитории нет. "
           "Опубликованные числа воспроизводятся, если веса из конца выборки держать постоянными на всём периоде "
           "2020–2025 (на файлах статьи: Signal Weighted 21,82% годовых, Шарп 0,97, просадка −36,60% против "
           "опубликованных 21,61%, 0,95 и −36,60%). Прогнозы и ковариация при этом относятся к концу периода, а "
           "доходность считается с его начала, поэтому это расчёт внутри выборки. Ниже он показан «как в статье», и "
           "рядом те же веса только на тестовых 20% дат. Строгая проверка вне выборки лежит в `docs/article_kase.md`.", "",
           "## Таблица IV статьи (NSE, опубликовано)", "", md(oa.PUBLISHED_TABLE, "Стратегия"), ""]
    rows = []
    for n in (11, 7):
        if not (res / oa.STAGE_A_FILES[n]).exists():
            continue
        for key, label in SETS.items():
            r = oa.run_kase(key, n)
            i = r.info
            title = f"KASE, {label}, {n} моделей"
            s = r.summary.set_index("Stock")[["Best_Model", "Predicted_Return", "Rank_Score"]]
            s.columns = ["Лучшая модель", "Прогноз, % в день", "Ранг"]
            s["Прогноз, % в день"] = s["Прогноз, % в день"] * 100
            out += [f"## {title}", "",
                    f"Период {i['start']:%d.%m.%Y} – {i['end']:%d.%m.%Y}, {i['days']} торговых дней; тестовая часть с "
                    f"{i['test_start']:%d.%m.%Y}.", "", "Сигналы:", "", md(s, "Бумага", 3, pct_cols=()), "",
                    "Веса портфелей:", "", md(r.weights, "Бумага", 1, pct_cols=tuple(r.weights.columns)), "",
                    "Как в статье (постоянные веса на всём периоде, внутри выборки):", "", md(r.table, "Стратегия"), "",
                    "Те же веса только на тестовой части:", "", md(r.test_table, "Стратегия"), ""]
            r.table.assign(run=title, period="full").to_csv(res / f"original_a4_kase_table_{key}_{n}.csv")
            r.weights.to_csv(res / f"original_a4_kase_weights_{key}_{n}.csv")
            for per, t in (("весь период", r.table), ("тестовая часть", r.test_table)):
                rows.append({"Расчёт": f"{title}, {per}", "Signal Weighted": t.loc["Signal Weighted", "sharpe"],
                             "Equal Weighted": t.loc["Equal Weighted", "sharpe"],
                             "Min Variance": t.loc["Min Variance", "sharpe"],
                             "Max Signal-to-Risk": t.loc["Max Signal-to-Risk", "sharpe"]})
    path = Path("docs/original_a4_kase.md")
    tail = ""
    if path.exists() and "## Выводы" in path.read_text(encoding="utf-8"):
        tail = "## Выводы" + path.read_text(encoding="utf-8").split("## Выводы", 1)[1]
    out += ["## Коэффициент Шарпа: сводка", "", md(pd.DataFrame(rows).set_index("Расчёт"), "Расчёт"), ""]
    path.write_text("\n".join(out) + ("\n" + tail if tail else ""), encoding="utf-8")
    print(f"written {path}")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true", help="7 scikit-learn models only")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        if not a.no_dl and not oa.tensorflow_available():
            sys.exit("TensorFlow is not installed: pip install tensorflow, or run with --no-dl")
        stage_a(not a.no_dl)
    report()


if __name__ == "__main__":
    main()
