"""Experiment of article [A4] repeated on KASE, next to the NSE sample of the article.

Writes docs/article_kase.md and docs/results/article_kase_*.csv.
Run: python scripts/article_kase.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dss import replication as rep  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

PCT = ("total_return", "annual_return", "annual_vol", "max_drawdown", "turnover")
RU = {"total_return": "Доходность за период", "annual_return": "Годовая доходность", "annual_vol": "Волатильность",
      "sharpe": "Шарп (ставка 5%)", "sharpe_local": "Шарп (ставка НБ РК)", "max_drawdown": "Макс. просадка",
      "turnover": "Оборот за ребаланс."}


def md(df: pd.DataFrame, index: str, d: int = 3) -> str:
    head = "| " + " | ".join([index, *[RU.get(c, c) for c in df.columns]]) + " |\n|" + "---|" * (len(df.columns) + 1)
    rows = []
    for i, r in df.iterrows():
        cells = [v if isinstance(v, str) else "—" if pd.isna(v) else pct(v, 1) if c in PCT else num(v, d)
                 for c, v in r.items()]
        rows.append("| " + " | ".join([str(i), *cells]) + " |")
    return "\n".join([head, *rows])


def section(r: rep.Replication, title: str) -> list:
    i = r.info
    t = r.sharpe_tests
    tests = pd.DataFrame({"Шарп": t.sharpe, "Разница с равными весами": t["diff"], "p": t.p_value, "p (Холм)": t.p_holm})
    models = r.models.rename(columns={"hit_rate": "Точность направления", "IC_t": "t(IC)", "dm": "DM", "p_value": "p",
                                      "verdict": "Против наивного прогноза"})
    out = [f"## {title}", "",
           f"{i['assets']} бумаг, данные {i['data_start']:%d.%m.%Y} – {i['data_end']:%d.%m.%Y}; тестовая выборка с "
           f"{i['test_start']:%d.%m.%Y}, {i['bt_days']} торговых дней. Доля дней с ростом портфеля равных весов: "
           f"{pct(i['up_days'], 0)}." + (f" Средняя базовая ставка НБ РК за тест: {pct(i['local_rate'], 1)}."
                                         if i["local_rate"] else ""), "",
           md(r.strategies, "Стратегия", 2), "", "Сравнение с равными весами (блочный бутстреп):", "",
           md(tests, "Стратегия", 2), "", "Модели прогноза на тестовой выборке:", "", md(models, "Модель", 4), ""]
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    runs = {"KASE, 9 бумаг": rep.run("kase", kase_set="nine"), "KASE, 7 бумаг": rep.run("kase", kase_set="seven"),
            "NSE, выборка статьи": rep.run("nse")}
    res = Path("docs/results")
    pd.concat([r.strategies.assign(run=k) for k, r in runs.items()]).to_csv(res / "article_kase_strategies.csv")
    pd.concat([r.models.assign(run=k) for k, r in runs.items()]).to_csv(res / "article_kase_models.csv")
    pd.concat([r.sharpe_tests.assign(run=k) for k, r in runs.items()]).to_csv(res / "article_kase_sharpe_tests.csv")

    def own(r):
        return r.strategies.loc[[i for i in r.strategies.index if not i.startswith("Рынок")]]

    rows = {"Статья [A4], NSE, опубликовано": rep.PUBLISHED}
    for k, r in runs.items():
        best = own(r)["sharpe"].idxmax()
        rows[f"{k}: {best}"] = r.strategies.loc[best]
        rows[f"{k}: Equal Weighted"] = r.strategies.loc["Equal Weighted"]
    summary = pd.DataFrame({c: {k: float(v[c]) for k, v in rows.items()} for c in ("annual_return", "annual_vol", "sharpe")})
    out = ["# Эксперимент статьи [A4] на KASE", "",
           "Файл создан скриптом `scripts/article_kase.py`. Метод статьи «Machine Learning-Based Portfolio Optimization Using "
           "Cross-Sectional Signals: Evidence from the NSE Market» повторён без изменений, заменён только файл цен. "
           "Параметры по выборке не подбирались.", "",
           "**Протокол статьи.** Признаки: лог-доходность, RSI(14), MACD и сигнальная линия, ATR(14), ширина полос "
           "Боллинджера, отношение к SMA(14), объём. Модели: Linear, Ridge, Lasso, SVR, Decision Tree, Random Forest, "
           "Gradient Boosting (четыре нейросетевые модели статьи требуют TensorFlow и здесь не считались). Одно "
           "хронологическое разбиение 80/20, сглаживание по 10 прогнозам, ранги s = Rank(r̂)/N. Четыре стратегии, "
           "ежедневная ребалансировка, выборочная ковариация за 252 дня, без издержек, 0 ≤ w ≤ 1. Шарп при ставке 5%, "
           "как в статье; для KASE дополнительно при средней базовой ставке НБ РК за тестовый период.", "",
           "**Наборы KASE.** 9 бумаг файла `data/prices_kase.csv` имеют общую историю только с 12.2022 (листинг KMGZ), "
           "поэтому тест короткий. Набор из 7 бумаг без KSPI и KMGZ даёт историю с 12.2018 и вдвое более длинный тест. "
           "Цены с дивидендами.", "", "## Сводка", "", md(summary, "Расчёт", 2), ""]
    for k, r in runs.items():
        out += section(r, k)
    path = Path("docs/article_kase.md")
    tail = ""
    if path.exists() and "\n## Выводы" in path.read_text(encoding="utf-8"):
        tail = "\n## Выводы" + path.read_text(encoding="utf-8").split("\n## Выводы", 1)[1]
    path.write_text("\n".join(out) + tail, encoding="utf-8")
    print(md(summary, "Расчёт", 2))


if __name__ == "__main__":
    main()
