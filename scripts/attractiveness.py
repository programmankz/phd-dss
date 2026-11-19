"""Investment attractiveness of Kazakhstan: all definitions on real data.

Writes docs/attractiveness.md and docs/results/attractiveness.csv.
Run: python scripts/attractiveness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dss import attractiveness as att  # noqa: E402
from dss.numfmt import num  # noqa: E402

MODES = [("sign", "quarterly"), ("sign", "annual"), ("accel", "annual"), ("accel", "quarterly"), ("panel", "annual")]


def md(df: pd.DataFrame, d: int = 3, index: str = "") -> str:
    head = "| " + " | ".join([index, *map(str, df.columns)]) + " |\n|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i), *[v if isinstance(v, str) else num(v, d) for v in r]]) + " |"
            for i, r in zip(df.index, df.to_numpy(dtype=object))]
    return "\n".join([head, *rows])


def main() -> None:
    out, rows = [], []
    out.append("# Инвестиционная привлекательность Казахстана: индекс KASE и рост ВВП\n")
    out.append("Файл создан скриптом `scripts/attractiveness.py`. Расчёт разведочный: гипотезы заранее не "
               "регистрировались, параметры по выборке не подбирались.\n")
    out.append("**Показатель.** A = 1, если выполнено условие на рост реального ВВП, иначе 0. Определение выбирает "
               "пользователь: `sign` (рост положителен), `accel` (рост ускоряется, ΔY > 0), `panel` (рост положителен, "
               "классификаторы обучаются на панели стран [A1]).\n")
    out.append(f"**Опережающий индикатор.** Доходность индекса KASE за {att.WINDOW_MONTHS} мес. до начала периода. "
               "Окно задано заранее.\n")
    out.append("**Модели.** ARIMA (основная): порядок интегрирования d по тесту Дики–Фуллера на уровне 5%, p и q по AIC, "
               "при d = 1 без дрейфа. ARIMA + KASE: та же модель того же порядка на той же выборке с доходностью индекса как "
               "внешним фактором. Для сравнения оставлена ARIMA по сетке порядков [A1] на всей истории роста, где d = 1 всегда. AR(1), AR(1) + KASE, логистическая регрессия, "
               "случайный лес, градиентный бустинг. Проверка вне выборки расширяющимся окном. Признаки машинного "
               "обучения: рост в предыдущем периоде, средний темп и стандартное отклонение роста за "
               f"{att.ROLL['annual']} года ({att.ROLL['quarterly']} квартала), доходность KASE.\n")
    out.append("**Данные.** Годовой рост ВВП: `data/gdp_growth.csv`. Квартальный рост г/г: "
               "`data/gdp_quarterly_kz.csv` (Бюро национальной статистики, с 2011 года). Индекс KASE с 12.07.2000: "
               "`data/index_kase_2000_2007.csv` и `data/index_kase.csv`.\n")
    cl = att.country_clusters()
    prof = cl["profiles"].rename(columns={"countries": "Стран", "mean": "Средний темп роста, %", "std": "Ст. откл. роста, п.п."})
    prof["Стран"] = prof["Стран"].astype(str)
    out.append("\n## Кластеры стран по среднему темпу и разбросу роста\n")
    out.append(f"K-means по стандартизованным среднему и стандартному отклонению годового роста за {cl['start']}–2024 гг. "
               f"(период индекса KASE), {len(cl['table'])} экономик с рядом не короче 20 лет. В статье [A1] кластеров четыре и "
               "добавлена асимметрия; здесь число кластеров выбрано по силуэту.\n")
    out.append(md(cl["silhouette"].to_frame("Силуэт"), 3, "Кластеров") + "\n")
    out.append(md(prof, 2, "Кластер") + "\n")
    peers = cl["table"][cl["table"]["cluster"] == cl["country_cluster"]].index.tolist()
    out.append(f"Казахстан в кластере {cl['country_cluster']}. Страны этого кластера: " + ", ".join(peers) + ".\n")
    for definition, freq in MODES:
        r = att.run(definition, freq)
        d, s = r.diagnostics, r.signal
        out.append(f"\n## {r.label}, данные {att.FREQS[r.freq]}\n")
        out.append(f"Наблюдений {d['n']} ({r.data.index[0]} – {r.data.index[-1]}), доля A = 1: {num(d['share_positive'] * 100, 0)}%. "
                   f"Проверка: {r.probs.index[0]} – {r.probs.index[-1]}, {len(r.probs)} прогнозов.\n")
        for note in r.notes:
            out.append(f"> {note}\n")
        out.append(md(r.metrics, 3, "Модель") + "\n")
        out.append(f"Диболд–Мариано, ARIMA против ARIMA + KASE: {num(d['dm_arima_vs_arimax'][0], 2)}, p = {num(d['dm_arima_vs_arimax'][1], 3)}; "
                   f"AR(1) против AR(1) + KASE: {num(d['dm_ar_vs_kase'][0], 2)}, p = {num(d['dm_ar_vs_kase'][1], 3)} "
                   "(положительное значение: модель с индексом точнее).\n")
        if definition != "panel" and not (definition == "accel" and freq == "quarterly") and not (definition == "sign" and freq == "annual"):
            out.append("Уравнение роста g(t) = c + a·g(t−1) + b·KASE, ошибки Ньюи–Уэста:\n")
            out.append(md(r.coefficients, 4, "Фактор") + "\n")
            out.append("Корреляция роста с доходностью KASE по сдвигу окна назад (справочно):\n")
            out.append(md(r.lag_profile, 3, "Сдвиг, периодов") + "\n")
            out.append(f"Гетероскедастичность остатков: ARCH-LM {num(d['arch_lm'][0], 2)} (p = {num(d['arch_lm'][1], 3)}), "
                       f"Бройш–Паган {num(d['breusch_pagan'][0], 2)} (p = {num(d['breusch_pagan'][1], 3)}). "
                       f"Стандартное отклонение роста: {num(d['std_first_half'], 2)} п.п. до {d['split']} и "
                       f"{num(d['std_second_half'], 2)} п.п. после.\n")
        if r.panel is not None:
            out.append("Качество на всех странах панели за те же годы:\n")
            out.append(md(r.panel, 3, "Модель") + "\n")
        if s["available"]:
            out.append(f"Сигнал на {s['period']}: A = {s['value']}, P(A = 1) = {num(s['probs']['ARIMA + KASE'], 2)}, "
                       f"прогноз роста {num(s['growth']['ARIMA + KASE'], 1)}% (ARIMA без индекса {num(s['growth']['ARIMA'], 1)}%, "
                       f"AR(1) + KASE {num(s['growth']['AR(1) + KASE'], 1)}%).\n")
        t = r.metrics.reset_index(names="model")
        t.insert(0, "freq", r.freq)
        t.insert(0, "definition", definition)
        rows.append(t)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv("docs/results/attractiveness.csv", index=False, float_format="%.4f")
    text = "\n".join(out)
    marker = "\n## Выводы\n"
    path = Path("docs/attractiveness.md")
    tail = path.read_text(encoding="utf-8").split(marker, 1)[1] if path.exists() and marker in path.read_text(encoding="utf-8") else ""
    path.write_text(text + (marker + tail if tail else ""), encoding="utf-8")
    print("docs/attractiveness.md, docs/results/attractiveness.csv")


if __name__ == "__main__":
    main()
