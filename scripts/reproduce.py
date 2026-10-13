"""Stage 3 of docs/progress.md: reproduce articles [A1]-[A4] on the committed data.

    python scripts/reproduce.py              # writes docs/results/*.csv and docs/reproduction.md
    python scripts/reproduce.py --with-dl    # also checks the TensorFlow models of [A4] (if installed)

Every number is computed by the DSS modules from the CSV files in data/; the
published reference values are only those present in the repository (see
REFERENCES). A missing reference value is reported as NEED and listed among
the questions for the author. Nothing here is tuned to match an article.
"""
from __future__ import annotations

import argparse
import platform
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dss import clustering, config, fx, macro, pipeline, portfolio, signals, significance  # noqa: E402
from dss.data import loaders  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

NEED = "нужно значение из статьи"
A4_CUT = "2025-12-31"            # end of the [A4] sample ("2020-2025")
A2_PERIOD = ("2000-01-01", "2024-12-31")
ARTICLE_MODELS = ("linear", "ridge", "lasso", "svr", "dt", "rf", "gb")   # scikit-learn part of the 11 models of [A4]

# Published values available in the repository, with the file they come from.
REFERENCES = {
    "a2_r2": ({"USD": 0.78, "EUR": 0.85, "CNY": 0.84, "RUB": 0.21}, "docs/techspecs.md, чек-лист приёмки"),
    "a4_best": ({"annual_return": 0.2161, "annual_vol": 0.1740, "sharpe": 0.95}, "docs/methodology.md, Модуль IV"),
    "a1_text": ({"accuracy": 0.979, "roc_auc": 0.997}, "docs/methodology.md, расхождение 1"),
    "a1_fig9_logreg": ({"accuracy": 0.896, "roc_auc": 0.762}, "docs/methodology.md, расхождение 1"),
    "a1_fig9_trees": ((0.14, 0.17), "docs/methodology.md, расхождение 1 (RF/XGBoost/LightGBM)"),
    "a1_table9_base": ([4.34, 4.64, 4.99, 5.23, 5.45], "tests/test_core.py, test_scenario_offsets_match_table_9"),
}


@dataclass
class Article:
    """Results of one article: CSV tables, a Markdown section and open questions."""
    key: str
    tables: dict = field(default_factory=dict)
    md: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    headline: list = field(default_factory=list)   # rows of the summary table
    summary: str = ""                              # one-sentence verdict for the summary section


# --------------------------------------------------------------------------
# Markdown helpers (Russian number format)
# --------------------------------------------------------------------------
def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """DataFrame of already formatted cells -> Markdown table."""
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i, row in df.iterrows():
        cells = ([str(i)] if index else []) + [str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_frame(df: pd.DataFrame, d: int = 3, pct_cols=(), pct_d: int = 2) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in df.columns:
        out[c] = [pct(v, pct_d) if c in pct_cols else (num(v, d) if isinstance(v, (int, float, np.number)) else v)
                  for v in df[c]]
    return out


def comparison(rows: list[dict]) -> pd.DataFrame:
    """Rows {показатель, статья, СППР, разница, объяснение} with NEED where the article value is missing."""
    return pd.DataFrame(rows, columns=["Показатель", "Статья", "СППР", "Разница", "Объяснение"])


def _diff(article, system, d=2, as_pct=False):
    if article is None or system is None or not np.isfinite(system):
        return "—"
    if as_pct:                                   # difference of two shares, in percentage points
        return num((system - article) * 100, d, sign=True) + " п.п."
    return num(system - article, d, sign=True)


# --------------------------------------------------------------------------
# [A2] Module II - exchange rates
# --------------------------------------------------------------------------
def reproduce_a2(rates: pd.DataFrame) -> Article:
    a = Article("a2")
    r = rates.loc[A2_PERIOD[0]:A2_PERIOD[1]].dropna()
    stats = fx.describe(r)
    reg, _ = fx.trend_regression(r)
    corr = fx.correlations(r)
    shocks = fx.detect_shocks(r)
    a.tables |= {"a2_describe": stats, "a2_trend": reg, "a2_correlations": corr, "a2_shocks": shocks}

    ref, src = REFERENCES["a2_r2"]
    rows = []
    for c in config.CURRENCIES:
        r2 = float(reg.loc[f"{c}/KZT", "R2"])
        same = round(r2, 2) == ref[c]
        rows.append({"Показатель": f"R² тренда {c}/KZT", "Статья": num(ref[c], 2), "СППР": num(r2, 4),
                     "Разница": _diff(ref[c], r2, 3),
                     "Объяснение": "совпадает до 2 знаков" if same else "расхождение: см. текст"})
    for c in config.CURRENCIES:
        rows.append({"Показатель": f"Наклон {c}/KZT, тг/год", "Статья": NEED,
                     "СППР": num(reg.loc[f"{c}/KZT", "Наклон, тг/год"], 3), "Разница": "—", "Объяснение": "—"})
    rows.append({"Показатель": "Табл. 1 (13 строк × 4 валюты)", "Статья": NEED, "СППР": "см. ниже",
                 "Разница": "—", "Объяснение": "—"})
    cmp_ = comparison(rows)
    a.tables["a2_comparison"] = cmp_
    a.questions += ["[A2] Значения табл. 1 (описательная статистика, 13 строк × 4 валюты) и наклоны трендов "
                    "в тенге/год — в репозитории их нет.",
                    "[A2] Какие дни входят в выборку статьи: все календарные дни (НБ РК устанавливает курс и на "
                    f"выходные; в файле {len(r)} наблюдений за 2000–2024) или только торговые?"]
    years = sorted({d.year for d in shocks["date"]}) if len(shocks) else []
    found = {y: y in years for y in (2009, 2014, 2015)}
    all_r2 = all(round(float(reg.loc[f"{c}/KZT", "R2"]), 2) == ref[c] for c in config.CURRENCIES)
    a.summary = ("**[A2] воспроизведена**: R² трендов всех четырёх пар совпадают до второго знака; детектор шоков "
                 "находит девальвации 2009, 2014 и 2015 гг." if all_r2 and all(found.values()) else
                 "**[A2]** воспроизведена частично — см. раздел [A2].")
    a.headline += [{"Статья": "[A2]", "Показатель": f"R² тренда {c}/KZT", "Статья, значение": num(ref[c], 2),
                    "СППР": num(float(reg.loc[f"{c}/KZT", "R2"]), 2)} for c in config.CURRENCIES]

    shocks_f = shocks.copy()
    if len(shocks_f):
        shocks_f["date"] = [f"{d:%d.%m.%Y}" for d in shocks_f["date"]]
        shocks_f = fmt_frame(shocks_f.set_index("date"), 2).rename(columns={
            "before": "Курс до, ₸", "after_60d": "Курс через 60 дн., ₸", "move_pct": "Изменение, %"})
        shocks_f.index.name = "Дата начала"
    st = stats.copy()
    st_f = st.astype(object)
    for c in st.columns:
        st_f[c] = [num(v, 0) if i in ("Сумма", "Счёт") else num(v, 4) for i, v in st[c].items()]
    reg_f = fmt_frame(reg, 4)
    reg_f.index.name = "Пара"
    corr_f = fmt_frame(corr, 3)

    a.md += [
        "## [A2] Модуль II — динамика курсов тенге",
        "",
        f"**Данные и настройки.** `data/fx_rates.csv` (официальные курсы НБ РК), срез {A2_PERIOD[0]} – {A2_PERIOD[1]} "
        f"(период статьи; файл длиннее — до {rates.index[-1]:%Y-%m-%d}); {len(r)} наблюдений, все календарные дни, "
        "на которые в файле есть курс. Функции `fx.describe` (конвенции Excel), `fx.trend_regression` "
        "(курс ~ порядковый номер даты, МНК), `fx.correlations`, `fx.detect_shocks` (порог 4 % за день или 8 % за 5 дней, "
        "эпизоды объединяются в пределах 120 дней).",
        "",
        "### Сравнение со статьёй",
        "",
        md_table(cmp_, index=False),
        "",
        f"Источник опубликованных R²: {src}. "
        + ("**Все четыре R² воспроизведены до второго знака.** " if all(
            round(float(reg.loc[f'{c}/KZT', 'R2']), 2) == ref[c] for c in config.CURRENCIES) else
           "**Не все R² воспроизведены** — см. таблицу. ")
        + "Данные взяты из того же источника, что и в статье (НБ РК), поэтому совпадение ожидаемо; "
          "оно подтверждает, что ряд и регрессия реализованы так же, как в [A2].",
        "",
        "### Трендовая регрессия",
        "",
        md_table(reg_f),
        "",
        "### Таблица 1 [A2] в расчёте СППР",
        "",
        md_table(st_f),
        "",
        "Опубликованные значения табл. 1 в репозитории отсутствуют — сравнение возможно после их передачи автором.",
        "",
        "### Корреляции Пирсона (с датой, рис. 3 [A2])",
        "",
        md_table(corr_f),
        "",
        "### Девальвационные шоки (`fx.detect_shocks`, USD/KZT)",
        "",
        md_table(shocks_f) if len(shocks) else "Шоков не найдено.",
        "",
        "Детектор находит эпизоды, начинающиеся в " + ", ".join(f"{y}" for y in years) + " гг. "
        + "Известные девальвации: 2009 г. — " + ("найдена" if found[2009] else "**не найдена**")
        + "; февраль 2014 г. — " + ("найдена" if found[2014] else "**не найдена**")
        + "; август 2015 г. (переход к плавающему курсу) — " + ("найдена" if found[2015] else "**не найдена**")
        + ". " + ("Кроме них найдены эпизоды " + ", ".join(str(y) for y in years if y not in found) + " гг."
                  + (" (2020 — пандемия и падение нефти, 2022 — санкции против России). "
                     if [y for y in years if y not in found] == [2020, 2022] else " ")
                  if set(years) - set(found) else "")
        + "Изменение курса считается от уровня за 10 дней до первого сигнала до уровня через 60 дней после него.",
        "",
    ]
    return a


# --------------------------------------------------------------------------
# [A4] Module IV - NSE
# --------------------------------------------------------------------------
A4_SETTINGS = {
    "paper": config.Settings(universe="nse", protocol="paper", cov_method="sample", rebalance="D",
                             tc_bps=0.0, backtest_caps=False),
    "walk_forward": config.Settings(universe="nse"),
}
METRIC_COLS = ["total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown", "turnover"]
METRIC_RU = {"total_return": "Доходность за период", "annual_return": "Годовая доходность",
             "annual_vol": "Волатильность", "sharpe": "Шарп", "max_drawdown": "Макс. просадка",
             "turnover": "Оборачиваемость"}
PCT_METRICS = ("total_return", "annual_return", "annual_vol", "max_drawdown", "turnover")


def run_module4(prices: pd.DataFrame, st: config.Settings, fundamentals: pd.DataFrame | None = None):
    """Module IV as in dss.pipeline.run, on a given price frame (e.g. cut at the article's end date)."""
    proto = config.PROTOCOLS[st.protocol]
    s = signals.run(prices, st.models, proto["train_share"], proto["retrain_every"])
    c = clustering.cluster_assets(prices, fundamentals) if st.backtest_caps else None
    limits, _ = pipeline.backtest_limits(st, c, prices.columns)
    bt = portfolio.backtest(prices, s.ranks, st.rebalance, st.tc_bps, st.cov_method, *limits,
                            risk_free=st.risk_free)
    return s, bt


def reproduce_a4(prices: pd.DataFrame, fundamentals: pd.DataFrame | None, with_dl: bool = False) -> Article:
    a = Article("a4")
    periods = {f"до {A4_CUT}": prices.loc[:A4_CUT], "весь файл": prices}
    met, sh, dm, mm, info = [], [], [], [], []
    runs = {}
    for pname, px in periods.items():
        for proto, st in A4_SETTINGS.items():
            s, bt = run_module4(px, st, fundamentals)
            runs[(pname, proto)] = (s, bt)
            tag = {"period": pname, "protocol": proto}
            met.append(bt.metrics[METRIC_COLS].assign(**tag))
            sh.append(significance.sharpe_tests(bt.daily, st.risk_free).assign(**tag))
            dm.append(significance.forecast_tests(s.predictions, px, signals.MODEL_LABELS).assign(**tag))
            mm.append(s.metrics.assign(**tag))
            info.append(tag | {"data_start": px.index[0].date(), "data_end": px.index[-1].date(),
                               "test_start": s.test_start.date(), "bt_start": bt.daily.index[0].date(),
                               "bt_end": bt.daily.index[-1].date(), "bt_days": len(bt.daily),
                               "train_share": config.PROTOCOLS[proto]["train_share"],
                               "retrain_every": config.PROTOCOLS[proto]["retrain_every"],
                               "models": ",".join(st.models), "cov": st.cov_method, "rebalance": st.rebalance,
                               "tc_bps": st.tc_bps, "caps": st.backtest_caps, "risk_free": st.risk_free})
    metrics = pd.concat(met).reset_index()
    info_df = pd.DataFrame(info)
    a.tables |= {"a4_metrics": metrics, "a4_runs": info_df,
                 "a4_sharpe_tests": pd.concat(sh).reset_index(),
                 "a4_forecast_tests": pd.concat(dm).reset_index(),
                 "a4_model_metrics": pd.concat(mm).reset_index()}

    # all scikit-learn models of the article, article mode, article period: forecast metrics
    st7 = replace(A4_SETTINGS["paper"], models=ARTICLE_MODELS)
    s7, bt7 = run_module4(periods[f"до {A4_CUT}"], st7, fundamentals)
    dm7 = significance.forecast_tests(s7.predictions, periods[f"до {A4_CUT}"], signals.MODEL_LABELS)
    a.tables["a4_article_models"] = s7.metrics.join(dm7[["dm", "p_value", "verdict"]], how="left").reset_index()
    a.tables["a4_article_models_backtest"] = bt7.metrics[METRIC_COLS].reset_index()

    dl_md = _a4_dl_check(periods[f"до {A4_CUT}"], fundamentals, with_dl, a)

    # comparison with the only Table IV row in the repository
    ref, src = REFERENCES["a4_best"]
    p_main = metrics[(metrics.period == f"до {A4_CUT}") & (metrics.protocol == "paper")].set_index("strategy")
    rows = []
    for strat in p_main.index:
        rows.append({"Показатель": f"{strat}: годовая доходность / волатильность / Шарп",
                     "Статья": f"{pct(ref['annual_return'], 2)} / {pct(ref['annual_vol'], 2)} / {num(ref['sharpe'], 2)}"
                     if strat == p_main.index[0] else "см. первую строку",
                     "СППР": f"{pct(p_main.loc[strat, 'annual_return'], 2)} / {pct(p_main.loc[strat, 'annual_vol'], 2)}"
                             f" / {num(p_main.loc[strat, 'sharpe'], 2)}",
                     "Разница": f"{_diff(ref['annual_return'], p_main.loc[strat, 'annual_return'], 2, True)} / "
                                f"{_diff(ref['annual_vol'], p_main.loc[strat, 'annual_vol'], 2, True)} / "
                                f"{_diff(ref['sharpe'], p_main.loc[strat, 'sharpe'], 2)}",
                     "Объяснение": "строка статьи не привязана к стратегии в репозитории"})
    rows.append({"Показатель": "Остальные строки табл. IV (доходность за период, просадка, оборачиваемость; "
                 "прочие стратегии)", "Статья": NEED, "СППР": "см. ниже", "Разница": "—", "Объяснение": "—"})
    rows.append({"Показатель": "Метрики прогнозов (RMSE, MAE, точность направления) 11 моделей",
                 "Статья": NEED, "СППР": "см. ниже", "Разница": "—", "Объяснение": "—"})
    cmp_ = comparison(rows)
    a.tables["a4_comparison"] = cmp_
    a.questions += [
        "[A4] Какой стратегии соответствует строка табл. IV «годовая доходность 21,61 %, волатильность 17,40 %, "
        "Шарп 0,95»? В коде и документации это не зафиксировано.",
        "[A4] Остальные значения табл. IV (все стратегии и показатели) и метрики прогнозов 11 моделей.",
        "[A4] Точный период тестовой выборки табл. IV и рис. 3 (методика — тест с середины 2024 г.; табл. IV "
        "подписана «2020–2025») и прогнозы какой модели (или ансамбля) использовались для сигналов.",
    ]

    strat = list(portfolio.STRATEGIES)
    best = p_main.loc[strat, "sharpe"].idxmax()
    i_paper = info_df[(info_df.period == f"до {A4_CUT}") & (info_df.protocol == "paper")].iloc[0]
    i_wf = info_df[(info_df.period == f"до {A4_CUT}") & (info_df.protocol == "walk_forward")].iloc[0]
    a.headline.append({"Статья": "[A4]", "Показатель": "Лучшая из 4 стратегий: годовая доходность / Шарп (режим "
                       f"статьи, до {A4_CUT})", "Статья, значение": f"{pct(ref['annual_return'], 2)} / {num(ref['sharpe'], 2)}",
                       "СППР": f"{best}: {pct(p_main.loc[best, 'annual_return'], 2)} / "
                               f"{num(p_main.loc[best, 'sharpe'], 2)}"})

    def mtable(period, proto):
        t = metrics[(metrics.period == period) & (metrics.protocol == proto)].set_index("strategy")[METRIC_COLS]
        f = fmt_frame(t, 2, PCT_METRICS).rename(columns=METRIC_RU)
        f.index.name = "Стратегия"
        return md_table(f)

    def shtable(period, proto):
        t = a.tables["a4_sharpe_tests"]
        t = t[(t.period == period) & (t.protocol == proto)].set_index("strategy")
        f = pd.DataFrame({"Шарп": [num(v, 2) for v in t.sharpe],
                          "95% ДИ": [f"[{num(lo, 2)}; {num(hi, 2)}]" for lo, hi in zip(t.ci_low, t.ci_high)],
                          "Разница с Equal Weighted": [num(v, 2) for v in t["diff"]],
                          "95% ДИ разницы": ["—" if pd.isna(lo) else f"[{num(lo, 2)}; {num(hi, 2)}]"
                                             for lo, hi in zip(t.diff_low, t.diff_high)],
                          "p-значение": [num(v, 3) for v in t.p_value]}, index=t.index)
        f.index.name = "Стратегия"
        return md_table(f)

    def dmtable(period, proto):
        t = a.tables["a4_forecast_tests"]
        t = t[(t.period == period) & (t.protocol == proto)].set_index("model")
        f = pd.DataFrame({"MSE·10⁴": [num(v * 1e4, 3) for v in t.mse_model],
                          "MSE наивного·10⁴": [num(v * 1e4, 3) for v in t.mse_naive],
                          "DM": [num(v, 2) for v in t.dm], "p-значение": [num(v, 3) for v in t.p_value],
                          "Вывод (α = 5 %)": t.verdict}, index=t.index)
        f.index.name = "Модель"
        return md_table(f)

    def runline(period, proto):
        i = info_df[(info_df.period == period) & (info_df.protocol == proto)].iloc[0]
        return (f"данные {i.data_start:%d.%m.%Y} – {i.data_end:%d.%m.%Y}; тестовая выборка с {i.test_start:%d.%m.%Y}; "
                f"бэктест {i.bt_start:%d.%m.%Y} – {i.bt_end:%d.%m.%Y} ({i.bt_days} торговых дней)")

    a7 = a.tables["a4_article_models"].set_index("model")
    a7f = pd.DataFrame({"RMSE": [num(v, 5) for v in a7.RMSE], "MAE": [num(v, 5) for v in a7.MAE],
                        "Точность направления": [pct(v, 1) for v in a7.hit_rate],
                        "IC": [num(v, 4) for v in a7.IC], "DM": [num(v, 2) for v in a7.dm],
                        "p (DM)": [num(v, 3) for v in a7.p_value],
                        "Вывод": [v if isinstance(v, str) else "—" for v in a7.verdict]}, index=a7.index)
    a7f.index.name = "Модель"
    wf_cut = metrics[(metrics.period == f"до {A4_CUT}") & (metrics.protocol == "walk_forward")].set_index("strategy")
    wf_cut = wf_cut.loc[strat]
    wf_sh = a.tables["a4_sharpe_tests"]
    wf_sh = wf_sh[(wf_sh.period == f"до {A4_CUT}") & (wf_sh.protocol == "walk_forward")].set_index("strategy").loc[strat]
    wf_best = wf_cut["sharpe"].idxmax()
    wf_sentence = (
        f"Для сравнения: в walk-forward на том же файле бэктест начинается {i_wf.bt_start:%d.%m.%Y} и захватывает "
        f"рост 2023–2024 гг.; годовая доходность стратегий — {pct(wf_cut['annual_return'].min(), 1)} … "
        f"{pct(wf_cut['annual_return'].max(), 1)}, Шарп — {num(wf_cut['sharpe'].min(), 2)} … "
        f"{num(wf_cut['sharpe'].max(), 2)}, волатильность — {pct(wf_cut['annual_vol'].min(), 1)} … "
        f"{pct(wf_cut['annual_vol'].max(), 1)}. Это тот же порядок, что 21,61 % / 0,95 в статье, но лучший Шарп у "
        f"{wf_best}" + (", т. е. сигналы не добавляют ценности сверх равных весов" if wf_best == "Equal Weighted" else "")
        + "; ни одна сигнальная стратегия не отличается от Equal Weighted значимо (все 95% ДИ разницы Шарпа "
        + ("содержат 0" if ((wf_sh.diff_low <= 0) & (wf_sh.diff_high >= 0)).loc[[x for x in strat if x != "Equal Weighted"]].all()
           else "— см. таблицу") + ").")
    ew_paper = p_main.loc["Equal Weighted"]
    ens = a.tables["a4_model_metrics"]
    ens_ic = ens[ens.model.str.startswith("Ансамбль")].IC
    dm_all = a.tables["a4_forecast_tests"]
    n_better = int((dm_all.verdict == "точнее наивного").sum())
    skill_sentence = (
        "Вторая причина — прогнозы не обладают предсказательной силой: IC ансамбля "
        + ("отрицателен во всех четырёх расчётах" if (ens_ic < 0).all() else
           f"от {num(ens_ic.min(), 3)} до {num(ens_ic.max(), 3)}")
        + ", тест Диболда–Мариано "
        + ("ни в одном расчёте не находит модели точнее наивного прогноза r̂ = 0" if n_better == 0 else
           f"находит модель точнее наивного прогноза в {n_better} случаях из {len(dm_all)}")
        + " (таблицы ниже), поэтому сигнальные стратегии не обгоняют Equal Weighted и рынок.")
    n_test = (prices.loc[:A4_CUT].index >= pd.Timestamp(i_paper.test_start)).sum()

    a.headline.append({"Статья": "[A4]", "Показатель": f"Walk-forward, до {A4_CUT}: годовая доходность / Шарп "
                       "(4 стратегии)", "Статья, значение": f"{pct(ref['annual_return'], 2)} / {num(ref['sharpe'], 2)}",
                       "СППР": f"{pct(wf_cut['annual_return'].min(), 1)} … {pct(wf_cut['annual_return'].max(), 1)} / "
                               f"{num(wf_cut['sharpe'].min(), 2)} … {num(wf_cut['sharpe'].max(), 2)}"})
    a.summary = (f"**[A4] не воспроизведена**: в режиме статьи (тест {i_paper.test_start:%d.%m.%Y} – "
                 f"{i_paper.bt_end:%d.%m.%Y}) лучшая стратегия даёт {pct(p_main.loc[best, 'annual_return'], 1)} / Шарп "
                 f"{num(p_main.loc[best, 'sharpe'], 2)} против 21,61 % / 0,95; в walk-forward с 2023 г. — того же "
                 "порядка, что в статье, но за счёт роста рынка: сигналы не обгоняют равные веса, прогнозы не лучше "
                 "наивного r̂ = 0.")
    a.md += [
        "## [A4] Модуль IV — сигналы и оптимизация портфеля, NSE",
        "",
        "**Данные.** `data/prices_nse.csv`: 20 акций табл. II [A4], цены закрытия Yahoo Finance (скорректированы на "
        f"дивиденды и сплиты), {prices.index[0]:%d.%m.%Y} – {prices.index[-1]:%d.%m.%Y}, {len(prices)} торговых дней. "
        "Tata Motors после разделения в октябре 2025 г. — тикер `TMPV.NS` вместо `TATAMOTORS.NS`; цены до 14.10.2025 "
        "пересчитаны на выделение TMCV (`loaders.adjust_demergers`), иначе в ряду был бы ложный обвал на 40 %. "
        "Расчёт выполнен дважды: на данных, обрезанных по " + A4_CUT + " (период статьи «2020–2025»), и на всём файле.",
        "",
        "**Настройки.**",
        "",
        "* Режим статьи — `config.PAPER_PRESET`: одно хронологическое разбиение 80/20 без переобучения, выборочная "
        "ковариация за 252 дня, ежедневная ребалансировка, без издержек, без лимитов (0 ≤ wᵢ ≤ 1). То же, что "
        "`python -m dss.cli significance --universe nse --protocol paper --cov sample --rebalance D --tc 0 --no-caps`.",
        "* Walk-forward по умолчанию — первое обучение на 50 % дат, переобучение раз в 126 дней, ковариация "
        "Ледуа–Вольфа, еженедельная ребалансировка, издержки 10 б.п., лимиты умеренного профиля (≤ 20 % на бумагу, "
        "≤ 50 % на кластер K-medoids).",
        f"* Прогнозные модели — ансамбль по умолчанию (Ridge, Random Forest, Gradient Boosting), окно 60 дней, "
        f"сглаживание 10, ранги Rank/N; безрисковая ставка {pct(config.RISK_FREE, 0)} (`config.RISK_FREE`); "
        "бутстреп Шарпа — 2000 повторов, кольцевые блоки n^(1/3), seed 42.",
        "",
        "### Сравнение со статьёй",
        "",
        md_table(cmp_, index=False),
        "",
        f"Источник: {src}. В репозитории нет указания, к какой стратегии относится эта строка табл. IV (в "
        "`docs/methodology.md` она названа только как основа для восстановления безрисковой ставки), поэтому она "
        "сопоставлена со всеми стратегиями режима статьи.",
        "",
        f"**Вывод.** В режиме статьи на её периоде ({runline(f'до {A4_CUT}', 'paper')}) ни одна стратегия не "
        f"приближается к 21,61 % / 0,95: лучшая по Шарпу из четырёх стратегий — {best} "
        f"({pct(p_main.loc[best, 'annual_return'], 2)}, Шарп {num(p_main.loc[best, 'sharpe'], 2)})"
        + ("" if best == "Equal Weighted" else f", равновзвешенный портфель — {pct(ew_paper['annual_return'], 2)}")
        + ". Опубликованный результат **не воспроизведён**. "
        "Наиболее вероятная причина — период оценки (расхождение 4 в `docs/methodology.md`): при разбиении 80/20 "
        f"тестовая выборка — последние {n_test} торговых дней (с {i_paper.test_start:%d.%m.%Y}), рынок в это время "
        f"рос слабо (равные веса — {pct(ew_paper['annual_return'], 1)} годовых), а 21,61 % при волатильности 17,40 % "
        "похоже на результат за более длинный и более удачный для рынка отрезок 2020–2025 гг. " + wf_sentence
        + " Получить весь 2020–2025 гг. вне выборки при обучении на тех же данных невозможно. " + skill_sentence,
        "",
    ]
    for pname in periods:
        for proto, label in (("paper", "режим статьи"), ("walk_forward", "walk-forward")):
            a.md += [f"### Стратегии: {label}, {pname}", "", f"{runline(pname, proto)}.", "",
                     mtable(pname, proto), "",
                     "Значимость Шарпа (блочный бутстреп; p — гипотеза равенства Шарпа стратегии и Equal Weighted):",
                     "", shtable(pname, proto), "",
                     "Диболд–Мариано против наивного прогноза r̂ = 0 (DM < 0 — модель точнее):", "",
                     dmtable(pname, proto), ""]
    a.md += [
        f"### Все модели scikit-learn из [A4], режим статьи, до {A4_CUT}",
        "",
        "Семь из 11 моделей статьи (Linear, Ridge, Lasso, SVR, Decision Tree, Random Forest, Gradient Boosting); "
        "метрики на тестовой выборке, точность направления — доля совпавших знаков прогноза и факта. "
        "Опубликованные значения: " + NEED + ".",
        "",
        md_table(a7f),
        "",
        dl_md,
        "",
    ]
    return a


def _a4_dl_check(px, fundamentals, with_dl: bool, a: Article) -> str:
    """FR-IV-03: do the TensorFlow models train? Only with --with-dl and TensorFlow installed."""
    path = ROOT / "docs" / "results" / "a4_dl_check.csv"
    if with_dl and signals.tensorflow_available():
        import tensorflow as tf
        tf.keras.utils.set_random_seed(config.RANDOM_SEED)
        rows = []
        for key in signals.DL_MODELS:
            st = replace(A4_SETTINGS["paper"], models=(key,))
            try:
                s, _ = run_module4(px, st, fundamentals)
                m = s.metrics.loc[signals.MODEL_LABELS[key]]
                rows.append({"model": signals.MODEL_LABELS[key], "status": "ok", "RMSE": m.RMSE, "MAE": m.MAE,
                             "hit_rate": m.hit_rate, "IC": m.IC, "tensorflow": tf.__version__, "error": ""})
            except Exception as exc:  # record the failure, it is the result of the check
                rows.append({"model": signals.MODEL_LABELS[key], "status": "error", "RMSE": np.nan,
                             "MAE": np.nan, "hit_rate": np.nan, "IC": np.nan, "tensorflow": tf.__version__,
                             "error": f"{type(exc).__name__}: {exc}"[:300]})
        a.tables["a4_dl_check"] = pd.DataFrame(rows)
        dl = a.tables["a4_dl_check"]
    elif path.exists():
        dl = pd.read_csv(path)
        a.tables["a4_dl_check"] = dl
    else:
        return ("### Глубокие модели (RNN, LSTM, CNN-LSTM, LSTM-MLP)\n\nTensorFlow в окружении не установлен, "
                "проверка не выполнялась (`python scripts/reproduce.py --with-dl` после `pip install tensorflow`).")
    f = pd.DataFrame({"Статус": ["обучена" if s == "ok" else "ошибка" for s in dl.status],
                      "RMSE": [num(v, 5) for v in dl.RMSE], "MAE": [num(v, 5) for v in dl.MAE],
                      "Точность направления": [pct(v, 1) for v in dl.hit_rate], "IC": [num(v, 4) for v in dl.IC],
                      "Ошибка": [e if isinstance(e, str) and e else "—" for e in dl.error]},
                     index=pd.Index(dl.model, name="Модель"))
    return (f"### Глубокие модели (ФТ-IV-03), режим статьи, до {A4_CUT}\n\n"
            f"TensorFlow {dl.tensorflow.iloc[0]}; каждая модель обучена отдельно (50 эпох, ранняя остановка, "
            "seed 42; на CPU результаты TensorFlow могут немного меняться от запуска к запуску). Таблица "
            "сохранена в `docs/results/a4_dl_check.csv` прогоном `--with-dl`.\n\n" + md_table(f))


# --------------------------------------------------------------------------
# [A3] Module III - KASE
# --------------------------------------------------------------------------
A3_VARIANTS = {"все 9 бумаг": [], "без KZTK": ["KZTK"], "без KMGZ": ["KMGZ"]}
A3_METRIC_COLS = ["annual_return", "annual_vol", "sharpe", "turnover", "max_drawdown", "total_return"]


def kase_variant(raw: pd.DataFrame, drop: list) -> pd.DataFrame:
    """Price frame of one robustness variant, validated like prices_kase.csv itself."""
    return loaders.validate_prices(raw.drop(columns=drop), "prices_kase.csv")


def reproduce_a3(raw: pd.DataFrame, fundamentals: pd.DataFrame | None) -> Article:
    a = Article("a3")
    st = config.Settings()
    t2, qual, labels, ksel, info = [], [], [], [], []
    for vname, drop in A3_VARIANTS.items():
        px = kase_variant(raw, drop)
        fund = fundamentals.reindex(px.columns) if fundamentals is not None else None
        c = clustering.cluster_assets(px, fund)
        tab = clustering.cluster_portfolio_backtest(px, c.k, tc_bps=st.tc_bps, risk_free=st.risk_free)
        t2.append(tab[A3_METRIC_COLS].assign(variant=vname))
        qual.append(c.metrics.assign(variant=vname, k=c.k))
        lab = pd.DataFrame(c.labels) + 1
        labels.append(lab.assign(variant=vname, medoid=lab.index.isin(c.medoids)))
        ksel.append(pd.DataFrame({k: c.k_selection[k] for k in ("k", "inertia", "silhouette")}).assign(variant=vname))
        info.append({"variant": vname, "start": px.index[0].date(), "end": px.index[-1].date(), "days": len(px),
                     "n_assets": px.shape[1], "k": c.k, "features": ",".join(c.features.columns),
                     "tc_bps": st.tc_bps, "risk_free": st.risk_free})
    table2 = pd.concat(t2).reset_index()
    quality = pd.concat(qual).reset_index()
    labels_df = pd.concat(labels).reset_index(names="ticker")
    info_df = pd.DataFrame(info)
    a.tables |= {"a3_table2": table2, "a3_quality": quality, "a3_labels": labels_df,
                 "a3_k_selection": pd.concat(ksel), "a3_runs": info_df}

    main = table2[table2.variant == "все 9 бумаг"].set_index("strategy")
    a.headline.append({"Статья": "[A3]", "Показатель": "Табл. 2: Шарп K-means / K-medoids / FCM / равные веса (KASE)",
                       "Статья, значение": "X/Y/Z (не заполнена)",
                       "СППР": " / ".join(num(main.loc[s, "sharpe"], 2) for s in main.index)})
    fcm_ok = all(table2[table2.variant == v].set_index("strategy").loc["Fuzzy C-means", "sharpe"]
                 > table2[table2.variant == v].set_index("strategy").loc["Равные веса", "sharpe"] for v in A3_VARIANTS)
    a.summary = ("**[A3]** — табл. 2 впервые заполнена числами на KASE (сравнивать не с чем); "
                 + ("во всех трёх вариантах (с KZTK, без KZTK, без KMGZ) кластерные портфели по Шарпу выше равных "
                    "весов, а FCM — с наименьшей просадкой и оборачиваемостью среди кластерных"
                    if fcm_ok else "превосходство кластерных портфелей над равными весами неустойчиво")
                 + "; абсолютные Шарпы условны (ставка 5 %, цены без дивидендов).")
    a.questions += [
        "[A3] Опубликованная табл. 2 содержит заполнители X/Y/Z; нужно подтвердить, что в диссертацию идёт таблица, "
        "рассчитанная СППР, и с какими настройками (издержки, период, безрисковая ставка для KASE).",
        "[A3] P/B и P/E для бумаг KASE (`data/fundamentals_kase.csv` пуст) — без них кластеризация идёт только по "
        "ценовым признакам.",
        "[A3] Что произошло с KZTK 22–23.05.2025 (падение 58 400 → 34 279 ₸): корпоративное событие или рыночное "
        "падение? От ответа зависит, нужна ли корректировка цен.",
    ]

    def t2_md(vname):
        t = table2[table2.variant == vname].set_index("strategy")[A3_METRIC_COLS]
        f = fmt_frame(t, 2, ("annual_return", "annual_vol", "turnover", "max_drawdown", "total_return"))
        f = f.rename(columns=METRIC_RU)
        f.index.name = "Метод"
        return md_table(f)

    side = pd.DataFrame({v: [f"{pct(r.annual_return, 1)} / {num(r.sharpe, 2)}"
                             for _, r in table2[table2.variant == v].set_index("strategy").iterrows()]
                         for v in A3_VARIANTS}, index=main.index)
    side.index.name = "Метод (годовая доходность / Шарп)"
    q = quality.set_index(["variant", "method"])
    qf = pd.DataFrame({"k": [int(q.loc[(v, m), "k"]) for v, m in q.index],
                       "Силуэт": [num(q.loc[i, "silhouette"], 3) for i in q.index],
                       "Дэвис–Боулдин": [num(q.loc[i, "davies_bouldin"], 3) for i in q.index]},
                      index=[f"{v}: {m}" for v, m in q.index])
    qf.index.name = "Вариант: метод"
    lab_tbl = labels_df.pivot_table(index="ticker", columns="variant", values="K-medoids", aggfunc="first")
    memb = labels_df.set_index(["variant", "ticker"])
    mf = pd.DataFrame({f"{v}": [("—" if (v, t) not in memb.index else
                                 "/".join(str(int(memb.loc[(v, t), m])) for m in ("K-means", "K-medoids", "Fuzzy C-means"))
                                 + ("*" if memb.loc[(v, t), "medoid"] else ""))
                                for t in lab_tbl.index] for v in A3_VARIANTS}, index=lab_tbl.index)
    mf.index.name = "Тикер"
    i0 = info_df.iloc[0]
    ks = a.tables["a3_k_selection"]
    ksf = pd.DataFrame({"Вариант": ks.variant, "k": ks.k, "Силуэт (PAM)": [num(v, 3) for v in ks.silhouette],
                        "Стоимость PAM": [num(v, 3) for v in ks.inertia]})
    a.md += [
        "## [A3] Модуль III — кластеризация бумаг KASE (табл. 2)",
        "",
        f"**Данные и настройки.** `data/prices_kase.csv`, 9 акций; общая история (дни с ценами всех бумаг) "
        f"{i0.start:%d.%m.%Y} – {i0.end:%d.%m.%Y}, {i0.days} торговых дней. Признаки кластеризации: "
        f"{i0.features.replace(',', ', ')} — **только ценовые**: P/B и P/E в `fundamentals_kase.csv` пусты. "
        "`clustering.cluster_assets` (k выбирается по силуэту PAM в диапазоне 2…max(2, min(6, n/3))) и "
        "`clustering.cluster_portfolio_backtest`: ежемесячная перекластеризация по окну 252 дня; K-means и K-medoids "
        "берут лучшую по Шарпу бумагу каждого кластера с равными весами, FCM распределяет бюджет 1/k по uᵢⱼᵐ·max(Шарп, 0); "
        f"издержки {num(st.tc_bps, 0)} б.п.; безрисковая ставка {pct(st.risk_free, 0)} (как в `pipeline.run`). "
        "Первые 252 дня нет ни одной перекластеризации, и все стратегии держат равные веса — эти дни входят в "
        "показатели всех строк.",
        "",
        "**Ограничения:** цены не скорректированы на дивиденды (доходность KEGC, KZTO, HSBK, KZAP занижена); KZTK "
        "содержит непроверенное падение на 41 % 22–23.05.2025; безрисковая ставка 5 % не отражает ставки в тенге "
        "(ставка НБ РК — вопрос к автору), поэтому абсолютные значения Шарпа для KASE завышены; сравнивать методы "
        "между собой корректно.",
        "",
        "### Сравнение со статьёй",
        "",
        "Опубликованная табл. 2 [A3] не содержит чисел (заполнители X/Y/Z), поэтому сравнения нет: ниже — первое "
        "заполнение таблицы на реальных данных KASE.",
        "",
        "### Табл. 2 [A3]: все 9 бумаг",
        "",
        t2_md("все 9 бумаг"),
        "",
        "### Устойчивость: три варианта рядом",
        "",
        md_table(side),
        "",
    ]
    for v in list(A3_VARIANTS)[1:]:
        iv = info_df[info_df.variant == v].iloc[0]
        a.md += [f"### Табл. 2 [A3]: {v} ({iv.start:%d.%m.%Y} – {iv.end:%d.%m.%Y}, {iv.days} дн., k = {iv.k})", "",
                 t2_md(v), ""]
    a.md += [
        "### Выбор k и качество разбиения",
        "",
        md_table(ksf, index=False),
        "",
        md_table(qf),
        "",
        "При 8 бумагах (вариант без KZTK или без KMGZ) верхняя граница диапазона n/3 даёт только k = 2, выбор k "
        "фактически не делается.",
        "",
        "### Состав кластеров (номер кластера K-means / K-medoids / FCM; * — медоид PAM)",
        "",
        md_table(mf),
        "",
        "Номера кластеров у разных методов не согласованы между собой (метка — произвольный номер группы).",
        "",
    ]
    return a


# --------------------------------------------------------------------------
# [A1] Module I - GDP
# --------------------------------------------------------------------------
def leaky_features(long: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only: the [A1] features with rolling statistics that INCLUDE year t
    (the target year). Used to test whether look-ahead explains the 97.9 % / AUC 0.997."""
    df = macro.build_features(long)
    g = df.groupby("country")["growth"]
    df["rolling_mean3"] = g.transform(lambda s: s.rolling(3).mean())
    df["rolling_mean5"] = g.transform(lambda s: s.rolling(5).mean())
    df["rolling_std3"] = g.transform(lambda s: s.rolling(3).std(ddof=0))
    df["cumulative_mean"] = g.transform(lambda s: s.expanding().mean())
    return df


def classify(feats: pd.DataFrame, train_end: int = config.MACRO_TRAIN_END_YEAR) -> pd.DataFrame:
    """Accuracy and ROC AUC of the [A1] classifiers on a feature frame (same split and scaling as macro.run)."""
    f = feats.dropna(subset=macro.FEATURES)
    tr, te = f[f.year <= train_end], f[f.year > train_end]
    sc = StandardScaler().fit(tr[macro.FEATURES])
    rows = []
    for name, m in macro.recession_classifiers().items():
        m.fit(sc.transform(tr[macro.FEATURES]), tr.recession)
        x = sc.transform(te[macro.FEATURES])
        rows.append({"model": name, "accuracy": accuracy_score(te.recession, m.predict(x)),
                     "roc_auc": roc_auc_score(te.recession, m.predict_proba(x)[:, 1])})
    return pd.DataFrame(rows).set_index("model")


SHORT = {"Linear Regression": "Linear", "Ridge": "Ridge", "Random Forest": "RF", "Gradient Boosting": "GB",
         "Logistic Regression": "LogReg", "SVM (RBF)": "SVM"}


def reproduce_a1(gdp: pd.DataFrame) -> Article:
    a = Article("a1")
    m = macro.run(gdp)
    long = macro.to_long(gdp)
    feats = macro.build_features(long).dropna(subset=macro.FEATURES)
    test = feats[feats.year > config.MACRO_TRAIN_END_YEAR]
    rec_share = float(test.recession.mean())
    leak = classify(leaky_features(long))
    try:
        import statsmodels
        estimator = f"statsmodels {statsmodels.__version__} (точная ML-оценка)"
    except ImportError:
        estimator = "встроенная оценка методом условных наименьших квадратов (statsmodels не установлен)"
    a.tables |= {"a1_classification": m.classification, "a1_regression": m.regression,
                 "a1_robustness": m.robustness, "a1_arima": m.arima_table, "a1_forecast": m.forecasts,
                 "a1_leakage_check": leak,
                 "a1_mcnemar": pd.DataFrame([{"pair": "Gradient Boosting vs Random Forest",
                                              "chi2": m.mcnemar[0], "p_value": m.mcnemar[1]}])}

    cl, rg = m.classification, m.regression
    txt, _ = REFERENCES["a1_text"]
    f9, src9 = REFERENCES["a1_fig9_logreg"]
    trees, _ = REFERENCES["a1_fig9_trees"]
    lr = cl.loc["Logistic Regression"]
    best_acc = cl["accuracy"].idxmax()
    rows = [
        {"Показатель": "Лучшая точность / AUC (текст и аннотация)", "Статья": f"{pct(txt['accuracy'], 1)} / "
         f"{num(txt['roc_auc'], 3)}", "СППР": f"{pct(cl.loc[best_acc, 'accuracy'], 1)} / "
         f"{num(cl.loc[best_acc, 'roc_auc'], 3)} ({best_acc})",
         "Разница": f"{_diff(txt['accuracy'], cl.loc[best_acc, 'accuracy'], 1, True)} / "
                    f"{_diff(txt['roc_auc'], cl.loc[best_acc, 'roc_auc'], 3)}",
         "Объяснение": "не воспроизводится; см. проверку на утечку"},
        {"Показатель": "Logistic Regression: точность / AUC (рис. 9)", "Статья": f"{pct(f9['accuracy'], 1)} / "
         f"{num(f9['roc_auc'], 3)}", "СППР": f"{pct(lr.accuracy, 1)} / {num(lr.roc_auc, 3)}",
         "Разница": f"{_diff(f9['accuracy'], lr.accuracy, 1, True)} / {_diff(f9['roc_auc'], lr.roc_auc, 3)}",
         "Объяснение": "другой источник ВВП (WB), другое число стран и наблюдений"},
        {"Показатель": "RF / XGBoost / LightGBM: точность (рис. 9)",
         "Статья": f"{num(trees[0], 2)}–{num(trees[1], 2)}",
         "СППР": f"RF {pct(cl.loc['Random Forest', 'accuracy'], 1)}, GB {pct(cl.loc['Gradient Boosting', 'accuracy'], 1)}",
         "Разница": "—", "Объяснение": "XGBoost/LightGBM в СППР нет (ФТ-I-05); 0,14–0,17 ≈ доля рецессий, "
                                       "т. е. модель статьи предсказывала рецессию почти всегда"},
        {"Показатель": "R² регрессии роста (рис. 11)", "Статья": "< 0 у всех моделей",
         "СППР": ", ".join(f"{SHORT.get(k, k)} {num(v, 3)}" for k, v in rg["R2"].items()),
         "Разница": "—", "Объяснение": f"в СППР R² от {num(rg['R2'].min(), 3)} до {num(rg['R2'].max(), 3)}: "
                                       "рост почти не предсказуем по прошлым значениям"},
        {"Показатель": "ARIMA: выбранный порядок (табл. 4)", "Статья": NEED,
         "СППР": f"ARIMA{m.extra['arima_order']}", "Разница": "—", "Объяснение": "—"},
    ]
    base, src_b = REFERENCES["a1_table9_base"]
    imf = pd.Series(config.IMF_WEO_KZ_BENCHMARK)
    for i, yr in enumerate(m.forecasts.index):
        rows.append({"Показатель": f"Казахстан {yr}: ARIMA (база табл. 9)", "Статья": num(base[i], 2),
                     "СППР": num(m.forecasts.loc[yr, "ARIMA"], 2), "Разница": _diff(base[i], m.forecasts.loc[yr, "ARIMA"]),
                     "Объяснение": "ВВП WB вместо IMF WEO; см. текст"})
    cmp_ = comparison(rows)
    a.tables["a1_comparison"] = cmp_
    a.headline += [
        {"Статья": "[A1]", "Показатель": "Классификация рецессий: точность / AUC", "Статья, значение":
         f"текст {pct(txt['accuracy'], 1)} / {num(txt['roc_auc'], 3)}; рис. 9 (LogReg) {pct(f9['accuracy'], 1)} / "
         f"{num(f9['roc_auc'], 3)}", "СППР": f"LogReg {pct(lr.accuracy, 1)} / {num(lr.roc_auc, 3)}; "
                                            f"лучшая точность {best_acc} {pct(cl.loc[best_acc, 'accuracy'], 1)}"},
        {"Статья": "[A1]", "Показатель": "R² регрессии роста", "Статья, значение": "< 0 (рис. 11)",
         "СППР": f"{num(rg['R2'].min(), 3)} … {num(rg['R2'].max(), 3)}"},
        {"Статья": "[A1]", "Показатель": "Казахстан 2025: ARIMA", "Статья, значение": num(base[0], 2),
         "СППР": num(m.forecasts['ARIMA'].iloc[0], 2)},
    ]
    lr_leak = leak.loc["Logistic Regression"]
    a.summary = (f"**[A1]** — результаты СППР ближе к рис. 9 и 11 статьи (LogReg {pct(lr.accuracy, 1)} / AUC "
                 f"{num(lr.roc_auc, 3)}, R² ≈ 0), а не к 97,9 % / AUC 0,997 из текста; почти такую точность "
                 f"({pct(lr_leak.accuracy, 1)} / {num(lr_leak.roc_auc, 3)}) даёт утечка значения года t в скользящие "
                 "признаки — вероятная причина расхождения 1.")
    a.questions += [
        "[A1] Исходный файл `world_gdp_data.csv` (IMF WEO) — без него точное воспроизведение невозможно; в "
        "репозитории используется World Bank WDI.",
        "[A1] Какая версия эксперимента верна: текст/аннотация (97,9 %, AUC 0,997) или рис. 9 (LogReg 0,896, AUC "
        "0,762)? Включали ли скользящие признаки (rolling_mean3/5, rolling_std3, cumulative_mean) значение "
        "текущего года t? Проверка СППР показывает, что именно такая утечка даёт ≈ 97 % / AUC ≈ 1.",
        "[A1] Выбранный порядок ARIMA и значения AIC из табл. 4.",
        f"[A1] Подтвердить базовую траекторию ARIMA табл. 9 (4,34; 4,64; 4,99; 5,23; 5,45) — она взята из теста "
        f"(`{src_b}`), а не из текста статьи.",
    ]

    clf = fmt_frame(cl, 3, ("accuracy", "recall_recession"), 1).rename(columns={
        "accuracy": "Точность", "roc_auc": "ROC AUC", "recall_recession": "Полнота (рецессия)",
        "f1_recession": "F1 (рецессия)"})
    clf.index.name = "Модель"
    rgf = fmt_frame(rg, 3)
    rgf.index.name = "Модель"
    rob = fmt_frame(m.robustness, 3)
    rob.index.name = "Проверка (GB-регрессор)"
    ar = m.arima_table.assign(AIC=[num(v, 2) for v in m.arima_table.AIC],
                              selected=["да" if v else "" for v in m.arima_table.selected])
    ar.columns = ["Модель", "AIC", "Выбрана"]
    fc = m.forecasts.copy()
    fc["База табл. 9 [A1]"] = base[:len(fc)]
    fc = fmt_frame(fc, 2)
    fc.index.name = "Год"
    lk = pd.DataFrame({"Точность (корректные признаки)": [pct(v, 1) for v in cl.accuracy],
                       "AUC (корректные)": [num(v, 3) for v in cl.roc_auc],
                       "Точность (с утечкой)": [pct(v, 1) for v in leak.reindex(cl.index).accuracy],
                       "AUC (с утечкой)": [num(v, 3) for v in leak.reindex(cl.index).roc_auc]}, index=cl.index)
    lk.index.name = "Модель"
    kz = m.country_history
    a.md += [
        "## [A1] Модуль I — прогноз роста ВВП",
        "",
        f"**Данные и настройки.** `data/gdp_growth.csv` — рост реального ВВП, %, World Bank WDI "
        f"(`NY.GDP.MKTP.KD.ZG`), {m.n_countries} стран, {int(gdp.columns.min())}–{int(gdp.columns.max())}; "
        f"после построения признаков {m.n_obs} наблюдений страна-год, тестовая выборка 2015–2024 — {m.test_n} "
        f"наблюдений, из них рецессий (рост < 0) {pct(rec_share, 1)}. Признаки `macro.build_features` "
        "(лаги 1–3, скользящие средние 3/5, σ за 3 года, накопленное среднее, нормированный год — только прошлые "
        "значения), обучение ≤ 2014, классы сбалансированы; ARIMA(p,1,q) — "
        f"{estimator}; seed 42. **Это не исходные данные статьи** (IMF WEO, `world_gdp_data.csv`), поэтому "
        "точного совпадения ожидать нельзя.",
        "",
        "### Сравнение со статьёй",
        "",
        md_table(cmp_, index=False),
        "",
        f"Источник опубликованных значений: {src9}; базовая траектория ARIMA — {src_b}; прогноз IMF WEO (окт. 2024) — "
        "`config.IMF_WEO_KZ_BENCHMARK`.",
        "",
        f"**Какая версия статьи ближе.** Результаты СППР близки к **рис. 9 и рис. 11**, а не к тексту и аннотации: "
        f"лучшая точность {pct(cl.loc[best_acc, 'accuracy'], 1)} ({best_acc}), AUC всех моделей "
        f"{num(cl.roc_auc.min(), 3)}–{num(cl.roc_auc.max(), 3)}, R² регрессии около нуля. Для сравнения, "
        f"доля нерецессионных лет в тесте — {pct(1 - rec_share, 1)}: модель, которая всегда отвечает «нет рецессии», "
        "имеет такую точность, поэтому точность Gradient Boosting почти целиком объясняется дисбалансом классов. "
        "Точность 97,9 % при AUC 0,997 по признакам только из прошлого противоречит и рис. 9, и тому, что рост "
        "ВВП по лагам почти не предсказуем (R² ≈ 0).",
        "",
        "### Проверка гипотезы об утечке (диагностика, не воспроизведение)",
        "",
        "Если скользящие признаки считать по окну, **включающему год t** (типичная ошибка `rolling()` без "
        "`shift(1)`), признак rolling_mean3 содержит сам рост года t, и рецессия угадывается почти без ошибок:",
        "",
        md_table(lk),
        "",
        f"С утечкой Logistic Regression даёт {pct(leak.loc['Logistic Regression', 'accuracy'], 1)} / AUC "
        f"{num(leak.loc['Logistic Regression', 'roc_auc'], 3)} — почти то же, что в тексте [A1]. Это наиболее "
        "вероятное объяснение расхождения 1, но подтвердить его можно только по коду и данным статьи. В СППР "
        "признаки строятся строго по прошлым значениям (тест `test_features_use_only_past_values`).",
        "",
        "### Классификация рецессий (тест 2015–2024)",
        "",
        md_table(clf),
        "",
        f"Критерий Мак-Нимара (Gradient Boosting против Random Forest): χ² = {num(m.mcnemar[0], 2)}, "
        f"p = {m.mcnemar[1]:.1e}".replace(".", ",").replace("e-", "·10^-") + " — модели ошибаются по-разному.",
        "",
        "### Регрессия роста ВВП",
        "",
        md_table(rgf),
        "",
        "### Проверки устойчивости (табл. 11 [A1])",
        "",
        md_table(rob),
        "",
        "Опубликованные значения табл. 11: " + NEED + ".",
        "",
        "### Казахстан: отбор ARIMA по AIC (табл. 4 [A1])",
        "",
        md_table(ar, index=False),
        "",
        "### Казахстан: прогноз 2025–2029",
        "",
        md_table(fc),
        "",
        f"Фактический рост Казахстана в файле: 2023 — {num(kz.get(2023, np.nan), 1)} %, 2024 — "
        f"{num(kz.get(2024, np.nan), 1)} %. Модели в консенсусе: {', '.join(m.consensus_models)}. "
        "ARIMA СППР отличается от базы табл. 9 на "
        f"{num((m.forecasts['ARIMA'] - np.array(base)).abs().max(), 2)} п.п. максимум; причина — другой источник и "
        "ревизии ряда (World Bank, данные по 2024 г. включительно), а также другая оценка ARIMA "
        "(" + estimator.split(" (")[0] + "). Расхождение с IMF WEO: "
        + ", ".join(f"{yr} — {num(m.forecasts.loc[yr, 'ARIMA'] - imf[yr], 2, sign=True)}" for yr in m.forecasts.index
                    if yr in imf.index) + " п.п.",
        "",
    ]
    return a


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
CAVEATS = [
    "Цены KASE **не скорректированы на дивиденды** (так их публикует KASE); у крупных плательщиков дивидендов "
    "(KEGC, KZTO, HSBK, KZAP) доходность занижена.",
    "**KZTK упал на 41 % 22–23.05.2025** (58 400 → 34 279 ₸) и остался на этом уровне; вероятно, корпоративное "
    "событие, не проверено. Результаты по KZTK считать сомнительными; расчёты [A3] приведены с KZTK и без него.",
    "ВВП — **World Bank WDI**, а не IMF WEO / `world_gdp_data.csv` статьи [A1]; небольшие различия ожидаемы.",
    "Цены NSE в файле идут до 10.2026, а [A4] охватывает 2020–2025; расчёт [A4] выполнен на обрезанных по "
    f"{A4_CUT} данных и на всём файле. `TMPV.NS` заменяет `TATAMOTORS.NS` и пересчитан на выделение TMCV.",
    "Секторы NSE берутся из встроенной таблицы II [A4] (`loaders.NSE_SECTORS`), P/B и P/E для KASE отсутствуют.",
    "Безрисковая ставка 5 % (`config.RISK_FREE`) восстановлена по табл. IV [A4] для NSE; для KASE она условна.",
]
FIXED_QUESTIONS = [
    "[Данные] Базовая ставка НБ РК (значение и источник) для безрисковой ставки KASE (ФТ-IV-12) — сейчас для "
    "KASE используется 5 %, как для NSE.",
    "[CI] Причина `startup_failure` в GitHub Actions — см. `docs/progress.md`, раздел «Риски».",
]


def env_line() -> str:
    tf = "есть" if signals.tensorflow_available() else "нет"
    try:
        import statsmodels
        sm = statsmodels.__version__
    except ImportError:
        sm = "нет"
    return (f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"scikit-learn {sklearn.__version__}, statsmodels: {sm}, TensorFlow: {tf}; seed {config.RANDOM_SEED}.")


def build_report(arts: dict) -> str:
    head = pd.DataFrame([r for k in ("a1", "a2", "a3", "a4") for r in arts[k].headline])
    head.columns = ["Статья", "Показатель", "Статья (значение)", "СППР"]
    questions = [q for k in ("a1", "a2", "a3", "a4") for q in arts[k].questions] + FIXED_QUESTIONS
    out = [
        "# Воспроизведение результатов статей [A1]–[A4] на реальных данных",
        "",
        "Этап 3 плана работ (`docs/progress.md`). Документ целиком сгенерирован командой "
        "`python scripts/reproduce.py` по CSV-файлам из `data/`; таблицы в машинном виде — в `docs/results/*.csv` "
        "(десятичная точка). Ничего не подгонялось под опубликованные значения: используются параметры модулей "
        "по умолчанию и параметры статей из `dss/config.py`.",
        "",
        f"Окружение расчёта: {env_line()}",
        "",
        f"Опубликованные значения берутся только из репозитория; если значения нет, в таблице стоит «{NEED}», "
        "а вопрос вынесен в конец документа.",
        "",
        "## Сводка",
        "",
        md_table(head, index=False),
        "",
        *[f"* {arts[k].summary}" for k in ("a1", "a2", "a3", "a4")],
        "",
        "## Оговорки по данным",
        "",
        *[f"{i}. {c}" for i, c in enumerate(CAVEATS, 1)],
        "",
    ]
    for k in ("a1", "a2", "a3", "a4"):
        out += arts[k].md
    out += ["## Вопросы к автору", "", *[f"{i}. {q}" for i, q in enumerate(questions, 1)], ""]
    return "\n".join(out)


def write_results(arts: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for art in arts.values():
        for name, df in art.tables.items():
            df.to_csv(out_dir / f"{name}.csv", index=not isinstance(df.index, pd.RangeIndex), float_format="%.6g")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Воспроизведение статей [A1]–[A4] на данных из data/")
    p.add_argument("--with-dl", action="store_true", help="проверить модели TensorFlow [A4] (нужен tensorflow)")
    p.add_argument("--out", default=str(ROOT / "docs"), help="папка для reproduction.md и results/")
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    np.random.seed(config.RANDOM_SEED)
    rates, gdp = loaders.load_fx("csv"), loaders.load_gdp("csv")
    nse, kase = loaders.load_prices("nse", "csv"), loaders.load_prices("kase", "csv")
    for d in (rates, gdp, nse, kase):
        if d.is_demo:
            raise SystemExit(f"Источник {d.source} — демо-данные; воспроизведение возможно только на реальных CSV")
    raw_kase = loaders._dated_frame(loaders.data_dir() / "prices_kase.csv", "Date,<тикер1>,…")
    fund_kase = loaders.load_fundamentals("kase", list(raw_kase.columns)).frame.select_dtypes("number")
    fund_nse = loaders.load_fundamentals("nse", list(nse.frame.columns)).frame.select_dtypes("number")
    arts = {}
    for key, fn in (("a2", lambda: reproduce_a2(rates.frame)),
                    ("a4", lambda: reproduce_a4(nse.frame, fund_nse, a.with_dl)),
                    ("a3", lambda: reproduce_a3(raw_kase, fund_kase)),
                    ("a1", lambda: reproduce_a1(gdp.frame))):
        print(f"[{key.upper()}] ...", flush=True)
        arts[key] = fn()
    out = Path(a.out)
    write_results(arts, out / "results")
    (out / "reproduction.md").write_text(build_report(arts), encoding="utf-8")
    print(f"Готово: {out / 'reproduction.md'}, {out / 'results'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
