"""Export of the result tables of all modules (Sec. 11 of the spec).

An ``.xlsx`` workbook with one sheet per table when ``openpyxl`` is installed
(optional dependency); otherwise a ``.zip`` of CSV files in UTF-8 with BOM, so
that Excel opens Cyrillic text correctly. Shares stay fractions (0.25 = 25 %)
so the numbers can be reused in calculations.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from . import config

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def excel_available() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except ImportError:
        return False


def tables(r) -> dict:
    """Sheet name -> DataFrame for a :class:`dss.pipeline.RunResult`."""
    from .pipeline import rule_sensitivity, significance_tables
    st, d, m, f, c, s = r.settings, r.decision, r.macro, r.fx, r.clusters, r.signals
    sens = rule_sensitivity(r)
    sig_sharpe, sig_dm = significance_tables(r)
    params = {"Профиль": config.PROFILES[st.profile].label, "Сценарий": config.SCENARIOS[st.scenario]["label"],
              "Рынок": st.universe, "Модели": ", ".join(st.models), "Протокол": config.PROTOCOLS[st.protocol]["label"],
              "Ребалансировка": st.rebalance, "Издержки, б.п.": st.tc_bps, "Ковариация": st.cov_method,
              "Нечёткие лимиты": st.use_fuzzy_caps, "Лимиты в бэктесте": st.backtest_caps,
              "Безрисковая ставка": st.risk_free, "Демо-режим": r.is_demo,
              "Доля рисковых активов": d.risky_share, "USD-хедж безрисковой части": d.usd_hedge_share,
              "Ожидаемая волатильность": d.exp_vol, "VaR 95% 1 мес.": d.var95_1m}
    feats = c.features.copy()
    for name, lab in c.labels.items():
        feats[f"кластер {name}"] = lab.reindex(feats.index) + 1
    out = {
        "Параметры и итог": pd.Series(params, name="Значение").to_frame(),
        "Решение": d.allocation,
        "Цепочка решения": pd.DataFrame([{"Модуль": x["module"], "Шаг": x["title"], "Факты": "; ".join(x["facts"]),
                                          "Расчёт": "; ".join(x["trail"]), "Результат": x["result"]}
                                         for x in d.steps]),
        "Предупреждения": pd.DataFrame({"Сообщение": r.data_errors + d.warnings}),
        "Источники данных": pd.Series(r.sources, name="Источник").to_frame(),
        "I Классификация": m.classification,
        "I Регрессия": m.regression,
        "I Устойчивость": m.robustness,
        "I Важность признаков": m.importance.to_frame("importance"),
        "I ARIMA отбор": m.arima_table,
        "I Прогнозы": m.forecasts,
        "I Сценарии": pd.DataFrame({config.SCENARIOS[k]["label"]: m.scenario_path(k) for k in config.SCENARIOS}),
        "II Табл. 1 [A2]": f.stats,
        "II Корреляции": f.corr,
        "II Тренды": f.regression,
        "II Индекс риска": f.risk_components,
        "II Шоки": f.shocks,
        "III Качество": c.metrics,
        "III Признаки и метки": feats,
        "III Матрица U": c.U,
        "III Табл. 2 [A3]": r.cluster_table,
        "IV Модели": s.metrics,
        "IV Текущие сигналы": s.latest,
        "IV Стратегии": r.backtest.metrics,
        "IV Значимость Шарпа": sig_sharpe,
        "IV Диболд–Мариано": sig_dm,
        "Интеграция чувствит.": sens["coefficients"],
        "Интеграция монотонн.": sens["monotonicity"],
        "Интеграция сетка": sens["risk_grid"],
    }
    return {_sheet_name(k): v for k, v in out.items()}


def _sheet_name(name: str) -> str:
    return re.sub(r"[\[\]:*?/\\]", "", name)[:31]          # Excel limits


def to_xlsx(frames: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in frames.items():
            df.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2"
            for col in ws.columns:                            # readable column widths
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    return buf.getvalue()


def to_zip(frames: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (name, df) in enumerate(frames.items(), 1):
            zf.writestr(f"{i:02d} {name}.csv", df.to_csv().encode("utf-8-sig"))
    return buf.getvalue()


def export(r, fmt: str | None = None) -> tuple[bytes, str, str]:
    """(content, file extension, MIME type); ``fmt`` - "xlsx", "zip" or None (best available)."""
    fmt = fmt or ("xlsx" if excel_available() else "zip")
    frames = tables(r)
    if fmt == "xlsx":
        return to_xlsx(frames), "xlsx", XLSX
    return to_zip(frames), "zip", "application/zip"
