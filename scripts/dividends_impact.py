"""Tables 4-5 before and after the full dividend correction: docs/e2e_dividends_impact.md.

Compares the results committed under tag v1.0-paper (corporate_actions.csv with the KZTK
dividend of May 2025 only) with e2e_table4_div / e2e_table5_div (all dividends of
data/dividends_kase.csv, 108 events). Run after::

    python scripts/end_to_end.py --end 2026-10-09 --tag table4_div
    python scripts/end_to_end.py --modules 12 --start 2020-01-06 --end 2026-10-09 --tag table5_div
    python scripts/dividends_impact.py
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dss.numfmt import num, pct  # noqa: E402

RES = ROOT / "docs" / "results"
TAG = "v1.0-paper"


def paper(name: str) -> pd.DataFrame:
    txt = subprocess.check_output(["git", "show", f"{TAG}:docs/results/{name}"], cwd=ROOT).decode()
    return pd.read_csv(io.StringIO(txt), index_col=0)


def paper_meta(name: str) -> dict:
    txt = subprocess.check_output(["git", "show", f"{TAG}:docs/results/{name}"], cwd=ROOT).decode()
    return pd.read_csv(io.StringIO(txt), index_col=0, header=None).iloc[:, 0].to_dict()


def meta(df: pd.DataFrame) -> dict:
    return df.iloc[:, 0].to_dict()


def order(t: pd.DataFrame) -> str:
    return " > ".join(f"«{n}»" for n in t["sharpe"].sort_values(ascending=False).index)


def table(old: pd.DataFrame, new: pd.DataFrame, ref: str) -> list[str]:
    out = [f"| Конфигурация | Доходность, % ({TAG} → все дивиденды) | Шарп | Макс. просадка, % | "
           f"Разность Шарпа к «{ref}» | p (Холм) |", "|---|---|---|---|---|---|"]
    for name in old.index:
        o, n = old.loc[name], new.loc[name]
        diff = f"{num(o['diff'])} → {num(n['diff'])}" if pd.notna(o["diff"]) else "—"
        ph = f"{num(o['p_holm'], 3)} → {num(n['p_holm'], 3)}" if pd.notna(o["p_holm"]) else "—"
        out.append(f"| {name} | {pct(o['annual_return'])} → {pct(n['annual_return'])} | {num(o['sharpe'])} → "
                   f"{num(n['sharpe'])} | {pct(o['max_drawdown'])} → {pct(n['max_drawdown'])} | {diff} | {ph} |")
    return out


def main() -> int:
    o4, n4 = paper("e2e_table4_table.csv"), pd.read_csv(RES / "e2e_table4_div_table.csv", index_col=0)
    o5, n5 = paper("e2e_table5_table.csv"), pd.read_csv(RES / "e2e_table5_div_table.csv", index_col=0)
    m4 = meta(pd.read_csv(RES / "e2e_table4_div_meta.csv", index_col=0, header=None))
    m5 = meta(pd.read_csv(RES / "e2e_table5_div_meta.csv", index_col=0, header=None))
    rank_same4 = list(o4["sharpe"].sort_values().index) == list(n4["sharpe"].sort_values().index)
    rank_same5 = list(o5["sharpe"].sort_values().index) == list(n5["sharpe"].sort_values().index)
    p_full = n4.loc["EW + KZT", "p_value"], n4.loc["EW + KZT", "p_holm"]
    lines = [
        "# Влияние полной поправки на дивиденды на таблицы 4–5", "",
        f"Сравнение результатов, зафиксированных тегом `{TAG}` (в `corporate_actions.csv` был один дивиденд — "
        "KZTK, май 2025 г.), с пересчётом на обновлённых данных: все дивиденды из `data/dividends_kase.csv` "
        "(108 событий в `corporate_actions.csv`, включая дробление HSBK 1:10 в 2012 г.), цены KASE по торговым "
        "дням с 2008 г., базовая ставка НБ РК и ставка T-bill с 2008 г. Дата отсечки реестра — правило "
        "`loaders.kase_ex_date`. Код схемы бэктеста не менялся; файлы `docs/results/e2e_table4_div_*` и "
        "`e2e_table5_div_*`. Сгенерировано `python scripts/dividends_impact.py`.", "",
        "## Таблица 4. Умеренный профиль, 7 бумаг KASE", "",
        f"Период {m4['start']} – {m4['end']}, {m4['n_days']} торговых дней, {m4['n_rebalancing']} ребалансировок, "
        f"блок {m4['block']} дн. По умолчанию `--start 2023-01-16`; ранги Модуля IV начинаются 17.01.2023, "
        "и старт теперь сдвигается на эту дату с предупреждением (раньше скрипт завершался ошибкой). "
        f"IC ансамбля Модуля IV: {num(float(m4['ic_ensemble']), 3)} (в `{TAG}` — "
        f"{num(float(paper_meta('e2e_table4_meta.csv')['ic_ensemble']), 3)}); ранги изменились, потому что "
        "признаки Модуля IV считаются по скорректированным ценам.",
        "", *table(o4, n4, "Full DSS"), "",
        "## Таблица 5. Модули I–II, равные веса в рисковой части", "",
        f"Период {m5['start']} – {m5['end']}, {m5['n_days']} торговых дней, блок {m5['block']} дн.", "",
        *table(o5, n5, "EW + KZT"), "",
        "## Выводы", "",
        f"* Порядок конфигураций по Шарпу (от лучшей): таблица 4 — {order(o4)} в `{TAG}`, {order(n4)} сейчас"
        f"{' (не изменился)' if rank_same4 else ''}; таблица 5 — {order(o5)} в `{TAG}`, {order(n5)} сейчас"
        f"{' (не изменился)' if rank_same5 else ''}.",
        "* Дивидендная поправка поднимает Шарп всех конфигураций с акциями примерно на 0,6–1,1: в `" + TAG +
        "` доходность акций считалась без большинства дивидендов, а дивидендная доходность бумаг KASE7 высока.",
        f"* Старт таблицы 4 — {m4['start']} (первая дневная доходность после ребалансировки 17.01.2023), "
        f"на один торговый день позже, чем в `{TAG}`.",
        f"* Полная СППР по-прежнему уступает «EW + KZT» по Шарпу, но разность больше не значима: p = "
        f"{num(p_full[0], 3)} без поправки, {num(p_full[1], 3)} с поправкой Холма (в `{TAG}` — "
        f"{num(o4.loc['EW + KZT', 'p_value'], 3)} и {num(o4.loc['EW + KZT', 'p_holm'], 3)}). "
        "Значимым остаётся проигрыш полной СППР конфигурации без валютного модуля.",
        "* Вывод статьи о том, что валютное правило снижает Шарп в спокойный период, сохраняется; вывод "
        "«полная СППР значимо хуже простого EW + KZT» на новых данных не подтверждается. Решение о правке "
        "статьи (она на рецензии) — за авторами.", "",
        "## Ограничения данных", "",
        "* `reserves_usd_bn` в `data/macro_monthly_kz.csv` пуст: резервы НБ РК не загружены, показатель не "
        "используется.",
        "* KEI (краткосрочный экономический индикатор) как индекс к соответствующему периоду прошлого года "
        "есть только с 2021 г.; номинальный ряд `kei_nominal_bn_kzt` — с 2009 г. Метка рецессии y_rec (раздел 7 "
        "ТЗ) до 2021 г. строится только по ВВП.",
        "* Колонки `published_*` (даты публикации) в `macro_monthly_kz.csv` заданы по допущениям о графике "
        "БНС и НБ РК, а не по фактическим датам релизов.", ""]
    (ROOT / "docs" / "e2e_dividends_impact.md").write_text("\n".join(lines), encoding="utf-8")
    print("written docs/e2e_dividends_impact.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
