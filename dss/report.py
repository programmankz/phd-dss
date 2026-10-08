"""Self-contained HTML decision report (charts embedded, no external files)."""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import pandas as pd

from . import config, plots

CSS_PATH = Path(__file__).resolve().parents[1] / "webapp" / "static" / "style.css"


def _tbl(df: pd.DataFrame, d: int = 3, pct=()) -> str:
    def f(c, v):
        if isinstance(v, float):
            return f"{v * 100:.1f}%" if c in pct else f"{v:,.{d}f}".replace(",", " ")
        return html.escape(str(v))
    head = "<th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr><th>" + html.escape(str(i)) + "</th>" +
                   "".join(f"<td class='num'>{f(c, r[c])}</td>" for c in df.columns) + "</tr>"
                   for i, r in df.iterrows())
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def _img(b64: str, alt: str) -> str:
    return f"<figure class='fig panel'><img alt='{alt}' src='data:image/png;base64,{b64}'></figure>"


def build_report(r) -> str:
    d, st = r.decision, r.settings
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    steps = "".join(
        f"<div class='link {c}'><div class='hd'><span class='rn {c}'>{s['module']}</span><b>{html.escape(s['title'])}</b></div>"
        f"<ul>{''.join(f'<li>{html.escape(x)}</li>' for x in s['facts'])}</ul><div class='res'>{html.escape(s['result'])}</div></div>"
        for s, c in zip(d.steps, ("m1", "m2", "m3", "m4")))
    out = (f"<div class='link out'><div class='hd'><span class='rn m0'>✓</span><b>Решение</b></div>"
           f"<div class='big'>{d.risky_share:.0%}</div><div class='cap'>в рисковых активах</div>"
           f"<div class='row'><span>USD-хедж безрисковой части</span><b>{d.usd_hedge_share:.0%}</b></div>"
           f"<div class='row'><span>Ожид. волатильность</span><b>{d.exp_vol:.1%}</b></div>"
           f"<div class='row'><span>VaR 95%, 1 мес.</span><b>{d.var95_1m:.1%}</b></div></div>")
    alloc = d.allocation[d.allocation.weight > 0.0005][["weight", "signal", "cluster"]]
    alloc.columns = ["Вес", "Сигнал s", "Кластер"]
    demo = ("<div class='demo'><b>Демо-режим.</b> Часть данных синтетическая; результаты не являются "
            "эмпирическими выводами.</div>") if r.is_demo else ""
    src = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in r.sources.items())
    warns = "".join(f"<div class='warn'>{html.escape(w)}</div>" for w in d.warnings)
    m, f, c, s, b = r.macro, r.fx, r.clusters, r.signals, r.backtest
    body = f"""
<main style="max-width:1180px;margin:0 auto">
{demo}
<h1>Отчёт СППР: рекомендация по инвестиционному портфелю</h1>
<p class="lede">Сформирован {dt.datetime.now():%d.%m.%Y %H:%M}. Профиль «{d.profile.label}», сценарий
{html.escape(config.SCENARIOS[d.scenario]['label'])}, модели сигналов: {', '.join(st.models)}, ребалансировка {st.rebalance},
издержки {st.tc_bps:.0f} б.п.</p>
<section class="chain">{steps}{out}</section>
{warns}
<h2>Рекомендуемая структура</h2>
<div class="grid g-7-5">{_img(plots.allocation(d.allocation), 'веса')}<div>{_tbl(alloc, 2, pct=('Вес',))}</div></div>
<h2><span class="rn m1">I</span> Макропрогноз</h2>
{_img(plots.kz_forecast(m, st.scenario), 'прогноз')}
<div class="grid g2"><div>{_tbl(m.classification)}</div><div>{_tbl(m.regression)}</div></div>
<h2><span class="rn m2">II</span> Валютные риски</h2>
{_img(plots.fx_rates(f.rates, f.shocks), 'курсы')}
<div class="grid g2"><div>{_tbl(f.regression)}</div><div>{_tbl(f.risk_components)}</div></div>
<h2><span class="rn m3">III</span> Кластеризация</h2>
<div class="grid g2">{_img(plots.asset_clusters(c), 'кластеры')}<div>{_tbl(c.metrics)}{_tbl(r.cluster_table, 2, pct=('total_return', 'annual_return', 'annual_vol', 'max_drawdown', 'turnover'))}</div></div>
<h2><span class="rn m4">IV</span> Сигналы и бэктест</h2>
{_img(plots.equity_curves(b.curves), 'бэктест')}
<div class="grid g2"><div>{_tbl(b.metrics, 2, pct=('total_return', 'annual_return', 'annual_vol', 'max_drawdown', 'turnover'))}</div><div>{_tbl(s.metrics, 4)}</div></div>
<h2>Источники данных</h2><table class="src"><tbody>{src}</tbody></table>
<p class="note">Отчёт носит аналитический характер и не является индивидуальной инвестиционной рекомендацией.</p>
</main>"""
    return f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Отчёт СППР</title><style>{css}</style></head><body>{body}</body></html>"
