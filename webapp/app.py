"""Web interface of the DSS (Flask).

Run:  python -m webapp.app            -> http://127.0.0.1:8050
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request
from werkzeug.exceptions import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import config, numfmt, pipeline, plots  # noqa: E402
from dss.report import build_report  # noqa: E402
from dss.signals import MODEL_LABELS, tensorflow_available  # noqa: E402

app = Flask(__name__)

MODEL_SETS = {
    "fast": ("ridge", "rf", "gb"),
    "sklearn": ("linear", "ridge", "lasso", "svr", "dt", "rf", "gb", "mlp"),
    "full": ("linear", "ridge", "lasso", "svr", "dt", "rf", "gb", "mlp", "rnn", "lstm", "cnn_lstm", "lstm_mlp"),
}
REBALANCE = {"D": "ежедневно", "W": "еженедельно", "M": "ежемесячно"}


def settings_from_request() -> tuple[config.Settings, dict]:
    a = request.args
    raw = {
        "profile": a.get("profile", "moderate"),
        "scenario": a.get("scenario", "base"),
        "universe": a.get("universe", "kase"),
        "models": a.get("models", "fast"),
        "rebalance": a.get("rebalance", "W"),
        "tc": a.get("tc", "10"),
        "cov": a.get("cov", "ledoit_wolf"),
        "fuzzy": a.get("fuzzy", "0"),
        "protocol": a.get("protocol", "walk_forward"),
        "caps": a.get("caps", "1"),
    }
    if raw["profile"] not in config.PROFILES:
        raw["profile"] = "moderate"
    if raw["scenario"] not in config.SCENARIOS:
        raw["scenario"] = "base"
    if raw["models"] not in MODEL_SETS:
        raw["models"] = "fast"
    if raw["rebalance"] not in REBALANCE:
        raw["rebalance"] = "W"
    if raw["protocol"] not in config.PROTOCOLS:
        raw["protocol"] = "walk_forward"
    raw["caps"] = "0" if raw["caps"] == "0" else "1"
    try:
        tc = float(raw["tc"])
    except ValueError:
        tc = 10.0
    st = config.Settings(profile=raw["profile"], scenario=raw["scenario"], universe=raw["universe"],
                         models=MODEL_SETS[raw["models"]], rebalance=raw["rebalance"], tc_bps=tc,
                         cov_method=raw["cov"] if raw["cov"] in ("sample", "ledoit_wolf") else "ledoit_wolf",
                         use_fuzzy_caps=raw["fuzzy"] == "1", protocol=raw["protocol"],
                         backtest_caps=raw["caps"] == "1")
    return st, raw


# ------------------------------------------------------------------ helpers
def fmt(v, kind="num", d=2):
    """Russian number format (dss.numfmt); non-numeric values are shown as they are."""
    if v is None or isinstance(v, (int, float, np.integer, np.floating)):
        return numfmt.pct(v, d) if kind == "pct" else numfmt.num(v, d)
    return str(v)


def table(df: pd.DataFrame, pct=(), d=3, index_name=None, highlight=None) -> str:
    """DataFrame -> HTML table with Russian number formatting."""
    cols = list(df.columns)
    esc = html.escape
    head = f"<th>{esc(str(index_name or (df.index.name or '')))}</th>" + "".join(f"<th>{esc(str(c))}</th>" for c in cols)
    rows = []
    for idx, r in df.iterrows():
        cls = ' class="hl"' if highlight is not None and idx == highlight else ""
        cells = "".join(
            f'<td class="num">{fmt(r[c], "pct" if c in pct else "num", 1 if c in pct else d)}</td>'
            if isinstance(r[c], (int, float, np.integer, np.floating)) else f"<td>{esc(str(r[c]))}</td>"
            for c in cols)
        rows.append(f"<tr{cls}><th scope='row'>{esc(str(idx))}</th>{cells}</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


app.jinja_env.globals.update(fmt=fmt, table=table, config=config, MODEL_LABELS=MODEL_LABELS,
                             REBALANCE=REBALANCE, MODEL_SETS=MODEL_SETS)


def context(page: str):
    st, raw = settings_from_request()
    res = pipeline.run(st)
    qs = "&".join(f"{k}={v}" for k, v in raw.items())
    paper_qs = "&".join(f"{k}={v}" for k, v in (raw | config.PAPER_PRESET).items())
    return {"page": page, "r": res, "st": st, "raw": raw, "qs": qs, "paper_qs": paper_qs,
            "tf": tensorflow_available()}


@app.route("/")
def decision():
    c = context("decision")
    r = c["r"]
    c["img_alloc"] = plots.allocation(r.decision.allocation)
    a = r.decision.allocation
    a = a[a["weight"] > 0.0005]
    c["alloc_table"] = pd.DataFrame({"Вес": a["weight"], "Сигнал s": a["signal"], "Кластер": a["cluster"]})
    c["img_kz"] = plots.kz_forecast(r.macro, r.settings.scenario)
    sens = pipeline.rule_sensitivity(r)
    grid = sens["risk_grid"].copy()
    grid.index = [fmt(float(v), "pct", 0) for v in grid.index]
    c["sens"], c["sens_grid"] = sens, grid
    return render_template("decision.html", **c)


@app.route("/macro")
def macro_page():
    c = context("macro")
    r = c["r"]
    c["img_kz"] = plots.kz_forecast(r.macro, r.settings.scenario)
    c["img_imp"] = plots.importance(r.macro.importance)
    c["img_static"] = plots.country_clusters(r.macro.clusters_static)
    c["img_dtw"] = plots.dtw_profiles(r.macro.dtw_profiles) if len(r.macro.dtw_profiles) else None
    sens_in = {k: float(request.args.get(f"s_{k}", v["baseline"])) for k, v in config.SENSITIVITY.items()}
    from dss.macro import sensitivity
    c["sens_in"], c["sens"] = sens_in, sensitivity(sens_in)
    sc = pd.DataFrame({k: r.macro.scenario_path(k) for k in config.SCENARIOS})
    sc.columns = [config.SCENARIOS[k]["label"] for k in sc.columns]
    sc.index.name = "Год"
    c["scen"] = sc
    return render_template("macro.html", **c)


@app.route("/fx")
def fx_page():
    c = context("fx")
    r = c["r"]
    c["img_rates"] = plots.fx_rates(r.fx.rates, r.fx.shocks)
    c["img_corr"] = plots.heatmap(r.fx.corr, "Корреляции Пирсона (вкл. дату)")
    c["img_r2"] = plots.r2_bars(r.fx.regression)
    c["img_risk"] = plots.fx_risk(r.fx.risk_components, r.fx.fx_risk)
    return render_template("fx.html", **c)


@app.route("/clusters")
def clusters_page():
    c = context("clusters")
    r = c["r"]
    c["img_pca"] = plots.asset_clusters(r.clusters)
    c["img_u"] = plots.memberships(r.clusters.U)
    c["img_k"] = plots.k_selection(r.clusters.k_selection)
    c["img_corr"] = plots.heatmap(r.prices.pct_change().corr().round(2), "Корреляции доходностей", "Purples")
    cl = r.clusters
    ft = cl.features.rename(columns={"ann_return": "Доходн., год", "ann_vol": "Волат., год", "sharpe": "Шарп",
                                     "max_dd": "Макс. просадка", "beta": "Бета", "mom_6m": "Импульс 6 мес."})
    for name, col in (("K-medoids", "K-medoids"), ("K-means", "K-means"), ("Fuzzy C-means", "FCM")):
        ft[col] = [f"К{int(v) + 1}" for v in cl.labels[name]]
    c["feat_table"] = ft
    return render_template("clusters.html", **c)


@app.route("/signals")
def signals_page():
    c = context("signals")
    r = c["r"]
    c["img_curves"] = plots.equity_curves(r.backtest.curves)
    c["img_ranks"] = plots.rank_signals(r.signals.latest)
    c["img_ic"] = plots.model_ic(r.signals.metrics)
    c["sig_sharpe"], c["sig_dm"] = pipeline.significance_tables(r, fmt=lambda v: fmt(v, "num", 2))
    return render_template("signals.html", **c)


@app.route("/data")
def data_page():
    c = context("data")
    from dss.data.loaders import data_dir
    c["data_dir"] = str(data_dir().resolve())
    return render_template("data.html", **c)


@app.route("/report")
def report():
    c = context("report")
    html = build_report(c["r"])
    return Response(html, mimetype="text/html",
                    headers={"Content-Disposition": "attachment; filename=dss_report.html"})


@app.route("/refresh")
def refresh():
    pipeline.clear_cache()
    return decision()


@app.errorhandler(Exception)
def internal_error(exc):
    """Unexpected failure: a readable cause and action instead of a bare 500 page."""
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("request failed")
    body = (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Ошибка расчёта — СППР</title>"
            f"<link rel='stylesheet' href='/static/style.css'></head><body><main style='max-width:760px;margin:40px auto'>"
            f"<h1>Не удалось выполнить расчёт</h1>"
            f"<div class='warn' role='alert'>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</div>"
            f"<p>Что сделать: проверьте файлы на странице «Данные», затем "
            f"<a href='/refresh'>сбросьте кэш и пересчитайте</a>. Если ошибка повторяется, запустите "
            f"<code>python -m dss.cli decide</code> — подробности будут в консоли.</p></main></body></html>")
    return Response(body, status=500, mimetype="text/html")


def main():
    port = int(os.environ.get("PORT", "8050"))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)


if __name__ == "__main__":
    main()
