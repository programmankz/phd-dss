"""System tests: the full pipeline on the demo set, isolation of a broken data
file (NFR-06) and the web screens (HTTP 200, readable errors).

The first test runs all modules from scratch (about 30-60 s); the rest reuse
its cache. Run with pytest (``pytest -q``) or ``python tests/run.py``.
"""
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loaders import data_dir, prices_csv  # noqa: E402

from dss import config, pipeline  # noqa: E402

_CACHE = Path(tempfile.mkdtemp(prefix="dss-test-cache-"))
PAGES = ["/", "/macro", "/fx", "/clusters", "/signals", "/data", "/report"]


@contextmanager
def isolated(files: dict | None = None):
    """Run against a temporary data directory and a test-only cache."""
    old = pipeline.CACHE_DIR
    pipeline.CACHE_DIR = _CACHE
    try:
        with data_dir(files or {}) as d:
            yield d
    finally:
        pipeline.CACHE_DIR = old


def test_full_pipeline_on_demo_data():
    with isolated():
        r = pipeline.run(config.Settings())
    d = r.decision
    assert r.is_demo and r.data_errors == []
    assert all(v.startswith("DEMO") for k, v in r.sources.items() if "Фундамент" not in k)
    assert [s["module"] for s in d.steps] == ["I", "II", "III", "IV"]
    assert np.isclose(d.allocation["weight"].sum(), 1.0)
    assert (d.allocation["weight"] >= -1e-9).all()
    prof = config.PROFILES["moderate"]
    assert prof.min_risky_share <= d.risky_share <= prof.max_risky_share
    assert np.isclose(d.risky_weights.sum(), 1.0) and d.risky_weights.max() <= prof.max_asset_weight + 1e-6
    assert 0 <= d.usd_hedge_share <= 0.80 and d.var95_1m > 0
    assert {"Signal Weighted", "Equal Weighted"} <= set(r.backtest.metrics.index)
    # FR-IV-13: significance of every strategy and every forecasting model
    assert list(r.sharpe_tests.index) == list(r.backtest.daily.columns)
    assert set(r.forecast_tests.index) == set(r.signals.metrics.index[:-1])
    sens = pipeline.rule_sensitivity(r)
    assert np.isclose(sens["coefficients"]["×1"].iloc[0], d.risky_share)


def test_profile_change_reuses_module_results():
    with isolated():
        a = pipeline.run(config.Settings(profile="moderate", scenario="base"))
        t0 = time.perf_counter()
        b = pipeline.run(config.Settings(profile="conservative", scenario="stress"))
        elapsed = time.perf_counter() - t0
    assert a.macro is b.macro and a.fx is b.fx and a.signals is b.signals      # served from cache
    assert b.decision.risky_share < a.decision.risky_share                       # monotone in risk
    # NFR-03: only the integration step re-runs; the profile-dependent backtest waits for its screen
    assert "backtest" not in b.__dict__ and elapsed < 5, f"{elapsed:.1f} s"


def test_paper_protocol_reproduces_the_a4_setup():
    st = config.Settings(protocol="paper", cov_method="sample", rebalance="D", tc_bps=0, backtest_caps=False)
    with isolated():
        r = pipeline.run(st)
        wf = pipeline.run(config.Settings())
    share = (r.prices.index < r.signals.test_start).mean()
    assert 0.78 <= share <= 0.85                         # single chronological 80/20 split
    assert r.signals.test_start > wf.signals.test_start and r.signals is not wf.signals
    assert r.backtest.daily.index[0] >= r.signals.test_start and len(r.backtest.daily) < len(wf.backtest.daily)
    assert r.backtest.weights_last.to_numpy().max() <= 1 + 1e-9
    assert "80/20" in r.decision.steps[3]["facts"][1]


def test_broken_file_does_not_stop_other_modules():
    files = {"prices_kase.csv": prices_csv().replace("Date,", "Дата,", 1),
             "gdp_growth.csv": "country,1990\nChile,2.0\n"}
    with isolated(files):
        r = pipeline.run(config.Settings())
    assert len(r.data_errors) == 2 and r.is_demo
    joined = " ".join(r.data_errors)
    assert "prices_kase.csv: нет столбца Date" in joined and "gdp_growth.csv" in joined
    assert "демо-данные" in joined
    assert r.sources["Цены (Модули III–IV)"].startswith("DEMO")
    assert np.isclose(r.decision.allocation["weight"].sum(), 1.0)


def _client():
    from webapp.app import app
    app.config["TESTING"] = False        # exercise the production error handler
    return app.test_client()


def test_all_web_screens_return_200():
    c = _client()
    with isolated():
        for url in PAGES:
            resp = c.get(url)
            assert resp.status_code == 200, f"{url}: {resp.status_code}"
            assert "Демо-режим" in resp.get_data(as_text=True), url
        assert "Чувствительность правил интеграции" in c.get("/").get_data(as_text=True)
        assert "Диболда–Мариано" in c.get("/signals").get_data(as_text=True)
        resp = c.get("/?profile=aggressive&scenario=bull&rebalance=M&tc=abc&models=bogus&protocol=x&caps=?")
        assert resp.status_code == 200                       # bad query values fall back to defaults
        home = c.get("/").get_data(as_text=True)
        assert "Параметры статьи [A4]" in home and "protocol=paper" in home and "caps=0" in home


def test_web_shows_data_error_text():
    c = _client()
    with isolated({"prices_kase.csv": prices_csv().replace("Date,", "Дата,", 1)}):
        for url in ("/", "/data"):
            body = c.get(url).get_data(as_text=True)
            assert "Ошибка данных" in body or "Файлы с ошибками" in body
            assert "нет столбца Date" in body and "Traceback" not in body


def test_web_error_page_is_readable():
    c = _client()
    orig = pipeline.run

    def boom(*a, **k):
        raise RuntimeError("тестовый сбой")
    from webapp.app import app
    pipeline.run, app.logger.disabled = boom, True     # the failure is expected: keep the log quiet
    try:
        resp = c.get("/fx")
    finally:
        pipeline.run, app.logger.disabled = orig, False
    body = resp.get_data(as_text=True)
    assert resp.status_code == 500 and "тестовый сбой" in body and "сбросьте кэш" in body
    assert "Traceback" not in body


def test_two_cold_runs_give_identical_numbers():
    """NFR-01: a repeated run from scratch (no memory or disk cache) reproduces the decision
    weights, the Sharpe ratios and the cluster labels to 4 decimal places."""
    st = config.Settings(models=("ridge",))
    runs = []
    for _ in range(2):
        pipeline._MEM.clear()
        old = pipeline.CACHE_DIR
        pipeline.CACHE_DIR = Path(tempfile.mkdtemp(prefix="dss-nfr01-"))
        try:
            with data_dir({}):
                r = pipeline.run(st)
                runs.append((r.decision.allocation["weight"].round(4), r.backtest.metrics["sharpe"].round(4),
                             {k: v.tolist() for k, v in r.clusters.labels.items()}, round(r.macro.recession_prob, 4)))
        finally:
            pipeline.CACHE_DIR = old
    (w1, s1, l1, p1), (w2, s2, l2, p2) = runs
    assert w1.equals(w2) and s1.equals(s2) and l1 == l2 and p1 == p2


def test_report_and_console_use_decimal_comma():
    """Russian number format in the HTML report, decision texts and console output; CSV stays machine-readable."""
    import contextlib
    import io
    import re

    from dss import cli
    from dss.report import build_report
    with isolated():
        r = pipeline.run(config.Settings())
        cells = re.findall(r"<td class='num'>([^<]*)</td>", build_report(r))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["decide"])
            cli.main(["significance"])
    decimal_point = re.compile(r"\d\.\d")
    assert cells and not any(decimal_point.search(c) for c in cells)
    texts = [x for s in r.decision.steps for x in s["facts"] + s["trail"] + [s["result"]]] + r.decision.warnings
    assert not any(decimal_point.search(t) for t in texts), texts
    assert "," in out.getvalue() and not decimal_point.search(out.getvalue()), out.getvalue()
