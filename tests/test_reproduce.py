"""Tests of the article reproduction script (scripts/reproduce.py) and of the
Russian number format. They use the synthetic demo set, so they do not depend
on the files in data/.

Run with pytest (``pytest -q``) or without it (``python tests/run.py``).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reproduce  # noqa: E402
from test_system import isolated  # noqa: E402

from dss import config, macro, pipeline  # noqa: E402
from dss.data import demo  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402


def test_russian_number_format():
    assert num(1234.5) == "1 234,50" and num(-0.026, 3) == "-0,026"
    assert num(9077) == "9 077" and num(0.17, 2, sign=True) == "+0,17"
    assert pct(0.2161, 2) == "21,61%" and pct(-0.179, 1, sign=True) == "-17,9%"
    assert num(float("nan")) == "—" and pct(None) == "—" and num(np.float64(2.5), 1) == "2,5"


def test_a2_reports_r2_shocks_and_missing_article_values():
    a = reproduce.reproduce_a2(demo.fx_rates())
    reg = a.tables["a2_trend"]
    assert list(reg.index) == [f"{c}/KZT" for c in config.CURRENCIES] and reg["R2"].between(0, 1).all()
    assert a.tables["a2_describe"].loc["Счёт"].iloc[0] == len(demo.fx_rates().loc[:"2024-12-31"].dropna())
    cmp_ = a.tables["a2_comparison"]
    assert (cmp_.loc[cmp_["Показатель"].str.startswith("Наклон"), "Статья"] == reproduce.NEED).all()
    assert cmp_.loc[0, "Статья"] == "0,78"                   # published R2 of USD/KZT, decimal comma
    years = {d.year for d in a.tables["a2_shocks"]["date"]}
    assert {2009, 2015} <= years                              # the demo set contains these devaluations
    assert any("табл. 1" in q for q in a.questions)


def test_a3_robustness_variants_drop_tickers():
    raw = demo.equity_prices()
    a = reproduce.reproduce_a3(raw, None)
    t2 = a.tables["a3_table2"]
    assert set(t2.variant) == set(reproduce.A3_VARIANTS)
    assert set(t2.strategy) == {"K-means", "K-medoids", "Fuzzy C-means", "Равные веса"}
    lab = a.tables["a3_labels"]
    assert "KZTK" not in set(lab[lab.variant == "без KZTK"].ticker)
    assert "KMGZ" not in set(lab[lab.variant == "без KMGZ"].ticker)
    assert lab[["K-means", "K-medoids", "Fuzzy C-means"]].min().min() >= 1     # 1-based cluster numbers
    runs = a.tables["a3_runs"].set_index("variant")
    assert (runs.n_assets == [raw.shape[1], raw.shape[1] - 1, raw.shape[1] - 1]).all()


def test_leaky_features_include_the_target_year():
    long = macro.to_long(pd.DataFrame([np.arange(12, dtype=float)], index=pd.Index(["A"], name="country"),
                                      columns=list(range(2000, 2012))))
    ok = macro.build_features(long).set_index("year")
    leak = reproduce.leaky_features(long).set_index("year")
    assert ok.loc[2006, "rolling_mean3"] == 4.0                # years 2003-2005: 3, 4, 5
    assert leak.loc[2006, "rolling_mean3"] == 5.0              # years 2004-2006: includes growth of 2006
    assert (ok["lag1"].dropna() == leak["lag1"].dropna()).all()


def test_module4_helper_matches_the_pipeline():
    st = config.Settings(protocol="paper", cov_method="sample", rebalance="D", tc_bps=0, backtest_caps=False)
    with isolated():
        r = pipeline.run(st)
    _, bt = reproduce.run_module4(r.prices, st)
    pd.testing.assert_frame_equal(bt.metrics, r.backtest.metrics)


def test_report_lists_questions_and_needed_values():
    arts = {}
    for k in ("a1", "a2", "a3", "a4"):
        a = reproduce.Article(k, summary=f"итог {k}", questions=[f"вопрос {k}"])
        a.headline.append({"Статья": k, "Показатель": "x", "Статья, значение": reproduce.NEED, "СППР": num(0.5)})
        a.md += [f"## раздел {k}"]
        arts[k] = a
    md = reproduce.build_report(arts)
    assert md.index("## раздел a1") < md.index("## раздел a4") < md.index("## Вопросы к автору")
    assert "1. вопрос a1" in md and reproduce.NEED in md and "0,50" in md
    assert all(c.split("**")[1] in md for c in reproduce.CAVEATS if "**" in c)
