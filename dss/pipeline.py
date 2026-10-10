"""Orchestration of the four modules and the integration module, with caching.

Each module result is cached in memory and on disk (``.dss_cache/``) under a
key made of the data fingerprint and the parameters that affect it, so the
web UI recomputes only what a change of settings actually invalidates
(e.g. changing the investor profile re-runs only the integration step).

A malformed user CSV does not stop the run: that dataset falls back to the
demo set, and the reason is reported in ``RunResult.data_errors``.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from . import clustering, fx, integration, macro, portfolio, significance, signals
from .config import PROFILES, PROTOCOLS, Settings
from .data import DataError, load_fundamentals, load_fx, load_gdp, load_prices

CACHE_DIR = Path(os.environ.get("DSS_CACHE_DIR", ".dss_cache"))
_MEM: dict = {}


def _fingerprint(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, (pd.DataFrame, pd.Series)):
            h.update(pd.util.hash_pandas_object(o, index=True).values.tobytes())
            h.update(str(getattr(o, "columns", "")).encode())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()[:16]


def cached(name: str, key: str, fn):
    k = f"{name}-{key}"
    if k in _MEM:
        return _MEM[k]
    path = CACHE_DIR / f"{k}.pkl"
    if path.exists():
        try:
            _MEM[k] = pickle.loads(path.read_bytes())
            return _MEM[k]
        except Exception:
            path.unlink(missing_ok=True)
    val = fn()
    _MEM[k] = val
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(val))
    except Exception:
        pass
    return val


@dataclass
class RunResult:
    settings: Settings
    sources: dict
    is_demo: bool
    macro: macro.MacroResult
    fx: fx.FxResult
    clusters: clustering.ClusterResult
    cluster_table: pd.DataFrame
    signals: signals.SignalResult
    decision: integration.Decision
    prices: pd.DataFrame
    sectors: pd.Series
    data_errors: list = field(default_factory=list)   # user-facing messages, one per broken file
    forecast_tests: pd.DataFrame | None = None        # FR-IV-13: Diebold-Mariano vs naive forecast
    deferred: dict = field(default_factory=dict, repr=False)

    # The backtest depends on the profile limits but the decision does not need it,
    # so it runs only when a screen shows it: a profile change re-runs integration only (NFR-03).
    @cached_property
    def backtest(self) -> portfolio.BacktestResult:
        return self.deferred["backtest"]()

    @cached_property
    def sharpe_tests(self) -> pd.DataFrame:
        """FR-IV-13: bootstrap CI of the Sharpe ratios of the backtested strategies."""
        return cached("sharpe_tests", _fingerprint(self.backtest.daily, self.settings.risk_free),
                      lambda: significance.sharpe_tests(self.backtest.daily, self.settings.risk_free))


def _load(errors: list, loader, *args):
    """Load a dataset; on a malformed user CSV record the reason and use the demo set."""
    try:
        return loader(*args)
    except DataError as exc:
        errors.append(f"{exc}. Пока файл не исправлен, для этого набора используются демо-данные.")
        return loader(*args, source="demo")


def run(settings: Settings | None = None) -> RunResult:
    st = settings or Settings()
    errors: list = []
    gdp, rates = _load(errors, load_gdp), _load(errors, load_fx)
    px = _load(errors, load_prices, st.universe)
    if px.is_demo:      # keep valuation data consistent with the (demo) tickers
        fund = load_fundamentals(st.universe, list(px.frame.columns), "demo")
    else:
        fund = _load(errors, load_fundamentals, st.universe, list(px.frame.columns))
    prices = px.frame

    m = cached("macro", _fingerprint(gdp.frame), lambda: macro.run(gdp.frame))
    f = cached("fx", _fingerprint(rates.frame), lambda: fx.run(rates.frame))
    fund_num = fund.frame.select_dtypes("number") if len(fund.frame.columns) else None
    c = cached("clusters", _fingerprint(prices, fund_num),
               lambda: clustering.cluster_assets(prices, fund_num))
    ct = cached("cluster_bt", _fingerprint(prices, c.k, st.tc_bps, st.risk_free),
                lambda: clustering.cluster_portfolio_backtest(prices, c.k, tc_bps=st.tc_bps,
                                                              risk_free=st.risk_free))
    proto = PROTOCOLS[st.protocol]
    s = cached("signals", _fingerprint(prices, st.models, proto),
               lambda: signals.run(prices, st.models, proto["train_share"], proto["retrain_every"]))
    prof = PROFILES[st.profile]
    if st.backtest_caps:
        labels = c.labels["K-medoids"].reindex(prices.columns).to_numpy()
        U = c.U.reindex(prices.columns).to_numpy() if st.use_fuzzy_caps else None
        limits = (prof.max_asset_weight, labels, U, prof.cluster_cap)
        lkey = (prof.max_asset_weight, prof.cluster_cap, st.use_fuzzy_caps)
    else:                                    # [A4]: only 0 <= w_i <= 1
        limits, lkey = (1.0, None, None, None), None
    bt_key = _fingerprint(prices, st.models, proto, st.rebalance, st.tc_bps, st.cov_method, lkey, st.risk_free)
    deferred = {"backtest": lambda: cached("backtest", bt_key, lambda: portfolio.backtest(
        prices, s.ranks, st.rebalance, st.tc_bps, st.cov_method, *limits, risk_free=st.risk_free))}
    dm = cached("forecast_tests", _fingerprint(prices, st.models, proto),
                lambda: significance.forecast_tests(s.predictions, prices, signals.MODEL_LABELS))
    sectors = fund.frame["sector"] if "sector" in fund.frame else pd.Series(dtype=str)
    d = integration.decide(m, f, c, s, prices, st, sectors)
    sources = {"ВВП (Модуль I)": gdp.source, "Курсы (Модуль II)": rates.source,
               "Цены (Модули III–IV)": px.source, "Фундаментальные (Модуль III)": fund.source}
    is_demo = gdp.is_demo or rates.is_demo or px.is_demo
    return RunResult(st, sources, is_demo, m, f, c, ct, s, d, prices, sectors, errors, dm, deferred)


def clear_cache():
    _MEM.clear()
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.pkl"):
            p.unlink()


def rule_sensitivity(r: RunResult) -> dict:
    """Sensitivity of the integration rules at the inputs of this run."""
    sc = r.settings.scenario
    return integration.sensitivity_analysis(
        PROFILES[r.settings.profile], r.macro.recession_prob, sc, r.macro.phase,
        float(r.macro.scenario_path(sc).iloc[0]), r.fx.fx_risk)


def significance_tables(r: RunResult, fmt=lambda v: f"{v:.2f}") -> tuple[pd.DataFrame, pd.DataFrame]:
    """User-facing (Russian) tables of the FR-IV-13 tests; ``fmt`` formats CI bounds."""
    sh, dm = r.sharpe_tests, r.forecast_tests

    def ci(lo, hi):
        return "—" if pd.isna(lo) else f"[{fmt(lo)}; {fmt(hi)}]"
    ref = sh.attrs.get("reference", "Equal Weighted")
    t1 = pd.DataFrame({"Шарп": sh["sharpe"], "95% ДИ": [ci(a, b) for a, b in zip(sh.ci_low, sh.ci_high)],
                       f"Разница с {ref}": sh["diff"],
                       "95% ДИ разницы": [ci(a, b) for a, b in zip(sh.diff_low, sh.diff_high)],
                       "p-значение": sh["p_value"]}, index=sh.index)
    t2 = pd.DataFrame({"MSE·10⁴": dm["mse_model"] * 1e4, "MSE наивного·10⁴": dm["mse_naive"] * 1e4,
                       "DM": dm["dm"], "p-значение": dm["p_value"], "Вывод (α = 5%)": dm["verdict"]},
                      index=dm.index)
    return t1, t2
