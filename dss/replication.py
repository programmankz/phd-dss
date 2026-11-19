"""Experiment of article [A4] repeated on another market (KASE instead of NSE).

Everything follows the article: daily technical features, the forecasting models
of [A4] (the seven scikit-learn models; the four TensorFlow models when
TensorFlow is installed), one chronological 80/20 split, smoothing over the
last 10 forecasts, rank signals s_i = Rank(r_i)/N, four strategies with daily
rebalancing, sample covariance over 252 days, no transaction costs and no
weight limits other than 0 <= w_i <= 1. The only thing that changes is the
price file. The same function runs the NSE sample of the article, so the two
markets are computed by identical code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config, portfolio, signals, significance
from .clustering import performance
from .data.loaders import load_prices, mean_base_rate, raw_prices, validate_prices

ARTICLE_MODELS = ("linear", "ridge", "lasso", "svr", "dt", "rf", "gb")   # scikit-learn models of [A4]
ARTICLE_END = {"nse": "2025-12-31"}       # the article covers 2020-2025
UNIVERSES = {"kase": "KASE", "nse": "NSE (выборка статьи)"}
# KASE sets: the price file as it is (common history from the latest listing) or without the two
# recently listed shares, which gives a history from December 2018 and a longer test sample
KASE_SETS = {"nine": {"label": "9 бумаг, общая история с 12.2022", "drop": ()},
             "seven": {"label": "7 бумаг, общая история с 12.2018 (без KSPI и KMGZ)", "drop": ("KSPI", "KMGZ")}}
# the only strategy row of Table IV of [A4] available in the repository (docs/methodology.md)
PUBLISHED = {"annual_return": 0.2161, "annual_vol": 0.1740, "sharpe": 0.95}
METRICS = ["total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown", "turnover"]


@dataclass
class Replication:
    universe: str
    info: dict
    models: pd.DataFrame          # forecast metrics and the Diebold-Mariano test per model
    strategies: pd.DataFrame      # backtest metrics per strategy
    sharpe_tests: pd.DataFrame    # bootstrap test of each strategy against equal weights
    curves: pd.DataFrame
    notes: list = field(default_factory=list)


def kase_prices(kase_set: str = "nine") -> tuple[pd.DataFrame, bool]:
    """(prices of a KASE set with dividends and corporate actions applied, demo flag)."""
    ds = load_prices("kase")
    drop = KASE_SETS[kase_set]["drop"]
    if ds.is_demo or not drop:
        return ds.frame, ds.is_demo
    raw, _ = raw_prices("kase")
    return validate_prices(raw.drop(columns=[c for c in drop if c in raw.columns]), "prices_kase.csv"), False


def run(universe: str = "kase", prices: pd.DataFrame | None = None, with_dl: bool = False,
        local_rate: float | None = None, kase_set: str = "nine") -> Replication:
    """Protocol of [A4] on ``universe``; ``prices`` overrides the committed price file."""
    notes = []
    if prices is None:
        if universe == "kase":
            prices, demo = kase_prices(kase_set)
        else:
            ds = load_prices(universe)
            prices, demo = ds.frame, ds.is_demo
        if demo:
            notes.append("Файл цен не найден, расчёт выполнен на демонстрационных данных.")
    if universe in ARTICLE_END:
        prices = prices.loc[:ARTICLE_END[universe]]
    models = ARTICLE_MODELS + (signals.DL_MODELS if with_dl and signals.tensorflow_available() else ())
    if not (with_dl and signals.tensorflow_available()):
        notes.append("Четыре нейросетевые модели статьи (RNN, LSTM, CNN-LSTM, LSTM-MLP) требуют TensorFlow и в этот "
                     "расчёт не входят; использованы семь моделей scikit-learn.")
    proto = config.PROTOCOLS["paper"]
    s = signals.run(prices, models, proto["train_share"], proto["retrain_every"])
    bt = portfolio.backtest(prices, s.ranks, "D", 0.0, "sample", 1.0, None, None, None, risk_free=config.RISK_FREE)
    dm = significance.forecast_tests(s.predictions, prices, signals.MODEL_LABELS)
    model_table = s.metrics.join(dm[["dm", "p_value", "verdict"]], how="left")
    strategies = bt.metrics[METRICS].copy()
    if universe == "kase":
        if local_rate is None:
            local_rate = mean_base_rate(str(bt.daily.index[0].date()))
        if local_rate is not None:
            strategies["sharpe_local"] = [performance(bt.daily[c], local_rate)["sharpe"] for c in strategies.index]
    tests = significance.sharpe_tests(bt.daily, config.RISK_FREE)
    info = {"assets": len(prices.columns), "data_start": prices.index[0], "data_end": prices.index[-1],
            "test_start": s.test_start, "bt_start": bt.daily.index[0], "bt_end": bt.daily.index[-1],
            "bt_days": len(bt.daily), "train_share": proto["train_share"], "models": len(models),
            "risk_free": config.RISK_FREE, "local_rate": local_rate,
            "up_days": float((bt.daily["Equal Weighted"] > 0).mean())}
    return Replication(universe, info, model_table, strategies, tests, bt.curves, notes)
