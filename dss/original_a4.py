"""Pipeline of article [A4] as it is in the authors' code, applied to any market.

The steps, the hyperparameters and the order of operations follow ``Final_calc.ipynb`` and
``Data collection.ipynb`` of the article's repository (github.com/Yerassyl04/signal-based-portopt):

Stage A, separately for every share
    * features Log_Ret, RSI_14, MACD, ATR_14, BB_Width, OBV, SMA_14, Volume;
    * a 60-day window of the eight features (480 inputs after flattening) predicts the next-day
      simple return; chronological 80/20 split; MinMax scaling fitted on the training part;
    * Linear, Ridge(1.0), Lasso(0.001), SVR(rbf, C=100, gamma=0.1, eps=0.01), Decision Tree(depth 10),
      Random Forest(100, depth 10), Gradient Boosting(100, depth 5, lr 0.1) and, with TensorFlow,
      RNN, LSTM, CNN-LSTM, LSTM-MLP (Adam 1e-3, 50 epochs, batch 32, early stopping 10);
    * the best model of a share is the one with the highest 0.6 * directional accuracy +
      0.4 * (R2 + 1) * 50 **on the test part**; its signal is the mean of its last 10 test forecasts.
Stage B
    * Rank_Score = rank of the signal / N; covariance of daily returns over the whole sample x 252;
    * Max Signal-to-Risk and Min Variance by SLSQP (0 <= w <= 1, sum 1), Signal Weighted
      (proportional to the rank), Equal Weighted.
Backtest
    * the cell that builds the strategy returns is not in the repository; the published table is
      reproduced by holding the Stage B weights constant (daily rebalancing) over the whole sample.
      The weights use the forecasts and the covariance of the end of the sample, so this backtest
      is in-sample: it is reported as in the article and labelled accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

LOOKBACK = 60
LAST_N = 10
TRAIN_SHARE = 0.8
RISK_FREE = 0.05
FEATURES = ["Log_Ret", "RSI_14", "MACD", "ATR_14", "BB_Width", "OBV", "SMA_14", "Volume"]
DL_MODELS = ("RNN", "LSTM", "CNN-LSTM", "LSTM-MLP")
STRATEGIES = ("Signal Weighted", "Equal Weighted", "Min Variance", "Max Signal-to-Risk")


# ------------------------------------------------------------------ features (Data collection.ipynb)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columns Date, Open, High, Low, Close, Volume -> the processed frame of the article."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    close = df["Close"]
    df["SMA_14"] = close.rolling(14).mean()
    df["Log_Ret"] = np.log(close / close.shift(1))
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + gain / loss)
    df["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    ranges = pd.concat([df["High"] - df["Low"], (df["High"] - close.shift()).abs(), (df["Low"] - close.shift()).abs()], axis=1)
    df["ATR_14"] = ranges.max(axis=1).rolling(14).mean()
    sma, std = close.rolling(20).mean(), close.rolling(20).std()
    df["BB_Width"] = 4 * std / sma
    df["OBV"] = (np.sign(close.diff()) * df["Volume"]).fillna(0).cumsum()
    return df.dropna().reset_index(drop=True)


# ------------------------------------------------------------------ Stage A (Final_calc.ipynb, cell 0)
def _sequences(df: pd.DataFrame):
    d = df.dropna(subset=FEATURES + ["Close"]).reset_index(drop=True)
    d["Target_Return"] = d["Close"].pct_change().shift(-1)
    d = d[:-1].reset_index(drop=True)
    f, t = d[FEATURES].to_numpy(dtype=float), d["Target_Return"].to_numpy(dtype=float)
    X = np.array([f[i - LOOKBACK:i] for i in range(LOOKBACK, len(d))])
    return X, t[LOOKBACK:]


def evaluate(y, p, name: str) -> dict:
    r2 = r2_score(y, p)
    acc = float(np.mean(np.sign(y) == np.sign(p)) * 100)
    return {"Model": name, "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2), "Dir_Acc": acc, "Combined_Score": 0.6 * acc + 0.4 * (r2 + 1) * 50}


def sklearn_models() -> dict:
    return {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=1.0), "Lasso": Lasso(alpha=0.001),
            "SVR": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01),
            "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)}


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def _keras_models(shape) -> dict:  # pragma: no cover - needs TensorFlow
    from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Input, MaxPooling1D, SimpleRNN
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    def seq(*layers):
        m = Sequential([Input(shape=shape), *layers])
        m.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return m
    return {
        "RNN": lambda: seq(SimpleRNN(50, activation="relu", return_sequences=True), Dropout(0.2), SimpleRNN(50, activation="relu"),
                           Dropout(0.2), Dense(25, activation="relu"), Dense(1)),
        "LSTM": lambda: seq(LSTM(64, activation="relu", return_sequences=True), Dropout(0.2), LSTM(32, activation="relu"),
                            Dropout(0.2), Dense(16, activation="relu"), Dense(1)),
        "CNN-LSTM": lambda: seq(Conv1D(filters=64, kernel_size=3, activation="relu"), MaxPooling1D(pool_size=2),
                                LSTM(50, activation="relu", return_sequences=True), Dropout(0.2), LSTM(50, activation="relu"),
                                Dropout(0.2), Dense(25, activation="relu"), Dense(1)),
        "LSTM-MLP": lambda: seq(LSTM(64, activation="relu", return_sequences=False), Dropout(0.2), Dense(128, activation="relu"),
                                Dropout(0.3), Dense(64, activation="relu"), Dropout(0.2), Dense(32, activation="relu"), Dense(1)),
    }


def stage_a_stock(processed: pd.DataFrame, with_dl: bool = False, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """(metrics of every model on the test part sorted by the combined score, stable forecasts)."""
    X, y = _sequences(processed)
    split = int(len(X) * TRAIN_SHARE)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xte_s = scaler.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)
    flat_tr, flat_te = Xtr_s.reshape(len(Xtr_s), -1), Xte_s.reshape(len(Xte_s), -1)
    rows, preds = [], {}
    for name, model in sklearn_models().items():
        p = model.fit(flat_tr, ytr).predict(flat_te)
        rows.append(evaluate(yte, p, name))
        preds[name] = p
    if with_dl and tensorflow_available():  # pragma: no cover - needs TensorFlow
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        tf.keras.utils.set_random_seed(seed)        # the notebook sets no seed; fixed here for reproducibility
        for name, build in _keras_models((Xtr_s.shape[1], Xtr_s.shape[2])).items():
            m = build()
            m.fit(Xtr_s, ytr, epochs=50, batch_size=32, validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
            p = m.predict(Xte_s, verbose=0).flatten()
            if not np.all(np.isfinite(p)):
                continue
            rows.append(evaluate(yte, p, name))
            preds[name] = p
    table = pd.DataFrame(rows).sort_values("Combined_Score", ascending=False).reset_index(drop=True)
    stable = {k: float(np.mean(v[-LAST_N:])) for k, v in preds.items()}
    return table, stable


def stage_a(processed: dict, with_dl: bool = False, log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per share: best model, forecast, Rank_Score; all model metrics per share)."""
    summary, metrics = [], []
    for stock, df in processed.items():
        table, stable = stage_a_stock(df, with_dl)
        best = table.iloc[0]["Model"]
        summary.append({"Stock": stock, "Best_Model": best, "Predicted_Return": stable[best]})
        metrics.append(table.assign(Stock=stock))
        if log:
            log(f"{stock}: {best}, {stable[best]:+.4%}")
    s = pd.DataFrame(summary)
    s["Rank_Score"] = s["Predicted_Return"].rank() / len(s)
    return s.sort_values("Predicted_Return", ascending=False).reset_index(drop=True), pd.concat(metrics, ignore_index=True)


# ------------------------------------------------------------------ Stage B (Final_calc.ipynb, cell 1)
def common_prices(processed: dict, stocks) -> pd.DataFrame:
    return pd.DataFrame({s: processed[s].set_index("Date")["Close"] for s in stocks}).dropna()


def stage_b(signals: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    """Weights of the four portfolios (rows: shares in the order of ``signals``)."""
    s = signals.to_numpy(dtype=float)
    cov = (prices[signals.index].pct_change().dropna().cov() * 252).to_numpy()
    n = len(s)
    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.full(n, 1 / n)
    msr = minimize(lambda w: -np.sum(w * s) / (np.sqrt(w @ cov @ w) + 1e-8), x0, method="SLSQP", bounds=bounds, constraints=cons).x
    mv = minimize(lambda w: w @ cov @ w, x0, method="SLSQP", bounds=bounds, constraints=cons).x
    pos = np.maximum(s, 0)
    sw = pos / pos.sum() if pos.sum() > 0 else x0
    return pd.DataFrame({"Signal Weighted": sw, "Equal Weighted": x0, "Min Variance": mv, "Max Signal-to-Risk": msr},
                        index=signals.index)


# ------------------------------------------------------------------ backtest (Final_calc.ipynb, cell 2)
def metrics(returns: pd.Series) -> dict:
    total = float((1 + returns).prod() - 1)
    ann = float((1 + returns).prod() ** (252 / len(returns)) - 1)
    vol = float(returns.std() * np.sqrt(252))
    cum = (1 + returns).cumprod()
    return {"total_return": total, "annual_return": ann, "annual_vol": vol,
            "sharpe": (ann - RISK_FREE) / vol if vol > 0 else 0.0, "max_drawdown": float((cum / cum.cummax() - 1).min())}


def backtest(weights: pd.DataFrame, prices: pd.DataFrame, start=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Constant weights, daily rebalancing; (metrics per strategy, value curves with base 100)."""
    r = prices[weights.index].pct_change().dropna()
    if start is not None:
        r = r.loc[start:]
    daily = pd.DataFrame({c: r.to_numpy() @ weights[c].to_numpy() for c in weights}, index=r.index)
    table = pd.DataFrame({c: metrics(daily[c]) for c in daily}).T
    table.index.name = "strategy"
    return table, 100 * (1 + daily).cumprod()


@dataclass
class Original:
    summary: pd.DataFrame       # Stage A: best model, forecast and Rank_Score per share
    model_metrics: pd.DataFrame
    weights: pd.DataFrame
    table: pd.DataFrame         # backtest over the whole sample, as in the article
    curves: pd.DataFrame
    test_table: pd.DataFrame    # the same weights on the test part (last 20 % of the dates) only
    info: dict


def run(processed: dict, with_dl: bool = False, summary: pd.DataFrame | None = None,
        model_metrics: pd.DataFrame | None = None, stocks=None, log=None) -> Original:
    """Whole pipeline; ``summary`` reuses stored Stage A results (e.g. of the article)."""
    stocks = list(stocks or processed)
    if summary is None:
        summary, model_metrics = stage_a({s: processed[s] for s in stocks}, with_dl, log)
    else:
        summary = summary[summary["Stock"].isin(stocks)].copy()
        summary["Rank_Score"] = summary["Predicted_Return"].rank() / len(summary)
    signals = summary.set_index("Stock")["Rank_Score"]
    prices = common_prices(processed, signals.index)
    weights = stage_b(signals, prices)
    table, curves = backtest(weights, prices)
    test_start = prices.index[int(len(prices) * TRAIN_SHARE)]
    test_table, _ = backtest(weights, prices, test_start)
    info = {"assets": len(stocks), "start": prices.index[0], "end": prices.index[-1], "days": len(prices) - 1,
            "test_start": test_start, "with_dl": bool(with_dl and tensorflow_available()),
            "models": 11 if with_dl and tensorflow_available() else 7}
    return Original(summary, model_metrics if model_metrics is not None else pd.DataFrame(), weights, table, curves, test_table, info)


# ------------------------------------------------------------------ KASE with stored Stage A results
PUBLISHED_TABLE = pd.DataFrame(      # table IV of the article, NSE, 2020-2025
    {"total_return": [2.0974, 1.9383, 1.3221, 1.3628, 1.1220], "annual_return": [0.2161, 0.2059, 0.1567, 0.1609, 0.1397],
     "annual_vol": [0.1740, 0.1802, 0.1529, 0.1705, 0.1830], "sharpe": [0.95, 0.86, 0.70, 0.65, 0.49],
     "max_drawdown": [-0.3660, -0.3792, -0.3237, -0.3368, -0.3844]},
    index=pd.Index([*STRATEGIES, "Nifty 50"], name="strategy"))
KASE_DROP = {"nine": (), "seven": ("KSPI", "KMGZ")}
STAGE_A_FILES = {11: "original_a4_kase_stage_a_11.csv", 7: "original_a4_kase_stage_a_7.csv"}


def kase_processed(adjust: bool = True) -> dict:
    """Processed frames (article's features) of the KASE shares from data/ohlcv_kase.csv."""
    from dss.data import loaders
    bars = loaders.load_kase_ohlcv(adjust)
    return {t: add_features(g.drop(columns="Ticker")) for t, g in bars.groupby("Ticker", sort=False)}


def results_dir():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "docs" / "results"


def run_kase(kase_set: str = "seven", models: int = 11) -> Original:
    """Stage B and the backtest on KASE with the Stage A results stored by scripts/original_a4_kase.py
    (the neural networks need TensorFlow, so Stage A is not recomputed on the fly)."""
    stored = pd.read_csv(results_dir() / STAGE_A_FILES[models])
    processed = kase_processed()
    stocks = [s for s in processed if s not in KASE_DROP[kase_set]]
    res = run(processed, summary=stored, stocks=stocks)
    res.info.update(models=models, with_dl=models == 11, kase_set=kase_set)
    return res
