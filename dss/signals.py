"""Module IV, stage A - ML return forecasts -> cross-sectional rank signals.  [A4]

* Technical features: log return, RSI(14), MACD(12-26) and signal line,
  ATR(14), Bollinger band width, OBV (if volume is given), SMA(14) ratio,
  trading volume; MinMax scaling fitted on the training sample only.
* Eleven forecasters of [A4]: Linear, Ridge, Lasso, SVR, Decision Tree,
  Random Forest, Gradient Boosting (scikit-learn) and RNN, LSTM, CNN-LSTM,
  LSTM-MLP (TensorFlow, optional - 60-day sequences). An MLP is added as a
  dependency-free neural baseline.
* Walk-forward evaluation (expanding window, periodic retraining) - a strict
  out-of-sample protocol that extends the single 80/20 split of [A4].
* Smoothing: mean of the last 10 predictions;  ranking: s_i = Rank(r_i)/N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from . import config

MODEL_LABELS = {
    "linear": "Linear Regression", "ridge": "Ridge", "lasso": "Lasso", "svr": "SVR",
    "dt": "Decision Tree", "rf": "Random Forest", "gb": "Gradient Boosting", "mlp": "MLP",
    "rnn": "RNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM", "lstm_mlp": "LSTM-MLP",
}
DL_MODELS = ("rnn", "lstm", "cnn_lstm", "lstm_mlp")
DL_EPOCHS, DL_BATCH, DL_PATIENCE = 50, 256, 10   # Adam(1e-3), early stopping 10 ([A4])


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def asset_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    lr = np.log(close).diff()
    f["log_ret"] = lr
    for k in (2, 3, 5):
        f[f"ret_lag{k}"] = lr.shift(k - 1)
    f["rsi14"] = _rsi(close)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    f["macd"] = macd
    f["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    f["atr14"] = close.diff().abs().rolling(14).mean() / close   # close-only true range
    sma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_width"] = 4 * sd20 / sma20
    f["sma14_ratio"] = close / close.rolling(14).mean() - 1
    f["vol20"] = lr.rolling(20).std()
    f["mom60"] = np.log(close).diff(config.LOOKBACK)
    if volume is not None:
        obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        f["obv_z"] = (obv - obv.rolling(60).mean()) / obv.rolling(60).std()
        f["volume_z"] = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
    f["target"] = lr.shift(-1)          # next-day log return
    return f


def build_panel(prices: pd.DataFrame, volumes: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = []
    for c in prices.columns:
        v = volumes[c] if volumes is not None and c in volumes else None
        f = asset_features(prices[c], v)
        f["asset"] = c
        parts.append(f)
    panel = pd.concat(parts).reset_index().rename(columns={"index": "Date"})
    panel = panel.rename(columns={panel.columns[0]: "Date"})
    feat_cols = [c for c in panel.columns if c not in ("Date", "asset", "target")]
    panel = panel.dropna(subset=feat_cols)
    return panel.sort_values(["Date", "asset"]).reset_index(drop=True)


def feature_columns(panel: pd.DataFrame) -> list:
    return [c for c in panel.columns if c not in ("Date", "asset", "target")]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(key: str):
    s = config.RANDOM_SEED
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1e-5, max_iter=5000),
        "svr": SVR(C=0.1, epsilon=1e-3),
        "dt": DecisionTreeRegressor(max_depth=10, min_samples_leaf=50, random_state=s),
        "rf": RandomForestRegressor(100, max_depth=10, min_samples_leaf=20, max_samples=0.25,
                                    max_features=0.5, n_jobs=-1, random_state=s),
        "gb": HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.05, random_state=s),
        "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300, early_stopping=True, random_state=s),
    }[key]


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _sequences(panel: pd.DataFrame, cols, scaler, lookback):
    """60-day feature sequences per asset -> (X[n, lookback, f], y, index rows)."""
    Xs, ys, idx = [], [], []
    for _, g in panel.groupby("asset", sort=False):
        a = scaler.transform(g[cols])
        for i in range(lookback - 1, len(g)):
            Xs.append(a[i - lookback + 1: i + 1])
            ys.append(g["target"].iloc[i])
            idx.append(g.index[i])
    return np.asarray(Xs, dtype="float32"), np.asarray(ys, dtype="float32"), np.asarray(idx)


def _test_windows(history: pd.DataFrame, test: pd.DataFrame, cols, scaler, lookback):
    """Windows ending on each test row: the last ``lookback - 1`` feature rows of
    ``history`` per asset (no gap before the block) followed by the test rows."""
    both = pd.concat([history.groupby("asset").tail(lookback - 1), test]).sort_values(["asset", "Date"])
    X, _, idx = _sequences(both, cols, scaler, lookback)
    keep = np.isin(idx, test.index)
    return X[keep], idx[keep]


def _keras_model(key: str, lookback: int, n_feat: int):  # pragma: no cover - needs TF
    from tensorflow import keras
    L = keras.layers
    inp = keras.Input((lookback, n_feat))
    if key == "rnn":
        x = L.SimpleRNN(32)(inp)
    elif key == "lstm":
        x = L.LSTM(32)(inp)
    elif key == "cnn_lstm":
        x = L.Conv1D(32, 3, activation="relu", padding="causal")(inp)
        x = L.MaxPooling1D(2)(x)
        x = L.LSTM(32)(x)
    else:  # lstm_mlp: LSTM sequence encoder + MLP on the last-day features
        h = L.LSTM(32)(inp)
        last = L.Lambda(lambda t: t[:, -1, :])(inp)
        m = L.Dense(32, activation="relu")(last)
        x = L.Concatenate()([h, m])
        x = L.Dense(32, activation="relu")(x)
    out = L.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


# --------------------------------------------------------------------------
# Walk-forward forecasting
# --------------------------------------------------------------------------
@dataclass
class SignalResult:
    predictions: dict            # model -> DataFrame (Date x asset), raw forecasts
    smoothed: pd.DataFrame       # ensemble, 10-period mean
    ranks: pd.DataFrame          # s_i,t in (0, 1]
    metrics: pd.DataFrame        # per model: RMSE, MAE, hit rate, IC
    test_start: pd.Timestamp
    models: tuple
    latest: pd.DataFrame         # latest raw, smoothed forecast and rank per asset


def _keras_seed():  # pragma: no cover - needs TF
    """Seed 42 for Python, NumPy and TensorFlow, deterministic ops (NFR-01)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:        # older TF or ops already initialised: seeds still apply
        pass


def _fit_predict(key, train, test, cols, lookback, history=None, state=None):
    """Fit on ``train`` (rows whose target is known) and forecast ``test``.

    ``history`` - feature rows before the test block (default: ``train``); the
    sequence models need the last ``lookback - 1`` of them per asset.
    ``state``   - dict shared across walk-forward blocks: a sequence model
    trained on an earlier block is fine-tuned instead of trained from scratch.
    """
    scaler = MinMaxScaler().fit(train[cols])
    if key in DL_MODELS:  # pragma: no cover - needs TF
        from tensorflow import keras
        Xtr, ytr, itr = _sequences(train, cols, scaler, lookback)
        order = np.argsort(itr, kind="stable")      # chronological: validation = latest 10 %
        Xtr, ytr = Xtr[order], ytr[order]
        Xte, idx = _test_windows(train if history is None else history, test, cols, scaler, lookback)
        state = {} if state is None else state
        if key not in state:
            _keras_seed()
            state[key] = _keras_model(key, lookback, len(cols))
        model = state[key]
        model.fit(Xtr, ytr, epochs=DL_EPOCHS, batch_size=DL_BATCH, validation_split=0.1, shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=DL_PATIENCE, restore_best_weights=True)])
        return pd.Series(model.predict(Xte, batch_size=1024, verbose=0).ravel(), index=idx)
    model = make_model(key)
    tr = train.dropna(subset=["target"])
    if key == "svr" and len(tr) > 5000:      # O(n^2): subsample
        tr = tr.sample(5000, random_state=config.RANDOM_SEED)
    model.fit(scaler.transform(tr[cols]), tr["target"])
    return pd.Series(model.predict(scaler.transform(test[cols])), index=test.index)


def rank_signal(pred: pd.DataFrame) -> pd.DataFrame:
    """s_{i,t} = Rank(r_hat_{i,t}) / N, average ranks for ties ([A4] Eq. 2)."""
    return pred.rank(axis=1, method="average").div(pred.notna().sum(axis=1), axis=0)


def evaluate(pred: pd.DataFrame, realised: pd.DataFrame) -> dict:
    p, y = pred.stack(), realised.reindex_like(pred).stack()
    both = pd.concat([p, y], axis=1, keys=["p", "y"]).dropna()
    ics = []
    for d, row in pred.iterrows():
        yy = realised.loc[d] if d in realised.index else None
        if yy is None:
            continue
        m = row.notna() & yy.notna()
        if m.sum() > 2:
            ics.append(spearmanr(row[m], yy[m]).statistic)
    ic = np.array(ics, dtype=float)
    return {"RMSE": float(np.sqrt(((both.p - both.y) ** 2).mean())),
            "MAE": float((both.p - both.y).abs().mean()),
            "hit_rate": float((np.sign(both.p) == np.sign(both.y)).mean()),
            "IC": float(np.nanmean(ic)),
            "IC_t": float(np.nanmean(ic) / (np.nanstd(ic) / np.sqrt(np.isfinite(ic).sum()))) if np.isfinite(ic).sum() > 2 else np.nan}


def run(prices: pd.DataFrame, models=("ridge", "rf", "gb"), train_share: float = 0.5,
        retrain_every: int = 126, smoothing: int = config.SMOOTHING,
        lookback: int = config.LOOKBACK, volumes: pd.DataFrame | None = None) -> SignalResult:
    models = tuple(m for m in models if m not in DL_MODELS or tensorflow_available())
    panel = build_panel(prices, volumes)
    cols = feature_columns(panel)
    dates = np.sort(panel["Date"].unique())
    start = int(len(dates) * train_share)
    test_dates = dates[start:]
    blocks = [test_dates[i:i + retrain_every] for i in range(0, len(test_dates), retrain_every)] if retrain_every else [test_dates]
    preds = {m: [] for m in models}
    state: dict = {}                       # sequence models carried over between blocks
    for block in blocks:
        # train on rows whose target is already known at the block start
        train = panel[panel["Date"] < block[0]].dropna(subset=["target"])
        train = train[train["Date"] < block[0] - np.timedelta64(1, "D")]
        test = panel[panel["Date"].isin(block)]
        history = panel[panel["Date"] < block[0]]      # features up to the eve of the block are known
        for m in models:
            preds[m].append(_fit_predict(m, train, test, cols, lookback, history, state))
    realised = panel.pivot(index="Date", columns="asset", values="target")
    out, rows = {}, []
    for m in models:
        s = pd.concat(preds[m])
        wide = panel.loc[s.index, ["Date", "asset"]].assign(pred=s.values).pivot(index="Date", columns="asset", values="pred")
        out[m] = wide
        rows.append({"model": MODEL_LABELS[m]} | evaluate(wide, realised))
    # ensemble = mean of per-model cross-sectional ranks (scale-free), then smoothing
    ens = sum(rank_signal(out[m]) for m in models) / len(models)
    smoothed = ens.rolling(smoothing, min_periods=1).mean()
    ranks = rank_signal(smoothed)
    ens_eval = evaluate(smoothed, realised)
    rows.append({"model": "Ансамбль (ранги, сглаж.)", "RMSE": np.nan, "MAE": np.nan,
                 "hit_rate": np.nan, "IC": ens_eval["IC"], "IC_t": ens_eval["IC_t"]})
    metrics = pd.DataFrame(rows).set_index("model")
    last = smoothed.index[-1]
    latest = pd.DataFrame({"raw_forecast_bp": sum(out[m].loc[last] for m in models) / len(models) * 1e4,
                           "smoothed_score": smoothed.loc[last], "rank_signal": ranks.loc[last]}
                          ).sort_values("rank_signal", ascending=False)
    return SignalResult(out, smoothed, ranks, metrics, pd.Timestamp(test_dates[0]), models, latest)
