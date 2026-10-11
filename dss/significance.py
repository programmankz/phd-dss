"""Statistical significance of the Module IV results (FR-IV-13).

[A4] reports point estimates only. Two tests answer "is it more than luck?":

* Sharpe ratios of the backtested strategies - 95 % confidence intervals and
  the difference to a reference strategy (Equal Weighted by default) by the
  circular block bootstrap: daily returns are autocorrelated and volatility
  clusters, so days are resampled in blocks; all strategies share the same
  resampled days (paired test). The Sharpe ratio is recomputed on every
  resample with the same formula as :func:`dss.clustering.performance`.
* Return forecasts - the Diebold-Mariano test of equal predictive accuracy
  against the naive forecast r_hat = 0 (no predictability), with the
  Harvey-Leybourne-Newbold small-sample correction. The loss of a day is the
  squared forecast error averaged across assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config

N_BOOT = 2000


def block_length(n: int) -> int:
    """Rule-of-thumb block length n^(1/3) (about 10 days for four years of data)."""
    return max(2, int(round(n ** (1 / 3))))


def block_bootstrap_indices(n: int, n_boot: int = N_BOOT, block: int | None = None,
                            seed: int = config.RANDOM_SEED) -> np.ndarray:
    """(n_boot, n) row indices of the circular block bootstrap."""
    block = block or block_length(n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def sharpe(daily: np.ndarray, risk_free: float = config.RISK_FREE) -> np.ndarray:
    """Annualised Sharpe of daily returns along the last axis:
    (geometric annual return - r_f) / annualised volatility."""
    r = np.asarray(daily, float)
    n = r.shape[-1]
    ann = np.expm1(np.log1p(r).sum(axis=-1) * 252 / n)
    vol = r.std(axis=-1, ddof=1) * np.sqrt(252)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 0, (ann - risk_free) / vol, np.nan)


def sharpe_tests(daily: pd.DataFrame, risk_free: float = config.RISK_FREE, reference: str = "Equal Weighted",
                 n_boot: int = N_BOOT, block: int | None = None, seed: int = config.RANDOM_SEED,
                 level: float = 0.95) -> pd.DataFrame:
    """Sharpe with bootstrap CI per strategy, and its difference to ``reference``
    with CI and a two-sided bootstrap p-value (H0: equal Sharpe ratios)."""
    r = daily.dropna().to_numpy().T                       # strategies x days
    idx = block_bootstrap_indices(r.shape[1], n_boot, block, seed)
    point = sharpe(r, risk_free)
    boot = np.vstack([sharpe(rj[idx], risk_free) for rj in r])   # strategies x n_boot
    a = (1 - level) / 2
    cols = list(daily.columns)
    ref = cols.index(reference) if reference in cols else None
    rows = []
    for j, name in enumerate(cols):
        row = {"sharpe": point[j], "ci_low": np.nanquantile(boot[j], a), "ci_high": np.nanquantile(boot[j], 1 - a)}
        if ref is not None and j != ref:
            diff, bd = point[j] - point[ref], boot[j] - boot[ref]
            # shift method: centre the bootstrap distribution on H0 (difference = 0)
            p = float(np.nanmean(np.abs(bd - diff) >= abs(diff)))
            row |= {"diff": diff, "diff_low": np.nanquantile(bd, a), "diff_high": np.nanquantile(bd, 1 - a),
                    "p_value": p}
        else:
            row |= {"diff": np.nan, "diff_low": np.nan, "diff_high": np.nan, "p_value": np.nan}
        rows.append(row)
    out = pd.DataFrame(rows, index=pd.Index(cols, name="strategy"))
    out.attrs |= {"reference": reference, "n_boot": n_boot, "block": block or block_length(r.shape[1]),
                  "n_days": r.shape[1], "level": level}
    return out


def diebold_mariano(loss_a, loss_b, h: int = 1) -> tuple[float, float]:
    """DM statistic and two-sided p-value for H0: E[loss_a - loss_b] = 0.
    Negative statistic -> forecast A is more accurate. Newey-West variance with
    h - 1 lags, HLN correction, Student t with n - 1 degrees of freedom."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    lrv = dc @ dc / n + 2 * sum((1 - k / h) * (dc[k:] @ dc[:-k]) / n for k in range(1, h))
    if lrv <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def forecast_tests(predictions: dict, prices: pd.DataFrame, labels: dict | None = None,
                   alpha: float = 0.05) -> pd.DataFrame:
    """DM test of every model against the naive zero-return forecast."""
    realised = np.log(prices).diff().shift(-1)           # target of Module IV: next-day log return
    rows = []
    for key, pred in predictions.items():
        y = realised.reindex(index=pred.index, columns=pred.columns)
        mask = pred.notna() & y.notna()
        loss_m = ((pred - y) ** 2).where(mask).mean(axis=1)
        loss_0 = (y ** 2).where(mask).mean(axis=1)
        dm, p = diebold_mariano(loss_m, loss_0)
        verdict = ("точнее наивного" if dm < 0 else "хуже наивного") if p < alpha else "не отличается"
        rows.append({"model": (labels or {}).get(key, key), "mse_model": float(loss_m.mean()),
                     "mse_naive": float(loss_0.mean()), "dm": dm, "p_value": p, "verdict": verdict,
                     "n_days": int(loss_m.notna().sum())})
    return pd.DataFrame(rows).set_index("model")
