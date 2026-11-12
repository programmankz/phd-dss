"""Reproduction of the forecasting set-up of the Iskakov thesis (ENU, 2026) in the DSS protocol.

Set-up as specified in docs/../TZ section 5.4: the seven KASE stocks of Table 4, target = cumulative
log return over the next 21 trading days, ElasticNet (and Random Forest, XGBoost for comparison),
features: returns and momentum 5/10/21 days, volatility 5/21/63, skewness and kurtosis 21/63,
RSI-14, MACD, Bollinger %b and width, Amihud 21 days (data/volumes_kase.csv), rolling correlations
21/63 days with Brent, the KASE index and USD/KZT, price to SMA50/SMA200, ATR-14 and cross-sectional
ranks of the features on each date.

Protocol (what the check is about):
* walk-forward with an expanding window, refit every 21 trading days; the training rows of a
  fit are those whose 21-day target window ends before the first test date (purge of 21 days);
* hyper-parameters of ElasticNet (alpha, l1_ratio) chosen by time-series cross-validation inside
  the training part of every fit (3 folds with a 21-day gap); RF and XGBoost with fixed settings;
* direction accuracy with the z-test against 1/2 (as in the thesis), and the Pesaran-Timmermann
  test (a) on all overlapping daily observations (naive), (b) on non-overlapping observations
  (every 21st date) and (c) on all dates with Newey-West errors, lag 20;
* portfolio on the forecasts: monthly rebalancing (every 21 trading days) on the forecast dates,
  max signal-to-risk on the forecast ranks with the moderate per-asset cap, 10 b.p. costs,
  fully invested in stocks, compared with equal weights on the same dates;
* the admission rule of Module IV (t_IC > 1.645 over the last 12 monthly decisions).

Also: Pesaran-Timmermann tests of the DSS Module IV ensemble at horizons 1, 5 and 20 days.

Outputs: docs/results/iskakov_metrics.csv, iskakov_portfolio.csv, module4_pt.csv,
docs/reproduction_iskakov.md. XGBoost needs ``pip install xgboost`` (otherwise it is skipped).

Usage::

    python scripts/reproduce_iskakov.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import end_to_end as e2e  # noqa: E402
from dss import config, portfolio, significance, signals  # noqa: E402
from dss.data import loaders  # noqa: E402
from dss.numfmt import num, pct  # noqa: E402

H = 21
REFIT = 21
EN_GRID = [(a, l1) for a in (1e-4, 1e-3, 1e-2) for l1 in (0.2, 0.5, 0.8)]
RANK_FEATURES = ("ret_21", "vol_21", "amihud_21", "mom_21")
GATE_WINDOW_M = 12


# ----------------------------------------------------------------------------- data and features
def load(tickers):
    px, _ = loaders.raw_prices("kase")
    px = px[tickers].ffill().dropna()
    vol = pd.read_csv(loaders.data_dir() / "volumes_kase.csv", index_col=0, parse_dates=True)[tickers]
    vol = vol.reindex(px.index)
    brent = pd.read_csv(loaders.data_dir() / "brent.csv", index_col=0, parse_dates=True)["brent"]
    idx = pd.read_csv(loaders.data_dir() / "index_kase.csv", index_col=0, parse_dates=True)["close"]
    usd = loaders.load_fx().frame["USD"]
    ext = pd.DataFrame({"brent": brent, "kase": idx, "usd": usd}).sort_index().ffill().reindex(px.index)
    return px, vol, ext


def features(px: pd.DataFrame, vol: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(px).diff()
    ext_r = np.log(ext).diff()
    parts = []
    for c in px.columns:
        p, r = px[c], lr[c]
        f = pd.DataFrame(index=px.index)
        for k in (5, 10, 21):
            f[f"ret_{k}"] = np.log(p / p.shift(k))
            f[f"mom_{k}"] = p / p.rolling(k).mean() - 1
        for k in (5, 21, 63):
            f[f"vol_{k}"] = r.rolling(k).std()
        for k in (21, 63):
            f[f"skew_{k}"] = r.rolling(k).skew()
            f[f"kurt_{k}"] = r.rolling(k).kurt()
        f["rsi_14"] = signals._rsi(p)
        ema12, ema26 = p.ewm(span=12, adjust=False).mean(), p.ewm(span=26, adjust=False).mean()
        f["macd"] = (ema12 - ema26) / p
        f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
        sma20, sd20 = p.rolling(20).mean(), p.rolling(20).std()
        f["bb_pctb"] = (p - (sma20 - 2 * sd20)) / (4 * sd20)
        f["bb_width"] = 4 * sd20 / sma20
        traded = vol[c].where(vol[c] > 0)
        f["amihud_21"] = (r.abs() / traded * 1e9).rolling(H, min_periods=5).mean()
        for e in ("brent", "kase", "usd"):
            for k in (21, 63):
                f[f"corr_{e}_{k}"] = r.rolling(k).corr(ext_r[e])
        f["p_sma50"] = p / p.rolling(50).mean() - 1
        f["p_sma200"] = p / p.rolling(200).mean() - 1
        f["atr_14"] = p.diff().abs().rolling(14).mean() / p
        f["target"] = np.log(p.shift(-H) / p)
        f["asset"] = c
        parts.append(f)
    panel = pd.concat(parts).rename_axis("Date").reset_index()
    for k in RANK_FEATURES:
        panel[f"xrank_{k}"] = panel.groupby("Date")[k].rank(pct=True)
    cols = [c for c in panel.columns if c not in ("Date", "asset", "target")]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    return panel.sort_values(["Date", "asset"]).reset_index(drop=True)


def feature_cols(panel):
    return [c for c in panel.columns if c not in ("Date", "asset", "target")]


# ----------------------------------------------------------------------------- models
def xgb_model():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                        random_state=config.RANDOM_SEED, n_jobs=4, verbosity=0)


def fit_elasticnet(X, y, dates):
    """ElasticNet with (alpha, l1_ratio) chosen by 3-fold time-series CV with a 21-day gap inside
    the training data only."""
    ud = np.unique(dates)
    folds = np.array_split(ud[len(ud) // 2:], 3)
    best, best_mse = EN_GRID[0], np.inf
    for a, l1 in EN_GRID:
        mse = []
        for fd in folds:
            tr = dates < fd[0] - np.timedelta64(int(H * 1.5), "D")
            te = np.isin(dates, fd)
            if tr.sum() < 200 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(X[tr])
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            mse.append(np.mean((m.predict(sc.transform(X[te])) - y[te]) ** 2))
        if mse and np.mean(mse) < best_mse:
            best, best_mse = (a, l1), np.mean(mse)
    sc = StandardScaler().fit(X)
    m = ElasticNet(alpha=best[0], l1_ratio=best[1], max_iter=5000).fit(sc.transform(X), y)
    return (lambda Z: m.predict(sc.transform(Z))), best


def walk_forward(panel, train_share=0.5, log=print):
    cols = feature_cols(panel)
    dates = np.sort(panel["Date"].unique())
    start = int(len(dates) * train_share)
    blocks = [dates[i:i + REFIT] for i in range(start, len(dates), REFIT)]
    models = ["ElasticNet", "Random Forest"] + (["XGBoost"] if xgb_model() is not None else [])
    preds = {m: [] for m in models}
    chosen = []
    t0 = time.perf_counter()
    for b in blocks:
        trd = dates[dates < b[0]]
        if len(trd) <= H:
            continue
        cutoff = trd[-H - 1]                    # target of the row ends before the first test date
        tr = panel[(panel["Date"] <= cutoff)].dropna(subset=["target"])
        te = panel[panel["Date"].isin(b)]
        X, y, d = tr[cols].to_numpy(), tr["target"].to_numpy(), tr["Date"].to_numpy()
        Xt = te[cols].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, best = fit_elasticnet(X, y, d)
        chosen.append(best)
        preds["ElasticNet"].append(pd.Series(f(Xt), index=te.index))
        rf = RandomForestRegressor(200, max_depth=6, min_samples_leaf=50, max_features=0.33, n_jobs=-1,
                                   random_state=config.RANDOM_SEED).fit(X, y)
        preds["Random Forest"].append(pd.Series(rf.predict(Xt), index=te.index))
        if "XGBoost" in preds:
            xg = xgb_model().fit(X, y)
            preds["XGBoost"].append(pd.Series(xg.predict(Xt), index=te.index))
    log(f"  walk-forward: {len(blocks)} refits, {time.perf_counter() - t0:.0f} s")
    out = {}
    for m in models:
        s = pd.concat(preds[m])
        out[m] = panel.loc[s.index, ["Date", "asset", "target"]].assign(pred=s.to_numpy())
    return out, chosen, pd.Timestamp(dates[start])


# ----------------------------------------------------------------------------- evaluation
def forecast_metrics(df: pd.DataFrame, train_mean: float) -> dict:
    d = df.dropna(subset=["target"])
    y, p = d["target"].to_numpy(), d["pred"].to_numpy()
    r2_zero = 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)
    r2_mean = 1 - np.sum((y - p) ** 2) / np.sum((y - train_mean) ** 2)
    dates = np.sort(d["Date"].unique())
    nonover = set(dates[::H])
    dn = d[d["Date"].isin(nonover)]
    # pooled panel: HAC over the time order of the stacked rows (sorted by date, then asset)
    naive = significance.pesaran_timmermann(p, y)
    non = significance.pesaran_timmermann(dn["pred"].to_numpy(), dn["target"].to_numpy())
    hac = significance.pesaran_timmermann(p, y, nw_lag=H * d["asset"].nunique() - 1)
    ic = signals.daily_ic(d.pivot(index="Date", columns="asset", values="pred"),
                          d.pivot(index="Date", columns="asset", values="target"))
    ic_non = ic.reindex(sorted(nonover))
    t_ic, n_ic = significance.ic_t_stat(ic_non.to_numpy())
    return {"n_obs": len(d), "r2_vs_zero": r2_zero, "r2_vs_train_mean": r2_mean, "hit_rate": naive["hit_rate"],
            "z_half": naive["z_half"], "p_half": naive["p_half"], "pt_naive": naive["stat"], "p_pt_naive": naive["p_value"],
            "n_nonoverlap": non["n"], "hit_nonoverlap": non["hit_rate"], "pt_nonoverlap": non["stat"],
            "p_pt_nonoverlap": non["p_value"], "pt_hac": hac["stat"], "p_pt_hac": hac["p_value"],
            "ic_mean": float(ic.mean()), "t_ic_nonoverlap": t_ic, "n_ic": n_ic,
            "share_up": float((y > 0).mean())}


def portfolio_backtest(df: pd.DataFrame, px: pd.DataFrame, rf_daily: pd.Series, tc_bps=10.0, w_max=0.20):
    """Monthly rebalancing on every 21st forecast date: max signal-to-risk on the forecast ranks vs equal
    weights; weights set at the close of t earn returns from t to t + 1; fully invested in stocks.
    Also reports the admission status of the signal (t_IC over the last 12 decisions)."""
    rets = px.pct_change()
    dates = np.sort(df["Date"].unique())
    reb = set(dates[::H])
    pred = df.pivot(index="Date", columns="asset", values="pred")[px.columns]
    tgt = df.pivot(index="Date", columns="asset", values="target")[px.columns]
    ic = signals.daily_ic(pred, tgt)
    days = rets.index[(rets.index >= dates[0]) & (rets.index <= dates[-1])]
    n = px.shape[1]
    w = {"forecast": np.full(n, 1 / n), "gated": np.full(n, 1 / n), "EW": np.full(n, 1 / n)}
    out = {k: [] for k in w}
    admitted_log, past = [], []
    for i, t in enumerate(days[:-1]):
        cost = dict.fromkeys(w, 0.0)
        if t in reb:
            cov = portfolio.covariance(rets.loc[:t].dropna().iloc[-config.COV_WINDOW:])
            s = signals.rank_signal(pred.loc[[t]]).iloc[0].fillna(0.5).to_numpy()
            # a decision's IC is known once its 21-day target has realised
            known = [d for d in past if d <= days[max(0, i - H)]]
            t_ic, n_ic = significance.ic_t_stat(ic.reindex(known[-GATE_WINDOW_M:]).to_numpy())
            ok = bool(np.isfinite(t_ic) and n_ic >= 6 and t_ic > e2e.GATE_T)
            admitted_log.append({"date": t, "t_ic": t_ic, "n_ic": n_ic, "admitted": ok})
            new = {"forecast": portfolio.max_signal_to_risk(s, cov, w_max),
                   "gated": portfolio.max_signal_to_risk(s, cov, w_max) if ok else portfolio.min_variance(cov, w_max),
                   "EW": np.full(n, 1 / n)}
            for k in w:
                cost[k] = np.abs(new[k] - w[k]).sum() * tc_bps / 1e4
                w[k] = new[k]
            past.append(t)
        r = rets.loc[days[i + 1]].fillna(0).to_numpy()
        for k in w:
            rp = float(w[k] @ r)
            out[k].append((days[i + 1], (1 - cost[k]) * (1 + rp) - 1))
            w[k] = w[k] * (1 + r) / (1 + rp)
    daily = pd.DataFrame({k: pd.Series(dict(v)) for k, v in out.items()})
    ex = daily.sub(rf_daily.reindex(daily.index).fillna(0), axis=0)
    tests = significance.sharpe_tests(ex, 0.0, reference="EW")
    rows = {}
    for k in daily:
        c = (1 + daily[k]).cumprod()
        rows[k] = {"annual_return": float(np.expm1(np.log1p(daily[k]).sum() * 252 / len(daily))),
                   "volatility": float(daily[k].std() * np.sqrt(252)), "sharpe": tests.loc[k, "sharpe"],
                   "max_drawdown": float((c / c.cummax() - 1).min()), "diff_vs_ew": tests.loc[k, "diff"],
                   "p_vs_ew": tests.loc[k, "p_value"]}
    return pd.DataFrame(rows).T, pd.DataFrame(admitted_log)


def module4_pt(px: pd.DataFrame) -> pd.DataFrame:
    """Pesaran-Timmermann tests of the DSS Module IV ensemble (ranks of Table 4) at horizons 1, 5, 20."""
    from dss.pipeline import _fingerprint, cached
    fn = lambda: signals.run(px, ("ridge", "rf", "gb"), config.WF_INITIAL_TRAIN, config.WF_RETRAIN_EVERY)  # noqa: E731
    s = cached("e2e_signals", _fingerprint(px), fn)
    raw = sum(s.predictions[m] for m in s.models) / len(s.models)
    score = s.smoothed.sub(s.smoothed.mean(axis=1), axis=0)          # above / below the cross-section
    lp = np.log(px)
    rows = []
    for h in (1, 5, 20):
        fwd = (lp.shift(-h) - lp).reindex(raw.index)
        fwd_rel = fwd.sub(fwd.mean(axis=1), axis=0)
        for name, f, y in (("forecast sign vs return sign", raw, fwd), ("ensemble score vs relative return", score, fwd_rel)):
            st = pd.concat([f.stack(), y.stack()], axis=1, keys=["f", "y"]).dropna()
            dates = st.index.get_level_values(0)
            ud = np.sort(dates.unique())
            non = st[dates.isin(ud[::h])]
            a = significance.pesaran_timmermann(non["f"], non["y"])
            b = significance.pesaran_timmermann(st["f"], st["y"], nw_lag=max(h * px.shape[1] - 1, 0))
            rows.append({"horizon": h, "test": name, "n_nonoverlap": a["n"], "hit_rate": a["hit_rate"],
                         "pt_nonoverlap": a["stat"], "p_nonoverlap": a["p_value"], "pt_hac": b["stat"],
                         "p_hac": b["p_value"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def markdown(metrics, chosen, test_start, port, gate, m4, n_assets) -> str:
    en = metrics.loc["ElasticNet"]
    out = ["# Воспроизведение постановки диплома Искакова Е.К. в протоколе СППР", "",
           "Результаты `python scripts/reproduce_iskakov.py`; таблицы `docs/results/iskakov_metrics.csv`, "
           "`iskakov_portfolio.csv`, `module4_pt.csv`. Текста диплома в репозитории нет; постановка взята из "
           "ТЗ (раздел 5.4): 7 бумаг KASE таблицы 4, цель — накопленная лог-доходность за 21 торговый день, "
           "ElasticNet, Random Forest и XGBoost, признаки по списку ТЗ (" f"{n_assets} бумаг, цены с дивидендами).", "",
           "## Протокол", "",
           "* walk-forward с расширяющимся окном, переобучение каждые 21 торговый день, тест с "
           f"{test_start:%d.%m.%Y}; в обучение входят только строки, у которых 21-дневное окно цели закончилось "
           "до первой даты теста (очистка 21 день);",
           "* гиперпараметры ElasticNet (α, l1_ratio) — по 3-кратной временной кросс-валидации с разрывом "
           "внутри обучающей части каждого окна; RF и XGBoost — с фиксированными настройками;",
           "* точность направления: z-тест доли против 1/2 (как в дипломе) и тест Песарана–Тиммерманна: "
           "(а) по всем перекрывающимся дневным наблюдениям, (б) по неперекрывающимся (каждая 21-я дата), "
           "(в) по всем датам с ошибками Ньюи–Уэста (лаг 21·N − 1 по сложенной панели N бумаг, т.е. 20 дней);",
           "* портфель на прогнозах — ежемесячная ребалансировка (каждые 21 день) в walk-forward, max сигнал/риск "
           "по рангам прогнозов с лимитом 20% на бумагу, издержки 10 б.п., 100% в акциях; сравнение с равными "
           "весами на тех же датах; «с допуском» — сигнал используется, только если t_IC > 1,645 по последним "
           "12 месячным решениям, иначе минимальная дисперсия.", "",
           "## Качество прогнозов", "",
           "| Модель | Набл. | R² к нулю | R² к среднему обучения | Точность, % | z (к 1/2), p | PT по всем, p | "
           "Неперекр. набл. | Точность неперекр., % | PT неперекр., p | PT HAC, p | IC | t_IC (неперекр.) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in metrics.iterrows():
        out.append(f"| {m} | {int(r['n_obs'])} | {num(r['r2_vs_zero'], 3)} | {num(r['r2_vs_train_mean'], 3)} | "
                   f"{pct(r['hit_rate'])} | {num(r['z_half'], 2)}, {num(r['p_half'], 3)} | {num(r['p_pt_naive'], 3)} | "
                   f"{int(r['n_nonoverlap'])} | {pct(r['hit_nonoverlap'])} | {num(r['p_pt_nonoverlap'], 3)} | "
                   f"{num(r['p_pt_hac'], 3)} | {num(r['ic_mean'], 3)} | {num(r['t_ic_nonoverlap'], 2)} |")
    grid = pd.Series([f"α={a:g}, l1={l1:g}".replace(".", ",") for a, l1 in chosen]).value_counts()
    out += ["", f"Доля 21-дневных доходностей больше нуля в тесте: {pct(en['share_up'])} — это база, от которой "
            "отсчитывается точность направления: прогноз «всегда рост» в растущем рынке даёт ту же точность без "
            "всякого умения, и z-тест против 1/2 этого не учитывает, а тест Песарана–Тиммерманна учитывает.", "",
            "Выбранные гиперпараметры ElasticNet по окнам: " + "; ".join(f"{k} — {v}" for k, v in grid.items()) + ".", "",
            "## Портфель на прогнозах", "",
            "| Стратегия | Доходность, % год. | Волатильность, % | Шарп | Макс. просадка, % | Разность Шарпа к EW | p |",
            "|---|---|---|---|---|---|---|"]
    for model, t in port.items():
        for k, r in t.iterrows():
            name = {"forecast": f"{model}: прогноз", "gated": f"{model}: прогноз с допуском", "EW": "Равные веса"}[k]
            if k == "EW" and model != next(iter(port)):
                continue
            out.append(f"| {name} | {pct(r['annual_return'])} | {pct(r['volatility'])} | {num(r['sharpe'])} | "
                       f"{pct(r['max_drawdown'])} | {num(r['diff_vs_ew'], sign=True)} | {num(r['p_vs_ew'], 3)} |")
    for model, g in gate.items():
        out += ["", f"Допуск сигнала {model}: допущен в {int(g['admitted'].sum())} из {len(g)} месячных решений "
                f"(t_IC: медиана {num(g['t_ic'].median())}, максимум {num(g['t_ic'].max())})."]
    out += ["", "## Модуль IV СППР: тест Песарана–Тиммерманна, горизонты 1, 5 и 20 дней", "",
            "Ансамбль Модуля IV (ridge, RF, GB) таблицы 4. «Знак прогноза» — знак сырого прогноза доходности против "
            "знака будущей доходности; «оценка ансамбля» — выше/ниже среднего по сечению против относительной "
            "доходности (то, что реально использует оптимизатор).", "",
            "| Горизонт, дн. | Проверка | Неперекр. набл. | Точность, % | PT, p (неперекр.) | PT HAC, p |",
            "|---|---|---|---|---|---|"]
    for _, r in m4.iterrows():
        test = {"forecast sign vs return sign": "знак прогноза", "ensemble score vs relative return": "оценка ансамбля"}[r["test"]]
        out.append(f"| {r['horizon']} | {test} | {int(r['n_nonoverlap'])} | {pct(r['hit_rate'])} | "
                   f"{num(r['p_nonoverlap'], 3)} | {num(r['p_hac'], 3)} |")
    out += ["", "## Выводы", ""]
    sig_naive = en["p_half"] < 0.05
    sig_robust = en["p_pt_nonoverlap"] < 0.05 or en["p_pt_hac"] < 0.05
    out.append(f"* ElasticNet: точность направления {pct(en['hit_rate'])}, z-тест против 1/2 "
               f"{'значим' if sig_naive else 'не значим'} (p = {num(en['p_half'], 3)}); с учётом базовой доли роста "
               f"и перекрытия целей — PT по неперекрывающимся наблюдениям p = {num(en['p_pt_nonoverlap'], 3)}, "
               f"PT HAC p = {num(en['p_pt_hac'], 3)}: {'направление предсказывается значимо' if sig_robust else 'значимого умения предсказывать направление нет'}.")
    r2 = ", ".join(f"{m} {num(r['r2_vs_zero'], 3)} (к среднему обучения {num(r['r2_vs_train_mean'], 3)})"
                   for m, r in metrics.iterrows())
    out.append(f"* R² вне выборки к нулевому прогнозу: {r2}. Положительный R² к нулю при R² около нуля к среднему "
               "обучения означает, что модель в основном выучила средний рост рынка.")
    out.append(f"* Точность ElasticNet {pct(en['hit_rate'])} близка к 60,4% диплома, но почти совпадает с долей "
               f"растущих 21-дневных периодов ({pct(en['share_up'])}): высокая точность против 1/2 отражает рост рынка, "
               "а не умение различать рост и падение. Сравнивать нужно не точность с 1/2, а тест PT с поправкой на "
               "перекрытие.")
    m4s = m4[(m4["test"] == "ensemble score vs relative return")]
    sig_h = [int(h) for h, pv in zip(m4s["horizon"], m4s["p_nonoverlap"]) if pv < 0.05]
    out.append("* Ансамбль Модуля IV: относительная (межсекционная) оценка значимо предсказывает направление "
               f"относительной доходности на горизонтах {', '.join(map(str, sig_h))} дн. (PT по неперекрывающимся "
               "наблюдениям, p < 0,05)." if sig_h else
               "* Ансамбль Модуля IV: значимой предсказуемости направления по PT нет ни на одном горизонте.")
    g = gate["ElasticNet"]
    out.append(f"* Правило допуска (t_IC > 1,645 по последним 12 месячным решениям) ElasticNet проходит в "
               f"{int(g['admitted'].sum())} из {len(g)} решений, т.е. не устойчиво; как вариант Модуля IV с месячной "
               "ребалансировкой сигнал не подключается. Портфель на прогнозах без допуска не отличается от равных "
               "весов значимо (таблица выше).")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(ROOT / "docs" / "results"))
    a = p.parse_args(argv)
    px, vol, ext = load(e2e.KASE7)
    panel = features(px, vol, ext)
    print(f"panel: {len(panel)} rows, {panel['Date'].min():%Y-%m-%d} – {panel['Date'].max():%Y-%m-%d}, "
          f"{len(feature_cols(panel))} features")
    preds, chosen, test_start = walk_forward(panel)
    train_mean = float(panel.loc[panel["Date"] < test_start, "target"].mean())
    metrics = pd.DataFrame({m: forecast_metrics(df, train_mean) for m, df in preds.items()}).T
    rates, _ = loaders.load_rates()
    days = px.index
    dt = pd.Series(days, index=days).diff().dt.days.fillna(0) / 365.0
    rf_daily = rates["kzt"].reindex(days, method="ffill").shift(1) * dt
    port, gate = {}, {}
    for m, df in preds.items():
        port[m], gate[m] = portfolio_backtest(df, px, rf_daily)
    m4 = module4_pt(px)
    out = Path(a.out)
    metrics.to_csv(out / "iskakov_metrics.csv")
    pd.concat(port, names=["model", "strategy"]).to_csv(out / "iskakov_portfolio.csv")
    m4.to_csv(out / "module4_pt.csv", index=False)
    (ROOT / "docs" / "reproduction_iskakov.md").write_text(
        markdown(metrics, chosen, test_start, port, gate, m4, px.shape[1]), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(metrics.round(3).to_string())
        print(m4.round(3).to_string())
    print("written docs/reproduction_iskakov.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
