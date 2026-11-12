"""End-to-end (point-in-time) backtest of the integrated DSS on KASE data.

At every weekly rebalancing date t each input is rebuilt from the information
available on that date:

* Module I  - re-estimated on the GDP panel truncated to the last published
              year (growth of year Y is assumed known from 1 April of Y + 1);
* Module II - FX risk index on NBK rates up to t;
* Module III - K-medoids clusters on the trailing 252 trading days;
* Module IV - walk-forward rank signals (ridge, random forest, gradient boosting);
* integration - rules (5)-(6): risky share d, dollar share h of the defensive
  part, feasible asset / cluster limits, weights of the risky part (4).

Accounting (in tenge): weights set at the close of t earn the return from t to
t + 1; the tenge component of the defensive part accrues the NBK base rate and
the dollar component the 3-month T-bill rate (simple interest, actual/365 on
calendar days) plus the change of the official USD/KZT rate; transaction costs
are ``--tc`` basis points of the two-sided turnover (sum of |weight changes|) of all
positions; the reported turnover is one-sided (half of that sum). Stock returns are
price returns from data/prices_kase.csv with data/corporate_actions.csv applied
(the KZTK dividend of May 2025); ``--no-corporate-actions`` uses raw prices.

Outputs (``--out``, default docs/results): e2e_<tag>_table.csv (one row per
configuration with bootstrap tests), e2e_<tag>_decisions.csv (one row per
rebalancing date of the full DSS), e2e_<tag>_daily.csv (daily returns of the
configurations) and e2e_<tag>.png; ``--sensitivity`` adds the coefficient and
defensive-part sensitivity tables; ``--perturb DATE`` repeats the run with all
prices, rates and the last GDP year replaced by random numbers from DATE on and
reports how many earlier decisions changed.

Usage::

    python scripts/end_to_end.py                                   # Table 4: 7 stocks, 2023-2026
    python scripts/end_to_end.py --modules 12 --start 2020-01-06   # Table 5: Modules I-II, 2020-2026
    python scripts/end_to_end.py --tickers HSBK,CCBN,KCEL,KEGC,KZTO,KZAP --tag no_kztk
    python scripts/end_to_end.py --tickers all --start 2024-12-23 --tag nine
    python scripts/end_to_end.py --sensitivity --perturb 2025-01-01
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import baselines, clustering, config, fx, integration, macro, portfolio, signals, significance  # noqa: E402
from dss.data import loaders  # noqa: E402
from dss.pipeline import cached, _fingerprint  # noqa: E402

KASE7 = ["HSBK", "CCBN", "KZTK", "KCEL", "KEGC", "KZTO", "KZAP"]
KZT, USD = "KZT (base rate)", "USD (T-bill)"
SHOCKS = {"20.02-31.03.2020": ("2020-02-20", "2020-03-31"), "15.02-31.03.2022": ("2022-02-15", "2022-03-31")}


# ----------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    prices: pd.DataFrame        # trading days x tickers of the universe (complete history)
    usd: pd.Series              # USD/KZT on trading days
    fx_rates: pd.DataFrame      # calendar days x USD, EUR, CNY, RUB
    gdp: pd.DataFrame           # wide GDP panel
    rates: pd.DataFrame         # calendar days x kzt, usd (fractions)
    source: str


def load_inputs(tickers, corporate_actions: bool = True) -> Inputs:
    px_path = loaders.data_dir() / "prices_kase.csv"
    if corporate_actions:
        px, label = loaders.raw_prices("kase")
    else:
        px, label = loaders._dated_frame(px_path, "Date,<тикер1>,…"), "CSV: prices_kase.csv (без корпоративных событий)"
    px = px.ffill()
    cols = list(px.columns) if tickers == "all" else list(tickers)
    missing = [t for t in cols if t not in px]
    if missing:
        raise SystemExit(f"нет цен для {', '.join(missing)}")
    prices = px[cols].dropna()
    fx_rates = loaders.load_fx().frame
    rates, rlabel = loaders.load_rates()
    if rates is None:
        raise SystemExit(f"ставки: {rlabel}")
    usd = fx_rates["USD"].reindex(prices.index, method="ffill")
    return Inputs(prices, usd, fx_rates, loaders.load_gdp().frame, rates, f"{label}; {rlabel}")


def perturb_inputs(inp: Inputs, date: str, seed: int = 7) -> Inputs:
    """Replace prices, exchange rates (from ``date``) and the last GDP year with random numbers."""
    rng = np.random.default_rng(seed)
    d = pd.Timestamp(date)
    px = inp.prices.copy()
    m = px.index >= d
    px.loc[m] = px.loc[~m].iloc[-1].to_numpy() * np.exp(np.cumsum(rng.normal(0, 0.05, (m.sum(), px.shape[1])), axis=0))
    fxr = inp.fx_rates.copy()
    m = fxr.index >= d
    fxr.loc[m] = fxr.loc[~m].iloc[-1].to_numpy() * np.exp(np.cumsum(rng.normal(0, 0.02, (m.sum(), fxr.shape[1])), axis=0))
    gdp = inp.gdp.copy()
    last = gdp.columns.max()
    gdp[last] = rng.normal(0, 5, len(gdp))
    return Inputs(px, fxr["USD"].reindex(px.index, method="ffill"), fxr, gdp, inp.rates, inp.source + "; perturbed")


# ----------------------------------------------------------------------------- module inputs as of t
def published_year(t: pd.Timestamp) -> int:
    """Last GDP year known at t: growth of year Y is published on 1 April of Y + 1."""
    return t.year - 1 if (t.month, t.day) >= (4, 1) else t.year - 2


def macro_vintage(gdp: pd.DataFrame, year: int, use_cache: bool = True) -> dict:
    panel = gdp.loc[:, gdp.columns <= year]
    fn = lambda: macro.run(panel, dtw=False)   # noqa: E731 - country DTW clustering is not a decision input
    m = cached("e2e_macro", _fingerprint(panel), fn) if use_cache else fn()
    g = m.scenario_path("base")
    return {"rec_prob": m.recession_prob, "phase": m.phase, "growth_next": float(g.iloc[0]), "year": year}


def weekly_dates(days: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    """First trading day of every ISO week in [start, end]."""
    d = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    weeks = pd.Series(d.isocalendar().year * 100 + d.isocalendar().week, index=d)
    return pd.DatetimeIndex(weeks.drop_duplicates(keep="first").index)


def resolve_start(start, first_signal, log=print) -> pd.Timestamp:
    """Start of the run: ``start`` or, when Module IV ranks begin later, the first date with ranks
    (with a warning instead of an error, so that the default ``--start`` survives data updates)."""
    start, first_signal = pd.Timestamp(start), pd.Timestamp(first_signal)
    if first_signal <= start:
        return start
    log(f"  WARNING: Module IV ranks start on {first_signal:%Y-%m-%d}, later than --start {start:%Y-%m-%d}; "
        f"the run starts on {first_signal:%Y-%m-%d}")
    return first_signal


# ----------------------------------------------------------------------------- configurations
@dataclass(frozen=True)
class Config:
    name: str
    macro: bool        # risky share from rule (5); else d = d0
    fx: bool           # dollar share from rule (6); else h = 0
    m4: bool           # weights from Module IV optimiser (4); else equal weights
    d_fixed: float | None = None   # overrides d (EW 100 %: 1.0)
    weights: str | None = None     # baseline rule of the risky part (dss.baselines), replaces m4/clusters
    clusters: bool | None = None   # Module III cluster limits; None: on together with m4 (as in Tables 4-5).
                                   # m4 off, clusters on: equal weights per cluster, capped (cluster_equal_weights);
                                   # m4 on, clusters off: optimiser (4) with the per-asset cap only

    @property
    def use_clusters(self) -> bool:
        if self.weights:
            return False
        return self.m4 if self.clusters is None else self.clusters


TABLE4 = [Config("EW 100%", False, False, False, 1.0), Config("EW + KZT", False, False, False),
          Config("No macro module", False, True, True), Config("No FX module", True, False, True),
          Config("Full DSS", True, True, True), Config("Full DSS, EW in risky part", True, True, False)]
TABLE5 = [Config("EW 100%", False, False, False, 1.0), Config("EW + KZT", False, False, False),
          Config("Macro rule only", True, False, False), Config("Currency rule only", False, True, False),
          Config("Macro and currency rules", True, True, False)]


class Engine:
    """Point-in-time inputs and decisions for one universe and period."""

    def __init__(self, inp: Inputs, profile: str, start, end, rules: config.IntegrationRules = config.RULES,
                 need_m4: bool = True, use_cache: bool = True, log=print):
        self.inp, self.profile, self.rules, self.log = inp, config.PROFILES[profile], rules, log
        self.use_cache = use_cache
        self.days = inp.prices.index
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.rebal = weekly_dates(self.days, start, end)
        self.returns = inp.prices.pct_change()
        self.ranks = None
        if need_m4:
            t0 = time.perf_counter()
            fn = lambda: signals.run(inp.prices, ("ridge", "rf", "gb"), config.WF_INITIAL_TRAIN, config.WF_RETRAIN_EVERY)  # noqa: E731
            s = cached("e2e_signals", _fingerprint(inp.prices), fn) if use_cache else fn()
            self.ranks, self.ic = s.ranks, float(s.metrics.iloc[-1]["IC"])
            first = s.ranks.dropna(how="all").index[0]
            start = resolve_start(self.start, first, log)
            if start != self.start:
                self.start, self.rebal = start, weekly_dates(self.days, start, end)
            log(f"  Module IV walk-forward: {time.perf_counter() - t0:.1f} s, ranks from {first:%Y-%m-%d}, IC = {self.ic:.3f}")
        self._macro, self._clusters, self._fx = {}, {}, {}

    # --- inputs as of t
    def macro_at(self, t):
        y = published_year(t)
        if y not in self._macro:
            t0 = time.perf_counter()
            self._macro[y] = macro_vintage(self.inp.gdp, y, self.use_cache)
            self.log(f"  Module I vintage {y}: P_rec = {self._macro[y]['rec_prob']:.2f}, phase {self._macro[y]['phase']}, "
                     f"g = {self._macro[y]['growth_next']:.2f} ({time.perf_counter() - t0:.1f} s)")
        return self._macro[y]

    def fx_at(self, t) -> float:
        if t not in self._fx:
            self._fx[t] = fx.risk_signal(self.inp.fx_rates.loc[:t].dropna())[1]
        return self._fx[t]

    def clusters_at(self, t) -> pd.Series:
        if t not in self._clusters:
            window = self.inp.prices.loc[:t].iloc[-config.COV_WINDOW:]
            self._clusters[t] = clustering.cluster_assets(window).labels["K-medoids"]
        return self._clusters[t]

    def cov_at(self, t) -> np.ndarray:
        return portfolio.covariance(self.returns.loc[:t].dropna().iloc[-config.COV_WINDOW:], "ledoit_wolf")

    # --- one decision
    def decide(self, t, cfg: Config) -> dict:
        n = self.inp.prices.shape[1]
        out = {"date": t}
        if cfg.d_fixed is not None:
            d = cfg.d_fixed
        elif cfg.macro:
            m = self.macro_at(t)
            d, _ = integration.risk_budget(self.profile, m["rec_prob"], "base", m["phase"], m["growth_next"], self.rules)
            out |= {k: m[k] for k in ("rec_prob", "phase", "growth_next")}
        else:
            d = self.profile.base_risky_share
        if cfg.fx:
            r = self.fx_at(t)
            h = integration.fx_hedge(r, self.rules)
            out["fx_risk"] = r
        else:
            h = 0.0
        labels = None
        if cfg.use_clusters:
            labels = self.clusters_at(t).to_numpy()
            wm, cc, relaxed = portfolio.feasible_limits(labels, self.profile.max_asset_weight, self.profile.cluster_cap)
            out |= {"k": int(len(set(labels))), "w_max_eff": wm, "c_eff": cc, "relaxed": relaxed}
        if cfg.weights:
            window = self.returns.loc[:t].dropna().iloc[-config.COV_WINDOW:]
            ranks = (self.ranks.loc[:t].iloc[-1].reindex(self.inp.prices.columns).fillna(0.5).to_numpy()
                     if self.ranks is not None else None)
            rf = float(self.inp.rates["kzt"].loc[:t].iloc[-1])
            w = baselines.weights(cfg.weights, window, self.profile.max_asset_weight, ranks, rf)
        elif cfg.m4:
            s = self.ranks.loc[:t].iloc[-1].reindex(self.inp.prices.columns).fillna(0.5).to_numpy()
            w = portfolio.max_signal_to_risk(s, self.cov_at(t), self.profile.max_asset_weight, labels, None,
                                             self.profile.cluster_cap if labels is not None else None)
        elif labels is not None:
            w = portfolio.cluster_equal_weights(labels, self.profile.max_asset_weight, self.profile.cluster_cap)
        else:
            w = np.full(n, 1.0 / n)
        out |= {"d": d, "h": h}
        weights = np.concatenate([d * w, [(1 - d) * (1 - h), (1 - d) * h]])
        out["weights"] = weights
        return out


def asset_returns(inp: Inputs, kzt_shift: float = 0.0) -> pd.DataFrame:
    """Daily returns of stocks, the tenge and the dollar component on trading days t (return t-1 -> t);
    ``kzt_shift`` lowers the tenge yield (fraction p.a.) to represent attainable deposit rates."""
    days = inp.prices.index
    dt = pd.Series(days, index=days).diff().dt.days.fillna(0) / 365.0
    r = inp.rates.reindex(days, method="ffill")
    kzt = (np.maximum(r["kzt"] - kzt_shift, 0.0).shift(1) * dt)              # rate known at t-1 accrues t-1 -> t
    usd = (1 + r["usd"].shift(1) * dt) * inp.usd / inp.usd.shift(1) - 1
    out = inp.prices.pct_change()
    out[KZT], out[USD] = kzt, usd
    return out


def run_config(engine: Engine, cfg: Config, rets: pd.DataFrame, tc_bps: float) -> tuple[pd.Series, pd.DataFrame]:
    """Daily net returns of a configuration and its decision log."""
    days = engine.days[(engine.days >= engine.start) & (engine.days <= engine.end)]
    rets = rets.reindex(days)
    n = rets.shape[1]
    w = np.zeros(n)
    w[-2] = 1.0                                   # start fully in tenge
    rebal = set(engine.rebal)
    daily, log, turnover = [], [], []
    for i, t in enumerate(days):
        cost = 0.0
        if t in rebal:
            dec = engine.decide(t, cfg)
            target = dec["weights"]
            to = float(np.abs(target - w).sum()) / 2      # one-sided turnover, units of portfolio value
            cost = 2 * to * tc_bps / 1e4                 # costs on both legs of the trade
            w = target.copy()
            turnover.append(to)
            log.append({k: v for k, v in dec.items() if k != "weights"} | {"turnover": to})
        if i + 1 < len(days):
            r = rets.iloc[i + 1].fillna(0.0).to_numpy()
            rp = float(w @ r)
            daily.append((days[i + 1], (1 - cost) * (1 + rp) - 1))
            w = w * (1 + r) / (1 + rp)
    s = pd.Series(dict(daily), name=cfg.name)
    return s, pd.DataFrame(log)


def metrics(r: pd.Series, rf_daily: pd.Series, turnover_per_year: float, log: pd.DataFrame) -> dict:
    ex = (r - rf_daily.reindex(r.index)).to_numpy()
    curve = (1 + r).cumprod()
    n = len(r)
    out = {"annual_return": float(np.expm1(np.log1p(r).sum() * 252 / n)),
           "volatility": float(r.std(ddof=1) * np.sqrt(252)),
           "sharpe": float(significance.sharpe(ex, 0.0)),
           "max_drawdown": clustering.max_drawdown(curve),
           "turnover_per_year": turnover_per_year,
           "mean_d": float(log["d"].mean()), "mean_h": float(log["h"].mean())}
    for name, (a, b) in SHOCKS.items():
        seg = r.loc[a:b]
        out[f"return {name}"] = float((1 + seg).prod() - 1) if len(seg) else np.nan
    if "relaxed" in log:
        out |= {"relaxed_share": float(log["relaxed"].mean()), "w_max_eff_max": float(log["w_max_eff"].max()),
                "c_eff_max": float(log["c_eff"].max())}
    return out


def run_table(engine: Engine, configs, rets: pd.DataFrame, tc_bps: float, reference: str, log=print):
    daily, logs, rows = {}, {}, {}
    rf = rets[KZT]
    for cfg in configs:
        t0 = time.perf_counter()
        s, lg = run_config(engine, cfg, rets, tc_bps)
        years = len(s) / 252
        daily[cfg.name], logs[cfg.name] = s, lg
        rows[cfg.name] = metrics(s, rf, float(lg["turnover"].sum() / years), lg)
        log(f"  {cfg.name}: Sharpe {rows[cfg.name]['sharpe']:.2f}, return {rows[cfg.name]['annual_return']:.1%} "
            f"({time.perf_counter() - t0:.1f} s)")
    daily = pd.DataFrame(daily)
    excess = daily.sub(rf.reindex(daily.index), axis=0)
    tests = significance.sharpe_tests(excess, 0.0, reference=reference)
    table = pd.DataFrame(rows).T
    for col in ("diff", "diff_low", "diff_high", "p_value", "p_holm"):
        table[col] = tests[col]
    table.attrs = {"n_days": len(daily), "block": tests.attrs["block"], "n_boot": tests.attrs["n_boot"],
                   "mean_base_rate": float(rf.reindex(daily.index).mean() * 252 / (daily.index.to_series().diff().dt.days.mean() / 365 * 252))}
    return table, daily, logs


# ----------------------------------------------------------------------------- sensitivity
def coefficient_variants(rules: config.IntegrationRules) -> dict:
    out = {}
    for name in ("rec_penalty", "neg_growth_cut", "hedge_base", "hedge_slope", "hedge_max"):
        for f in (0.75, 1.25):
            out[f"{name} x{f}"] = replace(rules, **{name: getattr(rules, name) * f})
    for f in (0.75, 1.25):
        out[f"scenario_mult x{f}"] = replace(rules, scenario_mult={k: v * f for k, v in rules.scenario_mult.items()})
        out[f"phase_mult x{f}"] = replace(rules, phase_mult={k: v * f for k, v in rules.phase_mult.items()})
    out["phase_mult removed"] = replace(rules, phase_mult={k: 1.0 for k in rules.phase_mult})
    return out


def sensitivity(engine: Engine, rets: pd.DataFrame, tc_bps: float, inp: Inputs, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = rets[KZT]
    full = Config("Full DSS", True, True, True)
    rows = {}
    base_rules = engine.rules
    variants = {"baseline": base_rules} | coefficient_variants(base_rules) | {"fx weights equal": base_rules}
    for name, rules in variants.items():
        engine.rules = rules
        if name == "fx weights equal":
            saved = dict(config.FX_RISK_WEIGHTS)
            config.FX_RISK_WEIGHTS.update({k: 0.25 for k in saved})
            engine._fx = {}
        s, lg = run_config(engine, full, rets, tc_bps)
        rows[name] = metrics(s, rf, float(lg["turnover"].sum() / (len(s) / 252)), lg)
        if name == "fx weights equal":
            config.FX_RISK_WEIGHTS.update(saved)
            engine._fx = {}
        log(f"  {name}: Sharpe {rows[name]['sharpe']:.2f}, MDD {rows[name]['max_drawdown']:.1%}")
    engine.rules = base_rules
    coef = pd.DataFrame(rows).T

    # defensive part: lower attainable tenge yields and higher costs, same decisions
    rows = {}
    for shift, tc in ((0.0, tc_bps), (0.02, tc_bps), (0.04, tc_bps), (0.0, 30.0), (0.04, 30.0)):
        r2 = asset_returns(inp, shift)
        for cfg in TABLE4:
            s, lg = run_config(engine, cfg, r2, tc)
            m = metrics(s, rf, float(lg["turnover"].sum() / (len(s) / 252)), lg)
            rows[(f"KZT yield -{shift:.0%}, costs {tc:.0f} b.p.", cfg.name)] = m
        log(f"  defensive sensitivity: yield -{shift:.0%}, costs {tc:.0f} b.p. done")
    dfn = pd.DataFrame(rows).T
    dfn.index.names = ["variant", "configuration"]
    return coef, dfn


# ----------------------------------------------------------------------------- plot
def plot(daily: pd.DataFrame, log_full: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = ["-", "-", ":", "-.", "--", (0, (1, 1))]
    greys = ["#1a1a1a", "#777777", "#1a1a1a", "#555555", "#1a1a1a", "#999999"]
    for (name, s), ls, c in zip(daily.items(), styles, greys):
        a.plot((1 + s).cumprod() * 100, ls=ls, c=c, lw=1.4, label=name)
    a.set_title(f"(a) Value in tenge, {title}")
    a.set_ylabel("Portfolio value, start = 100")
    a.legend(frameon=False, fontsize=8)
    b.step(log_full["date"], log_full["d"] * 100, where="post", c="#1a1a1a", lw=1.6, label="risky share d")
    b.step(log_full["date"], log_full["h"] * 100, where="post", c="#777777", lw=1.2, ls="--", label="USD share of defensive part h")
    for a0, b0 in SHOCKS.values():
        if pd.Timestamp(b0) >= log_full["date"].min():
            b.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), color="#dddddd")
    b.set_ylim(0, 100)
    b.set_title("(b) Decisions of the integration rules")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.grid(c="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", default=",".join(KASE7), help="comma list or 'all'")
    p.add_argument("--start", default="2023-01-16")
    p.add_argument("--end", default=None)
    p.add_argument("--profile", default="moderate", choices=list(config.PROFILES))
    p.add_argument("--modules", default="1234", choices=["1234", "12"], help="12: Table 5 (Modules I-II, EW in risky part)")
    p.add_argument("--tc", type=float, default=10.0, help="transaction costs, basis points of turnover")
    p.add_argument("--no-corporate-actions", action="store_true", help="raw prices (KZTK dividend not adjusted)")
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--perturb", default=None, help="future-perturbation check from this date")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "results"))
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args(argv)
    tickers = "all" if a.tickers == "all" else a.tickers.split(",")
    tag = a.tag or ("table5" if a.modules == "12" else "table4") + ("_raw" if a.no_corporate_actions else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    inp = load_inputs(tickers, not a.no_corporate_actions)
    end = a.end or inp.prices.index.max()
    print(f"Inputs: {inp.source}\n  universe {list(inp.prices.columns)}, prices {inp.prices.index.min():%Y-%m-%d} – "
          f"{inp.prices.index.max():%Y-%m-%d}, run {pd.Timestamp(a.start):%Y-%m-%d} – {pd.Timestamp(end):%Y-%m-%d}")
    configs = TABLE4 if a.modules == "1234" else TABLE5
    engine = Engine(inp, a.profile, a.start, end, need_m4=a.modules == "1234", use_cache=not a.no_cache)
    rets = asset_returns(inp)
    reference = "Full DSS" if a.modules == "1234" else "EW + KZT"
    table, daily, logs = run_table(engine, configs, rets, a.tc, reference)
    print(f"\n{tag}: {table.attrs['n_days']} trading days, bootstrap block {table.attrs['block']} days, "
          f"mean base rate {table.attrs['mean_base_rate']:.1%}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(4).to_string())
    table.to_csv(out / f"e2e_{tag}_table.csv")
    daily.to_csv(out / f"e2e_{tag}_daily.csv")
    full_name = "Full DSS" if a.modules == "1234" else "Macro and currency rules"
    logs[full_name].to_csv(out / f"e2e_{tag}_decisions.csv", index=False)
    plot(daily, logs[full_name], out / f"e2e_{tag}.png",
         f"{len(inp.prices.columns)} KASE stocks, {pd.Timestamp(a.start).year}–{pd.Timestamp(end).year}")
    meta = {"tag": tag, "universe": ",".join(inp.prices.columns), "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()), "n_days": table.attrs["n_days"], "block": table.attrs["block"],
            "n_boot": table.attrs["n_boot"], "mean_base_rate": table.attrs["mean_base_rate"],
            "n_rebalancing": len(engine.rebal), "tc_bps": a.tc, "corporate_actions": not a.no_corporate_actions,
            "ic_ensemble": getattr(engine, "ic", np.nan), "source": inp.source}

    if a.sensitivity:
        print("\nSensitivity of the integration coefficients (full DSS):")
        coef, dfn = sensitivity(engine, rets, a.tc, inp)
        coef.to_csv(out / f"e2e_{tag}_sensitivity_coefficients.csv")
        dfn.to_csv(out / f"e2e_{tag}_sensitivity_defensive.csv")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(coef[["sharpe", "max_drawdown", "annual_return", "mean_d", "mean_h"]].round(3).to_string())
            print("\nDefensive part (yields and costs):")
            print(dfn[["sharpe", "annual_return", "volatility"]].round(3).to_string())

    if a.perturb:
        print(f"\nFuture-perturbation check from {a.perturb}:")
        pert = perturb_inputs(inp, a.perturb)
        eng2 = Engine(pert, a.profile, a.start, end, need_m4=a.modules == "1234", use_cache=False, log=lambda *x: None)
        cfg = next(c for c in configs if c.name == full_name)
        before = pd.Timestamp(a.perturb)
        same = total = 0
        for t in engine.rebal:
            if t >= before:
                break
            w1, w2 = engine.decide(t, cfg)["weights"], eng2.decide(t, cfg)["weights"]
            total += 1
            same += bool(np.allclose(w1, w2, atol=1e-10))
        print(f"  {same} of {total} weekly decisions dated before {a.perturb} identical")
        meta |= {"perturb_date": a.perturb, "perturb_identical": same, "perturb_total": total}

    pd.Series(meta).to_csv(out / f"e2e_{tag}_meta.csv", header=False)
    print(f"\nDone in {time.perf_counter() - t_all:.0f} s; files e2e_{tag}_* in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
