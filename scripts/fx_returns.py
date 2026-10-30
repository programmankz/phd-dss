"""Currency risk on returns, not on levels (complement to Module II / [A2]).

[A2] concludes from correlations of exchange-rate levels (0.98-0.99) that USD,
EUR and CNY form one risk factor; correlations of levels are dominated by the
common trend. This script recomputes the analysis on daily logarithmic returns
of the official NBK rates (data/fx_rates.csv): pairwise correlations over
2000-2024 and after the move to a floating rate (20 August 2015), the share of
variance explained by the first principal component, and the correlation of
weekly returns of an equally weighted portfolio of KASE stocks with the weekly
change of USD/KZT (is the dollar position a hedge of the stock portfolio?).

    python scripts/fx_returns.py        # writes docs/results/fx_returns_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dss import config  # noqa: E402
from dss.data import loaders  # noqa: E402

FLOAT_DATE = "2015-08-20"            # transition of the tenge to a floating exchange rate
PERIODS = {"2000-2024": ("2000-01-01", "2024-12-31"), "after float (2015-08-20 - 2024)": (FLOAT_DATE, "2024-12-31")}


def log_returns(rates: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns on days when at least one rate changed (official rates
    are repeated on weekends and holidays, which would dilute the correlations)."""
    r = np.log(rates).diff().dropna()
    return r[(r != 0).any(axis=1)]


def pc1_share(r: pd.DataFrame) -> float:
    z = (r - r.mean()) / r.std()
    ev = np.linalg.eigvalsh(np.cov(z.to_numpy().T))
    return float(ev.max() / ev.sum())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(ROOT / "docs" / "results"))
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fxd = loaders.load_fx()
    rates = fxd.frame[config.CURRENCIES].dropna()
    print(f"Rates: {fxd.source}, {rates.index.min():%Y-%m-%d} – {rates.index.max():%Y-%m-%d}")

    corr_rows, pc_rows = [], []
    for name, (s, e) in PERIODS.items():
        lv = rates.loc[s:e]
        r = log_returns(lv)
        cl, cr = lv.corr(), r.corr()
        for i, x in enumerate(config.CURRENCIES):
            for y in config.CURRENCIES[i + 1:]:
                corr_rows.append({"period": name, "pair": f"{x}-{y}", "levels": cl.loc[x, y], "log_returns": cr.loc[x, y]})
        pc_rows.append({"period": name, "n_days": len(r), "pc1_levels_USD_EUR_CNY": pc1_share(lv[["USD", "EUR", "CNY"]]),
                        "pc1_returns_USD_EUR_CNY": pc1_share(r[["USD", "EUR", "CNY"]]),
                        "pc1_returns_all": pc1_share(r)})
    corr = pd.DataFrame(corr_rows)
    pcs = pd.DataFrame(pc_rows)
    corr.to_csv(out / "fx_returns_correlations.csv", index=False)
    pcs.to_csv(out / "fx_returns_pc1.csv", index=False)
    print("\nCorrelations of levels and of daily log returns:")
    print(corr.round(3).to_string(index=False))
    print("\nShare of variance explained by PC1:")
    print(pcs.round(3).to_string(index=False))

    # equally weighted KASE stocks vs USD/KZT, weekly returns
    px, src = loaders.raw_prices("kase")
    px = px.ffill()
    ew_daily = px.pct_change().mean(axis=1, skipna=True)        # equal weights over the stocks quoted that day
    ew_week = (1 + ew_daily).resample("W-FRI").prod() - 1
    usd_week = rates["USD"].reindex(px.index, method="ffill").resample("W-FRI").last().pct_change()
    both = pd.concat({"ew_kase": ew_week, "usdkzt": usd_week}, axis=1).dropna()
    both = both[(both != 0).any(axis=1)]
    hedge = pd.DataFrame([{"period": f"{both.index.min():%Y-%m-%d} – {both.index.max():%Y-%m-%d}", "n_weeks": len(both),
                           "corr_weekly": both["ew_kase"].corr(both["usdkzt"]),
                           "corr_weekly_spearman": both["ew_kase"].corr(both["usdkzt"], method="spearman"),
                           "corr_daily": ew_daily.corr(rates["USD"].reindex(px.index, method="ffill").pct_change()),
                           "stocks": ",".join(px.columns), "source": src}])
    hedge.to_csv(out / "fx_returns_kase_hedge.csv", index=False)
    print("\nEqually weighted KASE stocks vs USD/KZT:")
    print(hedge.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
