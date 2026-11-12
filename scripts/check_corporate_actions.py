"""Check dividend ex-dates in data/corporate_actions.csv against the raw KASE closes.

For every ``dividend`` event the script finds, within +-3 trading days of ``ex_date``, the
day with the largest one-day price fall in data/prices_kase.csv (unadjusted closes) and
compares it with the expected fall ``factor / last close before ex_date``. It then applies
all corporate actions and lists the daily moves of the adjusted closes above a threshold.

    python scripts/check_corporate_actions.py            # plain-text tables
    python scripts/check_corporate_actions.py --md       # Markdown tables (for docs/)
    python scripts/check_corporate_actions.py --write    # rebuild the dividend rows first

``--write`` regenerates the ``dividend`` rows of data/corporate_actions.csv from the exchange's
dividend history (data/dividends_kase.csv, ``python -m dss.cli fetch kase-dividends``): the
amount per share and the record date are taken as published, the ex-date follows
:func:`dss.data.loaders.kase_ex_date`. Rows with other methods are kept as they are.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dss.data import loaders  # noqa: E402

WINDOW = 3          # trading days on each side of ex_date
BIG_MOVE = 0.25     # |daily change| of adjusted closes reported as suspicious


def trading_calendar(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """KASE trading days: weekdays present in the price file, plus weekend days on which at
    least half of the listed shares changed price (the official working Saturdays/Sundays of
    Kazakhstan); single stray weekend bars of one or two shares are not trading days."""
    listed = raw.ffill().notna().sum(axis=1)          # shares that have started trading
    traded = raw.notna().sum(axis=1)                  # shares with a deal that day
    keep = (raw.index.dayofweek < 5) | (traded >= listed / 2)
    return raw.index[keep]


def trading_days(s: pd.Series) -> pd.Series:
    """Drop forward-filled repeats at the start of the series and missing values."""
    return s.dropna()


def load_raw() -> pd.DataFrame:
    """Unadjusted closes of data/prices_kase.csv joined with data/prices_kase_extra.csv (if any)."""
    frames = [loaders._dated_frame(p, "Date,<тикер1>,…") for p in
              (loaders.data_dir() / "prices_kase.csv", loaders.data_dir() / "prices_kase_extra.csv") if p.exists()]
    return pd.concat(frames, axis=1, sort=True)


def check_dividends(raw: pd.DataFrame, actions: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    rows = []
    for a in actions[actions["method"] == "dividend"].itertuples():
        if a.ticker not in raw:
            continue
        s = trading_days(raw[a.ticker])
        pos = int(s.index.searchsorted(a.ex_date))       # first trading day >= ex_date
        if pos == 0 or pos >= len(s):
            rows.append({"ticker": a.ticker, "ex_date": a.ex_date, "drop_day": pd.NaT, "dividend": a.factor,
                         "close_before": float("nan"), "expected": float("nan"), "fall_ex": float("nan"),
                         "fall_max": float("nan"), "match": "нет цен"})
            continue
        r = s.pct_change()                               # change since the previous trade
        lo, hi = max(pos - window, 1), min(pos + window, len(s) - 1)
        win = r.iloc[lo:hi + 1]
        drop_day = win.idxmin()
        close_before = s.iloc[pos - 1]
        expected = a.factor / close_before
        fall_ex, fall_max = -r.iloc[pos], -win.min()
        same_day = drop_day == s.index[pos]
        # the ex-date fall should be the window's largest and carry a sizeable part of the dividend
        if same_day and fall_ex >= 0.3 * expected:
            match = "да"
        elif same_day:
            match = "день совпадает, падение мало"
        else:
            match = "нет"
        rows.append({"ticker": a.ticker, "ex_date": a.ex_date, "drop_day": drop_day, "dividend": a.factor,
                     "close_before": close_before, "expected": expected, "fall_ex": fall_ex,
                     "fall_max": fall_max, "match": match})
    return pd.DataFrame(rows)


def build_dividend_actions(div: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Dividend rows of corporate_actions.csv for the tickers and the period of ``raw``."""
    calendar = trading_calendar(raw)
    div = div.assign(record_date=pd.to_datetime(div["record_date"]))
    rows = []
    for (t, rec), g in div.groupby(["ticker", "record_date"], sort=True):
        if t not in raw:
            continue
        first = raw[t].first_valid_index()
        if first is None or rec <= first or rec > calendar[-1]:
            continue
        t2_from = loaders.KASE_T2_FROM.get(t, loaders.KASE_T2_DEFAULT)
        try:
            ex = loaders.kase_ex_date(rec, calendar, t2_from)
        except ValueError:
            continue
        if ex <= first:
            continue
        amount = round(float(g["per_share"].sum()), 6)
        scheme = "T+2" if t2_from is not None and ex >= pd.Timestamp(t2_from) else "T+0"
        parts = " + ".join(f"{v:g}" for v in g["per_share"]) if len(g) > 1 else ""
        note = (f"KASE record date {rec:%Y-%m-%d} (register as of 00:00); {scheme} settlement -> ex-date "
                f"{ex:%Y-%m-%d}; KZT {amount:g} per share" + (f" ({parts})" if parts else "") +
                f"; source {loaders.KASE_DIVIDENDS_URL.format(ticker=t)}")
        rows.append({"ticker": t, "ex_date": ex.strftime("%Y-%m-%d"), "method": "dividend",
                     "factor": amount, "note": note})
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "method", "factor", "note"])


def write_actions(raw: pd.DataFrame) -> pd.DataFrame:
    path = loaders.data_dir() / "corporate_actions.csv"
    div = pd.read_csv(loaders.data_dir() / "dividends_kase.csv")
    new = build_dividend_actions(div, raw)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["method"].str.strip().str.lower() != "dividend"], new], ignore_index=True)
    new = new.sort_values(["ticker", "ex_date"]).reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def big_moves(adj: pd.DataFrame, threshold: float = BIG_MOVE) -> pd.DataFrame:
    r = adj.ffill().pct_change()[adj.notna()].stack()   # change since the previous trade
    r = r[r.abs() > threshold]
    return r.rename("change").reset_index().rename(columns={"level_1": "ticker"})


def _fmt(df: pd.DataFrame, md: bool) -> str:
    if df.empty:
        return "(нет)"
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d").fillna("—")
    if not md:
        return out.to_string(index=False)
    head = "| " + " | ".join(out.columns) + " |\n|" + "---|" * len(out.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--md", action="store_true", help="Markdown tables")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--threshold", type=float, default=BIG_MOVE)
    p.add_argument("--write", action="store_true", help="rebuild dividend rows from data/dividends_kase.csv")
    a = p.parse_args(argv)

    raw = load_raw()
    if a.write:
        n = (write_actions(raw)["method"] == "dividend").sum()
        print(f"corporate_actions.csv: {n} дивидендов пересобрано из dividends_kase.csv\n")
    actions = loaders.load_corporate_actions()
    if actions is None:
        print("нет data/corporate_actions.csv")
        return 1
    res = check_dividends(raw, actions, a.window)
    show = res.assign(
        dividend=res["dividend"].map(lambda v: f"{v:g}"),
        expected=res["expected"].map(lambda v: f"{v:.1%}"),
        fall_ex=res["fall_ex"].map(lambda v: f"{v:.1%}"),
        fall_max=res["fall_max"].map(lambda v: f"{v:.1%}"),
    ).drop(columns="close_before").rename(columns={
        "ticker": "тикер", "ex_date": "ex_date (файл)", "drop_day": "день макс. падения",
        "dividend": "дивиденд, ₸", "expected": "ожид. падение", "fall_ex": "падение в ex_date",
        "fall_max": "макс. падение", "match": "совпадает"})
    print(f"Дивиденды: {len(res)}, совпадают: {(res['match'] == 'да').sum()}\n")
    print(_fmt(show, a.md))
    adj, _ = loaders.apply_corporate_actions(raw, actions)
    moves = big_moves(adj, a.threshold)
    print(f"\nДневные изменения скорректированных цен больше {a.threshold:.0%} по модулю:\n")
    print(_fmt(moves.assign(change=moves["change"].map(lambda v: f"{v:+.1%}")), a.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
