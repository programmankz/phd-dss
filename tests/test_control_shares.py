"""Constant-share control portfolios: Config.h_fixed overrides the dollar share, default is unchanged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import end_to_end as e2e  # noqa: E402


def _engine():
    days = pd.bdate_range("2014-06-02", "2015-06-30")
    prices = pd.DataFrame({"A": np.linspace(100, 120, len(days)), "B": 50.0}, index=days)
    cal = pd.date_range("2008-01-01", "2015-07-10")
    fxr = pd.DataFrame({"USD": 180.0, "EUR": 240.0, "CNY": 29.0, "RUB": 5.0}, index=cal)
    rates = pd.DataFrame({"kzt": 0.055, "usd": 0.001}, index=cal)
    gdp = pd.DataFrame({y: [5.0, 3.0] for y in range(2000, 2015)}, index=["Kazakhstan", "Other"])
    inp = e2e.Inputs(prices, fxr["USD"].reindex(days), fxr, gdp, rates, "test", None, None)
    return e2e.Engine(inp, "moderate", "2014-06-02", "2015-06-30", need_m4=False, log=lambda *a: None)


def test_h_fixed_overrides_the_currency_rule_and_is_off_by_default():
    eng, t = _engine(), pd.Timestamp("2015-02-02")
    assert e2e.Config("x", False, False, False).h_fixed is None
    rule = eng.decide(t, e2e.Config("rule", False, True, False))
    const = eng.decide(t, e2e.Config("const", False, True, False, h_fixed=0.37))
    no_fx = eng.decide(t, e2e.Config("const no fx", False, False, False, h_fixed=0.37))
    assert const["h"] == 0.37 and no_fx["h"] == 0.37
    assert const["d"] == rule["d"]
    assert eng.decide(t, e2e.Config("plain", False, False, False))["h"] == 0.0
