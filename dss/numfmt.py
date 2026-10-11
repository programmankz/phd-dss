"""Russian number formatting for user-facing text: decimal comma, a space as the
thousands separator. Machine-readable CSV output keeps the decimal point and
must not use these helpers."""
from __future__ import annotations

import numbers

import numpy as np

DASH = "—"


def _missing(v) -> bool:
    return v is None or (isinstance(v, numbers.Real) and not isinstance(v, numbers.Integral)
                         and not np.isfinite(float(v)))


def num(v, d: int = 2, sign: bool = False) -> str:
    """1234.5 -> '1 234,50'; integers without decimals; NaN/None -> '—'."""
    if _missing(v):
        return DASH
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, numbers.Integral):
        return f"{int(v):+,}".replace(",", " ") if sign else f"{int(v):,}".replace(",", " ")
    s = f"{float(v):+,.{d}f}" if sign else f"{float(v):,.{d}f}"
    return s.replace(",", " ").replace(".", ",")


def pct(v, d: int = 1, sign: bool = False) -> str:
    """Fraction -> percent: 0.2161 -> '21,6%'."""
    if _missing(v):
        return DASH
    return num(float(v) * 100, d, sign) + "%"
