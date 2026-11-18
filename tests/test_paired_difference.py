"""Test of the paired bootstrap difference test (dss.evaluation.paired_difference_test)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss import evaluation  # noqa: E402


def test_paired_difference_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2020-01-01", periods=800)
    base = pd.Series(rng.normal(0.0003, 0.01, 800), index=d)
    better = base + 0.001
    mean = lambda r: np.asarray(r).mean(axis=-1)          # noqa: E731
    t = evaluation.paired_difference_test(better, base, mean, n_boot=500)
    assert np.isclose(t["diff"], 0.001) and t["p_greater"] < 0.01 and t["low"] > 0
    noise = base + rng.normal(0, 1e-4, 800)
    t0 = evaluation.paired_difference_test(noise, base, mean, n_boot=500)
    assert t0["p_two_sided"] > 0.05
    # non-inferiority: base is worse than better by 0.001, i.e. not worse by more than the margin 0.0015
    ni = evaluation.paired_difference_test(base, better, mean, null=-0.0015, n_boot=500)
    assert ni["p_greater"] < 0.05
