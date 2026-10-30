"""Computational performance of the DSS on the committed data (one process).

Measures, with wall-clock time and the median of ``--repeats`` runs where the
operation is cheap enough:

* Module I: ``macro.run`` on the full GDP panel;
* Module IV: ``signals.run`` in the walk-forward protocol for 3 and 8
  scikit-learn models on the KASE prices and for 3 models on the NSE prices;
* a cold ``pipeline.run`` (empty cache, all modules, including the Module IV
  backtest shown on the signals screen), a repeated run with a warm cache (a
  recommendation for cached settings), the integration step alone
  (``integration.decide``) and a change of the investor profile with and
  without the Module IV backtest, which depends on the profile limits and is
  recomputed for a new profile.

A private cache directory is used, so the cache of the application is untouched.

    python scripts/timing.py                 # writes docs/results/timing.csv
    python scripts/timing.py --repeats 3
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dss import config, integration, macro, pipeline, signals  # noqa: E402
from dss.data import loaders  # noqa: E402

SK8 = ("linear", "ridge", "lasso", "svr", "dt", "rf", "gb", "mlp")


def timed(fn, repeats: int) -> tuple[float, list[float]]:
    """Median and all wall-clock times of ``repeats`` calls of ``fn``."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repeats", type=int, default=5, help="repeats of the cheap operations (median is reported)")
    p.add_argument("--out", default=str(ROOT / "docs" / "results"))
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline.CACHE_DIR = Path(tempfile.mkdtemp(prefix="dss-timing-"))
    rows = []

    def add(name, med, times, note=""):
        rows.append({"operation": name, "seconds_median": med, "repeats": len(times),
                     "seconds_min": min(times), "seconds_max": max(times), "note": note})
        print(f"  {name}: {med:.2f} s (median of {len(times)}; min {min(times):.2f}, max {max(times):.2f}) {note}")

    gdp = loaders.load_gdp().frame
    kase = loaders.load_prices("kase").frame
    nse = loaders.load_prices("nse").frame
    print(f"Data: GDP panel {gdp.shape[0]} economies x {gdp.shape[1]} years; KASE {kase.shape[1]} stocks x {len(kase)} days; "
          f"NSE {nse.shape[1]} stocks x {len(nse)} days")

    print("Module I")
    add("Module I: macro.run (full GDP panel)", *timed(lambda: macro.run(gdp), a.repeats), f"{gdp.shape[0]} economies")

    print("Module IV, walk-forward protocol")
    wf = config.PROTOCOLS["walk_forward"]
    for label, px, models in (("KASE, 3 models (ridge, rf, gb)", kase, ("ridge", "rf", "gb")),
                              ("KASE, 8 scikit-learn models", kase, SK8),
                              ("NSE, 3 models (ridge, rf, gb)", nse, ("ridge", "rf", "gb"))):
        n = 3 if len(px) < 2500 else 1
        add(f"Module IV: signals.run, {label}", *timed(lambda: signals.run(px, models, wf["train_share"], wf["retrain_every"]), n),
            f"{px.shape[1]} stocks")

    print("Pipeline (moderate profile, KASE)")
    st = config.Settings()

    def cold():
        pipeline.clear_cache()
        pipeline.run(st).backtest
    add("pipeline.run, cold cache (all modules and the Module IV backtest)", *timed(cold, 1))
    add("pipeline.run, warm cache (recommendation for cached settings)",
        *timed(lambda: pipeline.run(st).backtest, a.repeats))
    r = pipeline.run(st)
    add("integration.decide alone", *timed(lambda: integration.decide(r.macro, r.fx, r.clusters, r.signals, r.prices, st, r.sectors),
                                           a.repeats))
    other = config.Settings(profile="conservative")
    add("change of investor profile, decision only", *timed(lambda: pipeline.run(other), 1), "Modules I-IV reused from the cache")
    add("change of investor profile, with the Module IV backtest", *timed(lambda: pipeline.run(other).backtest, 1),
        "backtest recomputed for the new limits")
    add("same profile change, cached", *timed(lambda: pipeline.run(other).backtest, a.repeats))

    df = pd.DataFrame(rows)
    df.attrs["machine"] = f"{platform.processor() or platform.machine()}, {platform.python_version()}"
    df.to_csv(out / "timing.csv", index=False)
    info = pd.Series({"python": platform.python_version(), "platform": platform.platform(),
                      "cpu": platform.processor() or platform.machine(), "numpy": np.__version__, "pandas": pd.__version__,
                      "date": time.strftime("%Y-%m-%d")})
    info.to_csv(out / "timing_environment.csv", header=False)
    print(f"\nWritten {out / 'timing.csv'}; {info['platform']}, Python {info['python']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
