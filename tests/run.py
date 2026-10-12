"""Minimal test runner for environments without pytest: python tests/run.py"""
import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

failed = skipped = 0
for path in sorted(HERE.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        t0 = time.perf_counter()
        try:
            getattr(module, name)()
            print(f"ok    {path.stem}.{name}  ({time.perf_counter() - t0:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL  {path.stem}.{name}")
            traceback.print_exc()
        except BaseException as exc:      # pytest.skip() raises a BaseException subclass
            if type(exc).__name__ != "Skipped":
                raise
            skipped += 1
            print(f"skip  {path.stem}.{name}: {exc}")
note = f" ({skipped} skipped)" if skipped else ""
print(f"\n{failed} failed{note}" if failed else f"\nall passed{note}")
sys.exit(1 if failed else 0)
