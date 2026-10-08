"""Minimal test runner for environments without pytest: python tests/run.py"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_core  # noqa: E402

failed = 0
for name in sorted(n for n in dir(test_core) if n.startswith("test_")):
    try:
        getattr(test_core, name)()
        print(f"ok    {name}")
    except Exception:
        failed += 1
        print(f"FAIL  {name}")
        traceback.print_exc()
print(f"\n{failed} failed" if failed else "\nall passed")
sys.exit(1 if failed else 0)
