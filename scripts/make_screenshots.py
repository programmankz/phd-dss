"""Screenshots of the web interface for the presentation / dissertation.

1) start the app:   python -m webapp.app
2) run:             python scripts/make_screenshots.py [--out docs/screenshots]
Requires: pip install playwright && playwright install chromium
(or pass --chromium /path/to/chrome to use an already installed browser)
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {"decision": "/", "macro": "/macro", "fx": "/fx", "clusters": "/clusters",
         "signals": "/signals", "data": "/data"}

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8050")
ap.add_argument("--out", default="docs/screenshots")
ap.add_argument("--query", default="", help="e.g. profile=conservative&scenario=stress or universe=nse")
ap.add_argument("--chromium", default=None, help="path to an installed Chromium instead of Playwright's own")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=a.chromium)
    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    for name, path in PAGES.items():
        pg.goto(f"{a.base}{path}?{a.query}", timeout=300_000)
        pg.screenshot(path=out / f"{name}.png")
        pg.screenshot(path=out / f"{name}_full.png", full_page=True)
        print("saved", name)
    b.close()
