# Plan for the next Claude Code session (cloud)

You are continuing work on **phd-dss**, the software part of a PhD dissertation: a decision support system for investment portfolio management (four ML modules + an integration module + a Russian-language Flask UI). Read `README.md`, `docs/SETUP.md`, `docs/methodology.md`, `docs/progress.md` and `docs/techspecs.md` before changing anything — they are the source of truth for what the code is supposed to do and why.

The main goal of this session is **stage 3 of `docs/progress.md`: reproduce articles [A1]–[A4] on real data and document how the system's numbers compare with the published ones.** Everything else is secondary.

---

## 0. What changed since `docs/progress.md` was last updated

The previous session (local, on the author's Mac) did stage 2 — real data — and committed the data files:

| File in `data/` | Content | Source |
|---|---|---|
| `gdp_growth.csv` | Real GDP growth, %, 214 countries, 1980–2024 | World Bank WDI `NY.GDP.MKTP.KD.ZG` (IMF API returns 403 to scripts; `fetch gdp` now falls back to WB) |
| `fx_rates.csv` | KZT per USD/EUR/CNY/RUB, daily, 2000-01-01 – 2026-10-12 | National Bank of Kazakhstan |
| `prices_nse.csv` | 20 NSE stocks of [A4] Table II, 2020-01-01 – 2026-10-09 | Yahoo Finance |
| `prices_kase.csv` | 9 KASE stocks (HSBK KSPI CCBN KZTK KCEL KEGC KZTO KMGZ KZAP), 2019-01-03 – 2026-10-09; common history from 2022-12-08 (≈ 950 days) | kase.kz chart feed (`/tv-charts/securities`) |
| `fundamentals_kase.csv` | Sectors only; **P/B and P/E are empty** | — |

Code changes already on `main`: `loaders.fetch_gdp` (IMF → World Bank fallback), `loaders.fetch_kase` + `python -m dss.cli fetch kase`, `TATAMOTORS.NS` → `TMPV.NS` (renamed after the Oct 2025 demerger), `loaders.adjust_demergers` (back-adjusts TMPV before 2025-10-14, otherwise a fake −40% day), new guide `docs/SETUP.md`. 45 tests pass.

**Known data caveats** (must be stated wherever results are reported):

1. KASE prices are **not dividend-adjusted** (as KASE publishes them). Large dividend payers (KEGC, KZTO, HSBK, KZAP) have understated returns.
2. **KZTK fell 41% on 2025-05-22/23** (58 400 → 34 279 ₸) and stayed there. Probably a corporate event, not verified. Treat KZTK results as suspect; run KASE analyses with and without KZTK.
3. GDP data are World Bank, not the IMF WEO / `world_gdp_data.csv` used in [A1] — small differences are expected.
4. NSE data run to 2026-10; [A4] covers 2020–2025.

## 1. Ground rules

- **Do not fetch data.** The cloud network blocks `imf.org`, `nationalbank.kz`, `kase.kz` and Yahoo Finance (see `docs/progress.md`). Use the committed CSVs. Do not modify files in `data/`.
- **Never invent article numbers.** The only published reference values available in the repo are listed in §3. If a comparison needs a number that is not in the repo, write «нужно значение из статьи» in the table and list it in the final summary as a question for the author.
- Work on a new branch (e.g. `claude/stage3-reproduction`); never push to `main`. Commit after each phase with a clear message. Open **one PR** at the end.
- After every code change: `python -m pytest -q` and `ruff check .` must pass. Add tests for new code in the style of `tests/` (they also run without pytest via `python tests/run.py`).
- Match the codebase: Python code and comments in English; user-facing strings, docs and the UI in **Russian**; numbers in Russian docs with a decimal comma.
- Reproduction results are a deliverable for the dissertation. Be precise and honest: if the system does not reproduce a published number, say so and explain the most likely reason; do not tune parameters to hit the published value.
- If a step is blocked (missing data, a package won't install), document it and move on to the next phase.

## 2. Setup

```bash
python --version                                   # need >= 3.10; lock file needs >= 3.12
python -m pip install -r requirements-lock.txt     # on 3.10–3.11: requirements.txt
python -m pip install pytest
python -m pytest -q                                # expect 45 passed
python -m dss.cli warmup                           # sources must all read "CSV: …" (no DEMO)
python -m dss.cli warmup --universe nse
```

If `warmup` shows any `DEMO:` source, stop and find out why before continuing.

## 3. Phase A — reproduce the articles (main task)

Create `scripts/reproduce.py` that computes every comparison below from the committed data and writes:
- machine-readable results to `docs/results/*.csv`;
- a Russian report `docs/reproduction.md`: for each article a table «статья / СППР / разница / объяснение», the exact settings and data period used, and the caveats from §0.

Make the script deterministic (seed from `config.RANDOM_SEED`) and runnable as `python scripts/reproduce.py`. Reuse the module functions; do not duplicate their logic.

### A2 — Module II, exchange rates (start here: simplest, data match the article's source)

- Period: **2000-01-01 – 2024-12-31** (the article's period; the file runs longer — slice it).
- Compute `fx.describe` (Table 1, Excel conventions) and `fx.trend_regression` (slope in ₸/year and R²) for USD, EUR, CNY, RUB.
- Published R² (from `docs/techspecs.md`): **USD 0,78; EUR 0,85; CNY 0,84; RUB 0,21**. Compare. Table 1 values themselves are not in the repo → «нужно значение из статьи».
- Also report `fx.detect_shocks` output and check it finds the known devaluations (2009, 2014–2015) — state what it finds, don't assert.

### A4 — Module IV, NSE

- Run both protocols on `--universe nse`:
  - article mode: `config.PAPER_PRESET` (`protocol=paper`, sample covariance, daily rebalance, tc 0, no caps) — same as `python -m dss.cli significance --universe nse --protocol paper --cov sample --rebalance D --tc 0 --no-caps`;
  - default walk-forward.
- Run once on data cut at **2025-12-31** (article period) and once on the full file; report both.
- Strategies: Signal Weighted, Max Signal-to-Risk, Min Variance, Equal Weighted, market. Metrics: total and annual return, volatility, Sharpe (`config.RISK_FREE` = 5%), max drawdown, turnover; plus the bootstrap Sharpe CIs and Diebold–Mariano tests from `dss.significance`.
- The only Table IV figures in the repo: **annual return 21,61 %, volatility 17,40 %, Sharpe 0,95** (`docs/methodology.md`, used to infer the 5% risk-free rate; the row is the article's best strategy — confirm which one from the code/docs, otherwise mark it unknown). Everything else in Table IV → «нужно значение из статьи».
- Note in the report: TMPV.NS stands in for TATAMOTORS.NS and is demerger-adjusted; explain discrepancy 4 of `docs/methodology.md` (out-of-sample period) with the numbers you got.
- Optional: try `pip install tensorflow` and run the 11-model set; if it installs, record whether RNN/LSTM/CNN-LSTM/LSTM-MLP train without errors (closes FT-IV-03). If not, note it and skip.

### A3 — Module III, KASE (fills the published Table 2 that only has X/Y/Z placeholders)

- Run `clustering.cluster_assets` and `clustering.cluster_portfolio_backtest` on `prices_kase.csv` (K-means, K-medoids/PAM, Fuzzy C-means, equal-weight benchmark).
- Report Table 2: annual return, volatility, Sharpe, turnover, max drawdown per method; chosen k with silhouette and Davies–Bouldin; cluster membership per ticker.
- Robustness: repeat **without KZTK**, and without KMGZ (longer history from 2021-07). Show all three side by side.
- State plainly: clustering uses price features only (no P/B, P/E); prices not dividend-adjusted; period 2022-12 – 2026-10.

### A1 — Module I, GDP

- Run Module I on `gdp_growth.csv` (World Bank). Report recession classification (accuracy, AUC per model, McNemar), growth regression (R², RMSE per model), ARIMA selection, Kazakhstan 2025–2029 forecast vs `config.IMF_WEO_KZ_BENCHMARK`.
- Compare with both versions in the article (discrepancy 1 in `docs/methodology.md`): text/abstract **97,9 % / AUC 0,997** vs Fig. 9 **LogReg 0,896 / AUC 0,762**, RF/XGBoost/LightGBM 0,14–0,17, Fig. 11 all R² < 0. Say which one the system's results are closer to.
- Clearly mark that exact reproduction needs the article's original `world_gdp_data.csv` (not in the repo) — put it in the questions for the author.

### Wrap-up of phase A

- Add the findings to `docs/methodology.md` → «Известные расхождения и решения» (new numbered items, short, linking to `docs/reproduction.md`).
- Commit.

## 4. Phase B — offline improvements (do as many as time allows, in this order)

1. **Corporate-action adjustments as data, not code.** Generalise `loaders.adjust_demergers` / `NSE_DEMERGERS`: an optional `data/corporate_actions.csv` (`ticker,ex_date,method,factor,note`; `method` = `ratio` (observed close ratio, current behaviour) or `factor` (explicit multiplier)), applied in `load_prices` for any universe, validated with readable `DataError` messages like the other loaders. Keep the TMPV rule working (move it into the file or keep it as a built-in default — your call, document it). Do **not** add a KZTK entry — the event is unverified; document in `docs/SETUP.md` how the author can add one once verified. Tests.
2. **NFR-01 reproducibility test:** two consecutive runs of the pipeline give identical numbers to 4 decimal places (decision weights, Sharpe, cluster labels). Use the demo data so the test does not depend on `data/`.
3. **Russian number format** in the HTML report and console output (decimal comma), as already done in the web UI (`docs/progress.md`, row «Числа в русском формате»). Do not break machine-readable CSV output.
4. **Risk-free rate for KASE** (FT-IV-12): make the risk-free rate depend on the universe via config (`RISK_FREE` stays 5 % for NSE). Use the NBK base rate only if a value with a source is already in the repo; otherwise add the setting with a clearly marked placeholder and ask the author for the value. Do not guess the rate.
5. **CI `startup_failure`:** inspect with `gh run list` / `gh api repos/programmankz/phd-dss/actions/runs` (and `.../actions/permissions`). If the cause is a repository/account setting, write the exact steps the author must take; don't change `ci.yml` unless you find a real defect in it.

## 5. Phase C — documentation and PR

- Update `docs/progress.md`: stage 2 → done (with the caveats), stage 3 → status per article, checklist items, test count, and remove outdated statements (e.g. «Данные не загружены», «отдельного руководства пользователя нет» — `docs/SETUP.md` now exists).
- Update `README.md` if commands or files changed; link `docs/reproduction.md`.
- Screenshots (`scripts/make_screenshots.py`, needs `pip install playwright && playwright install chromium`): if Chromium installs in the cloud, start the app, retake all screenshots on real data for both KASE and NSE, commit them. If not, skip and note it.
- Open the PR. Its description: what was done per phase, a short table of headline reproduction results, caveats, and a **«Вопросы к автору»** list (missing article values, `world_gdp_data.csv`, KZTK event, NBK rate, P/B and P/E for KASE, CI settings).

## 6. Definition of done

- `python scripts/reproduce.py` regenerates `docs/results/` and `docs/reproduction.md` from scratch.
- `docs/reproduction.md` covers A1–A4 with settings, numbers, comparisons, explanations and caveats; no invented reference values.
- Tests and ruff pass; new code has tests.
- `docs/progress.md` and `docs/methodology.md` reflect the new state.
- One PR from a feature branch with a clear summary and the list of questions for the author.
