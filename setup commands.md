# ChargedUp Profits Bot — Manual Command Reference

All commands are run from the **project root folder** (`c:\Apps\AI\trading_bot`).
Open a terminal there and use `python` (or `py` on Windows).

---

## Virtual Environment

| Action | Command |
|--------|---------|
| **Activate venv** | `.\venv\Scripts\Activate.ps1` |
| **Deactivate venv** | `deactivate` |

> Always activate the venv before running any bot commands.

### Rebuilding a Broken venv
If you see `Fatal error in launcher: Unable to create process` or `No Python at '...'`, the venv launchers have stale hardcoded paths (this happens if the project folder is renamed/moved). Fix it by rebuilding the venv in place:

```powershell
# 1. Save current package list
python -m pip freeze > requirements_backup.txt

# 2. Delete the broken venv
Remove-Item -Recurse -Force venv

# 3. Recreate it with the correct Python
python -m venv venv

# 4. Reinstall all packages
.\venv\Scripts\python.exe -m pip install -r requirements_backup.txt
```

After rebuilding, activate normally and all commands work again.

---

## Dashboard (Web Panel)

| Action | Command |
|--------|---------|
| **Start the dashboard** | `streamlit run monitoring/dashboard.py` |
| **Stop the dashboard** | `Ctrl + C` in the terminal running it |
| **Restart the dashboard** | `Ctrl + C`, then run the start command again |

Open in browser: **http://localhost:8501**

---

## Live / Paper Trading Bots

### Single Symbol

| Action | Command |
|--------|---------|
| Start ES bot | `python src/bot_runner.py --symbol ES` |
| Start NQ bot | `python src/bot_runner.py --symbol NQ` |
| Start MNQ bot | `python src/bot_runner.py --symbol MNQ` |
| Stop any bot | `Ctrl + C` in the terminal running it |

### Multi-Symbol (Recommended — shares one connection)

| Action | Command |
|--------|---------|
| Start ES + NQ together | `python src/multi_bot_runner.py` |
| Start ES + NQ + MNQ | `python src/multi_bot_runner.py --symbols ES NQ MNQ` |
| Start ES + MNQ only | `python src/multi_bot_runner.py --symbols ES MNQ` |
| Stop all | `Ctrl + C` in the terminal running it |

> **Note:** Only run ONE connection at a time (single or multi). Starting two separate bot_runner.py processes will disconnect the first one.

---

## AI Re-Learning (Background Daemon)

| Action | Command |
|--------|---------|
| Start AI learning (ES + NQ) | `python src/continuous_learner.py` |
| Start AI learning (specific symbols) | `python src/continuous_learner.py --symbols ES NQ MNQ` |
| Stop AI learning | `Ctrl + C` in the terminal running it |

The learner only retrains during market downtime. Status writes to `data/learner_state.json`.
New models appear in the **Test & Train AI** tab for your approval before going live.

---

## Backtesting

> ⚠️ **Important**: Standard backtests run with the ML model applied. If the ML was trained on the same
> date range being tested, the results will be inflated (92–96% WR is data leakage, not real performance).
> Use **Clean Eval** below for honest results.

| Action | Command |
|--------|---------|
| Backtest ES (raw, 2022–2024) | `python backtest/harness.py --symbol ES --start 2022-01-03 --end 2024-12-31` |
| Backtest NQ (raw) | `python backtest/harness.py --symbol NQ --start 2022-01-03 --end 2024-12-31` |
| Backtest MNQ (raw) | `python backtest/harness.py --symbol MNQ --start 2022-01-03 --end 2024-12-31` |
| Backtest all symbols at once | `python backtest/run_all.py --start 2022-01-03 --end 2024-12-31` |
| Backtest specific symbols only | `python backtest/run_all.py --start 2022-01-03 --end 2024-12-31 --symbols ES NQ` |

---

## Clean Eval (Honest Forward-Test — Use This for Real Performance)

Trains the ML on pre-cutoff data only, then tests on a period the model has **never seen**.
This gives the true out-of-sample performance, free from data leakage.

| Action | Command |
|--------|---------|
| **Honest NQ holdout (2024–2025)** | `python backtest/clean_eval.py --symbol NQ --cutoff 2024-01-01 --end 2025-12-31` |
| **Honest ES holdout (2024–2025)** | `python backtest/clean_eval.py --symbol ES --cutoff 2024-01-01 --end 2025-12-31` |
| Custom cutoff date | `python backtest/clean_eval.py --symbol NQ --cutoff 2023-01-01` |

**Current honest results (2024–2025, after setup filter — SWEEP_REVERSE disabled on both symbols):**

| Symbol | Setup | Trades | Win% | Mean R | Total R |
|--------|-------|--------|------|--------|---------|
| ES | BREAK_RETEST | 182 | 44% | +0.194 | +35.3R |
| ES | BOUNCE | 5 | 80% | +0.745 | +3.7R |
| **ES total** | — | **187** | **44.7%** | +0.209 | **+39.1R** |
| NQ | BREAK_RETEST | 133 | 40.6% | +0.099 | +13.1R |
| NQ | BOUNCE | 9 | 44.4% | −0.010 | −0.09R |
| **NQ total** | — | **142** | **41.0%** | +0.092 | **+13.0R** |

> **Setup filter (v2026-02):** SWEEP_REVERSE disabled on both ES and NQ — showed 34.5% WR and 35.1% WR
> respectively in OOS 2024–2025. Re-enable when 200+ live trades show positive expectancy.
>
> **ML gate:** ES bypasses ML entirely (`ml_bypass: true`) — model AUC ≈ 0.50, no discriminative value yet.
> NQ ML gate is active but in pass-through mode until live trade count grows.
>
> Training dataset: 852 trades, 59% win rate, +0.750 mean_R (clean, honest labels).

---

## Walk-Forward Testing

The WFA now **retrains the ML model for each window** using only that window's training period.
This gives honest out-of-sample results per window.

| Action | Command |
|--------|---------|
| Walk-forward ES | `python backtest/walk_forward.py --symbol ES --start 2022-01-03 --end 2024-12-31` |
| Walk-forward NQ | `python backtest/walk_forward.py --symbol NQ --start 2022-01-03 --end 2024-12-31` |
| Walk-forward all symbols | `python backtest/wf_all.py --start 2022-01-03 --end 2024-12-31` |

---

## AI / ML Model Training

### Full Pipeline (Recommended — use after each major data update)

| Step | Command | Description |
|------|---------|-------------|
| **Step 1 — Rebuild backtests** | `python scripts/rebuild_backtests.py --symbols ES NQ` | Regenerates CSVs with all 57 features (run whenever feature_builder changes) |
| **Step 2 — Build dataset** | `python scripts/build_dataset.py` | Labels trades using `realized_R_binary`, expands features_json, enforces schema |
| **Step 3 — Train models** | `python scripts/train_models.py --symbols ES NQ` | Trains with purged CV + calibration + EV policy. Models go to `pending/` for approval. |
| **Optional — Activate immediately** | `python scripts/train_models.py --symbols ES NQ --activate` | Skips pending approval; writes directly to `active/` |
| **Check dataset report** | Open `data/build_dataset_report.json` | Shows schema_hash, label stats, coverage |
| **Check train report** | Open `data/train_models_report.json` | Shows AUC, calibration, EV policy per model |

### Quick Retrain (via legacy command — uses continuous learner path)

| Action | Command |
|--------|---------|
| Full retrain ES (label + train + evaluate) | `python src/ml_filter.py retrain --symbol ES` |
| Full retrain NQ | `python src/ml_filter.py retrain --symbol NQ` |
| Full retrain MNQ | `python src/ml_filter.py retrain --symbol MNQ` |
| Retrain all symbols at once | `python src/ml_retrain_all.py` |
| Label only (no training) | `python src/ml_filter.py label --symbol ES` |
| Train only (skip labeling) | `python src/ml_filter.py train --symbol ES` |
| Evaluate existing model | `python src/ml_filter.py eval --symbol ES` |

### Run unit tests

| Action | Command |
|--------|---------|
| Purged CV self-test | `python -m src.cv.purged_time_series_split` |

---

## Historical Data Download

| Action | Command |
|--------|---------|
| Download all symbols | `python data/download_historical.py --symbols ES NQ MNQ` |
| Download ES only | `python data/download_historical.py --symbols ES` |
| Download NQ only | `python data/download_historical.py --symbols NQ` |

---

## Typical Daily Workflow

```
# 1. Start the dashboard (keep open in a tab the whole time)
streamlit run monitoring/dashboard.py

# 2. Start the bots before 9am ET
python src/multi_bot_runner.py

# 3. After market close — let AI re-learn overnight (optional, runs automatically from dashboard)
python src/continuous_learner.py

# 4. Next morning — check Test & Train AI tab in dashboard, approve any new models
#    Then restart the bots:
#    Ctrl+C the multi_bot_runner, then:
python src/multi_bot_runner.py
```

---

## Full Reset / Restart Everything

```
# Stop everything with Ctrl+C in each terminal, then:

# 1. Dashboard
streamlit run monitoring/dashboard.py

# 2. Bots
python src/multi_bot_runner.py

# 3. AI learner (optional — starts automatically from Auto-Run tab in dashboard)
python src/continuous_learner.py
```
