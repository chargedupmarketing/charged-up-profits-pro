# Windows VPS Setup Guide — RP Profits Trading Bot

Getting from a fresh Windows Server VPS to a fully running backtest and live bot.
Tested on Windows Server 2022 / Windows 10/11.

---

## Recommended VPS Specs

| Spec | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Disk | 40 GB SSD | 80 GB SSD |
| OS | Windows Server 2022 | Windows Server 2022 |
| Location | Any | US East (NY/NJ) — lower latency to CME |

Good providers: Vultr (New Jersey), AWS EC2 (us-east-1), Contabo, Hetzner (Ashburn VA).

---

## Step 1 — Connect to Your VPS

Use Remote Desktop (RDP) or PowerShell remoting:

```powershell
# From your local machine (PowerShell):
mstsc /v:YOUR_VPS_IP
```

Or open **Remote Desktop Connection** from the Start menu and enter your VPS IP.

---

## Step 2 — Install Python 3.12

Open **PowerShell as Administrator** on the VPS and run:

```powershell
# Install Python 3.12 via winget (built into Windows 10/11/Server 2022)
winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
```

After install completes, add Python to your PATH (run once, then restart PowerShell):

```powershell
$p = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312"
[System.Environment]::SetEnvironmentVariable(
    "PATH",
    "$p;$p\Scripts;" + [System.Environment]::GetEnvironmentVariable("PATH","User"),
    "User"
)
```

Close and reopen PowerShell, then verify:

```powershell
python --version
# Expected: Python 3.12.x

pip --version
# Expected: pip 25.x from ...Python312...
```

---

## Step 3 — Transfer Project Files

### Option A — SCP from your local machine (run this in local PowerShell)

```powershell
# Requires OpenSSH. If you get an error, install it:
# Settings -> Apps -> Optional Features -> Add: OpenSSH Client

scp -r "C:\Apps\AI\trading_bot" your_user@YOUR_VPS_IP:C:\trading_bot
```

### Option B — Zip, upload, and extract on VPS

```powershell
# On your LOCAL machine — create zip:
Compress-Archive -Path "C:\Apps\AI\trading_bot" -DestinationPath "C:\Apps\AI\trading_bot.zip"

# Upload the zip to your VPS via RDP (copy/paste to shared clipboard or use a file transfer tool)

# On the VPS — extract:
Expand-Archive -Path "C:\trading_bot.zip" -DestinationPath "C:\"
```

### Option C — GitHub (recommended for ongoing updates)

```powershell
# Install Git on the VPS first:
winget install Git.Git --silent --accept-package-agreements --accept-source-agreements

# Then clone your repo:
git clone https://github.com/YOUR_USERNAME/trading_bot.git C:\trading_bot
```

---

## Step 4 — Create Virtual Environment and Install Dependencies

Open PowerShell and run:

```powershell
# Navigate to project
cd C:\trading_bot

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 4a — Install dependencies (Windows workaround for project-x-py)

The TopstepX SDK (`project-x-py`) lists `uvloop` and `msgpack-python` as dependencies;
both have Windows build issues.  Install in this order to work around them:

```powershell
# 1. Install msgpack (the renamed replacement for msgpack-python)
pip install "msgpack>=1.0.0"

# 2. Install project-x-py without its broken optional deps
pip install project-x-py --no-deps

# 3. Install the remaining project-x-py dependencies manually
pip install "httpx[http2]>=0.27.0" "orjson>=3.11.1" "polars>=1.31.0" `
            "pydantic>=2.11.7" "rich>=14.0.0" "lz4>=4.0.0" `
            "deprecated>=1.2.18" "cachetools>=5.0.0" "signalrcore>=0.9.5"

# 4. Install all remaining bot dependencies
pip install -r requirements.txt
```

Verify the SDK installed correctly:
```powershell
python -c "from project_x_py import ProjectX, TradingSuite; print('TopstepX SDK OK')"
```

> You must activate the venv every time you open a new PowerShell window:
> ```powershell
> cd C:\trading_bot
> .\venv\Scripts\Activate.ps1
> ```

---

## Step 5 — Configure Environment Variables

```powershell
# Navigate to project folder
cd C:\trading_bot

# Copy the example file
Copy-Item dotenv.example .env

# Open in Notepad to edit
notepad .env
```

Fill in these values and save:

```
DATABENTO_API_KEY=your_databento_api_key_here

PROJECT_X_USERNAME=your_topstepx_email@example.com
PROJECT_X_API_KEY=your_api_key_from_topstepx_dashboard
PROJECT_X_ACCOUNT_NAME=          # e.g. "Combine 50K" — leave blank for first account
```

> Get your Databento API key: https://databento.com/portal/keys
>
> **TopstepX API key:**
> 1. Log in at https://dashboard.topstepx.com
> 2. Go to **Account → API Access**
> 3. Generate an API key and copy it to `PROJECT_X_API_KEY`
> 4. `PROJECT_X_ACCOUNT_NAME` — copy the exact account name shown on your dashboard
>    (e.g. "Combine 50K", "Funded Account", etc.).  Leave blank to use the first account.

### TopstepX Prerequisites

The bot connects to TopstepX entirely over HTTPS/WebSocket — **no desktop app
needs to be running**.  Just supply your credentials in `.env` and the bot will
authenticate on startup.

- Paper trading = your **Combine** or **Evaluation** account on TopstepX
- Live trading = your **Funded** account

The `PROJECT_X_ACCOUNT_NAME` variable selects which account is used at runtime.
Set `execution.paper_mode: true` in `config\settings.yaml` to use MES micro contracts;
set to `false` to trade ES full-size contracts on your funded account.

---

## Step 6 — Download Historical Data

This downloads 3 years of 1-minute ES and NQ futures bars from Databento.
**Allow 10-20 minutes** — the files are ~500MB each.

```powershell
cd C:\trading_bot
.\venv\Scripts\Activate.ps1

python data\download_historical.py --start 2022-01-03 --end 2024-12-31 --symbols ES NQ
```

Expected output when complete:

```
INFO | Saved XXXXXX rows to data\historical\ES_ohlcv-1m.parquet
INFO | Saved XXXXXX rows to data\historical\ES_ohlcv-15m.parquet
INFO | Saved XXXXXX rows to data\historical\NQ_ohlcv-1m.parquet
INFO | Saved XXXXXX rows to data\historical\NQ_ohlcv-15m.parquet
INFO | Download complete.
```

---

## Step 7 — Run Tests

Always verify nothing broke in transfer before backtesting:

```powershell
cd C:\trading_bot
.\venv\Scripts\Activate.ps1

python -m pytest tests\ -v
```

Expected: **24 passed**

---

## Step 8 — Run Backtest

```powershell
cd C:\trading_bot
.\venv\Scripts\Activate.ps1

python backtest\harness.py --symbol ES --start 2022-01-03 --end 2024-12-31 --save
```

This takes ~25-30 minutes. To keep it running if your RDP session disconnects,
use Windows Task Scheduler or run it in a detached PowerShell window:

```powershell
# Run in a new detached PowerShell process (survives RDP disconnect)
Start-Process powershell -ArgumentList `
    "-NoExit -Command `"cd C:\trading_bot; .\venv\Scripts\Activate.ps1; python backtest\harness.py --symbol ES --start 2022-01-03 --end 2024-12-31 --save`""
```

Results print at the end and trade log saves to:
```
data\backtest_ES_2022-01-03_2024-12-31.csv
```

**Gate to proceed:** `profit_factor > 1.3` AND `max_drawdown < 15%`

---

## Step 9 — Walk-Forward Evaluation

```powershell
python backtest\walk_forward.py --symbol ES --start 2022-01-03 --end 2024-12-31
```

**Gate to proceed:** At least 2 of 3 test windows are profitable after costs.

---

## Step 10 — Launch Live Dashboard

```powershell
# Run dashboard in a separate PowerShell window
Start-Process powershell -ArgumentList `
    "-NoExit -Command `"cd C:\trading_bot; .\venv\Scripts\Activate.ps1; streamlit run monitoring\dashboard.py --server.port 8501 --server.address 0.0.0.0`""
```

Open in browser: `http://YOUR_VPS_IP:8501`

If Windows Firewall blocks the port, open it:

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "Trading Bot Dashboard" -Direction Inbound `
    -Protocol TCP -LocalPort 8501 -Action Allow
```

---

## Step 11 — Run the Live Bot (Paper Mode)

Paper mode is ON by default (`execution.paper_mode: true` in `config\settings.yaml`).
The bot runs 7:45am–11:30am EST and shuts down automatically after each session.

> **No desktop app required** — TopstepX connects over HTTPS/WebSocket.
> Just make sure your `.env` credentials are correct before starting.

```powershell
cd C:\trading_bot
.\venv\Scripts\Activate.ps1
python src\bot_runner.py
```

You should see log lines like:
```
INFO | Connecting to TopstepX (paper_mode=True, symbol=MES)
INFO | Connected to TopstepX | account='Combine 50K' | contract_id='CON.F.US.MES.H25'
INFO | Seeded bar store with 960 historical 1m bars
INFO | ExecutionEngine ready
```

If you see `ProjectXAuthenticationError`, verify that:
- `PROJECT_X_USERNAME` and `PROJECT_X_API_KEY` are correct in `.env`
- Your TopstepX account is active (not expired or in violation)
- The API key has not been revoked on the TopstepX dashboard

To run it automatically every trading day, set up a Windows Scheduled Task:

```powershell
# Create a scheduled task that runs the bot daily at 7:40am EST
# Run as Administrator
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -Command `"cd C:\trading_bot; .\venv\Scripts\Activate.ps1; python src\bot_runner.py >> C:\trading_bot\data\bot_log.txt 2>&1`""

$trigger = New-ScheduledTaskTrigger -Daily -At "7:40AM"

$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 5)

Register-ScheduledTask `
    -TaskName "RPProfitsBot" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Highest `
    -Force
```

To view/manage the scheduled task:

```powershell
Get-ScheduledTask -TaskName "RPProfitsBot"   # Check status
Start-ScheduledTask -TaskName "RPProfitsBot" # Run manually
Stop-ScheduledTask -TaskName "RPProfitsBot"  # Stop it
Unregister-ScheduledTask -TaskName "RPProfitsBot" -Confirm:$false  # Remove it
```

---

## Step 12 — Go Live (After 30 Profitable Paper Days)

1. Log in to your TopstepX dashboard and confirm your **Funded** account is active.
2. Open `.env` and update:
   ```
   PROJECT_X_ACCOUNT_NAME=Funded Account   # exact name shown on your dashboard
   ```
3. Open `config\settings.yaml`:
   - Set `execution.paper_mode: false`
   - Confirm `instrument.symbol: "ES"` (full-size) or keep `"MES"` to start small
4. Run: `python src\bot_runner.py`

> **Start with 1 MES contract** for the first few live sessions regardless of paper results.
> Once consistently profitable, increase to 1 ES.
>
> TopstepX funded accounts have daily loss limits — the bot's built-in kill-switch
> (`risk.max_daily_loss_dollars`) should be set *more conservatively* than your
> TopstepX account's daily loss limit to avoid a rules violation.

---

## Quick Reference — All Commands

```powershell
# ---- Setup (run once) ----

# Install Python
winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements

# Fix PATH (run once, then restart PowerShell)
$p = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312"
[System.Environment]::SetEnvironmentVariable("PATH","$p;$p\Scripts;" + [System.Environment]::GetEnvironmentVariable("PATH","User"),"User")

# Create and activate venv
cd C:\trading_bot
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (TopstepX Windows workaround)
pip install "msgpack>=1.0.0"
pip install project-x-py --no-deps
pip install "httpx[http2]>=0.27.0" "orjson>=3.11.1" "polars>=1.31.0" `
            "pydantic>=2.11.7" "rich>=14.0.0" "lz4>=4.0.0" `
            "deprecated>=1.2.18" "cachetools>=5.0.0" "signalrcore>=0.9.5"
pip install -r requirements.txt

# Create .env from template (fill in TopstepX credentials + Databento key)
Copy-Item dotenv.example .env
notepad .env


# ---- Daily usage ----

# Activate venv (required each new PowerShell window)
cd C:\trading_bot; .\venv\Scripts\Activate.ps1

# Download data
python data\download_historical.py --start 2022-01-03 --end 2024-12-31 --symbols ES NQ

# Run tests
python -m pytest tests\ -v

# Run backtest
python backtest\harness.py --symbol ES --start 2022-01-03 --end 2024-12-31 --save

# Run walk-forward
python backtest\walk_forward.py --symbol ES --start 2022-01-03 --end 2024-12-31

# Train ML filter (Phase 2 only)
python src\ml_filter.py train

# Evaluate ML filter
python src\ml_filter.py eval

# Start dashboard
streamlit run monitoring\dashboard.py --server.port 8501 --server.address 0.0.0.0

# Run live bot
python src\bot_runner.py
```

---

## File Structure After Full Setup

```
C:\trading_bot\
├── .env                            # Your credentials (never share or commit)
├── config\settings.yaml            # All bot parameters — edit this to tune
├── data\
│   ├── historical\
│   │   ├── ES_ohlcv-1m.parquet
│   │   ├── ES_ohlcv-15m.parquet
│   │   ├── NQ_ohlcv-1m.parquet
│   │   └── NQ_ohlcv-15m.parquet
│   ├── audit.db                    # Created automatically on first bot run
│   └── backtest_ES_*.csv           # Created after backtest
├── src\                            # Strategy engine
├── backtest\                       # Backtesting framework
├── monitoring\                     # Dashboard and audit logging
├── tests\                          # Unit tests (24 tests)
└── venv\                           # Python virtual environment
```

---

## Troubleshooting

**`pip` not recognized:**
```powershell
# Use python -m pip instead:
python -m pip install -r requirements.txt
```

**`python` opens Microsoft Store instead of running:**
```powershell
# Fix PATH (then restart PowerShell):
$p = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312"
[System.Environment]::SetEnvironmentVariable("PATH","$p;$p\Scripts;" + [System.Environment]::GetEnvironmentVariable("PATH","User"),"User")

# Or: Settings -> Apps -> Advanced app settings -> App execution aliases
# Turn OFF both python.exe and python3.exe aliases
```

**`Activate.ps1` blocked by execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**`ProjectXAuthenticationError` on bot start:**
```powershell
# Verify credentials are loaded:
cd C:\trading_bot; .\venv\Scripts\Activate.ps1
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
print('Username:', os.getenv('PROJECT_X_USERNAME', '(not set)'))
print('API key length:', len(os.getenv('PROJECT_X_API_KEY', '')))
"
# If either is missing, edit .env and try again
# Also verify on the TopstepX dashboard that your API key is still active
```

**`ProjectXInstrumentError: instrument not found for ES`:**
- Confirm your TopstepX account has futures trading permissions
- Try `instrument.topstepx_symbol: "MES"` in `config\settings.yaml` to trade micro contracts
- Check the front-month contract is still active (update symbol near quarterly expiry)

**`ProjectXOrderError` when placing bracket:**
- TopstepX enforces price alignment to tick size (0.25 pts for ES/MES)
- The SDK auto-aligns prices; if you see this, check that stop and target prices
  are on the correct side of entry (stop below entry for long, above for short)

**`uvloop does not support Windows` during pip install:**
- Do NOT run `pip install project-x-py` directly on Windows
- Follow the workaround in **Step 4a** above (install with `--no-deps` then manually)

**Databento download returns 0 rows:**
- Confirm `DATABENTO_API_KEY` is correct in `.env`
- The symbol format must be `ES.c.0` with `stype_in="continuous"` — already set in the downloader
- Check your Databento account has sufficient credits: https://databento.com/portal

**Dashboard not accessible from browser:**
```powershell
# Run as Administrator — open firewall port
New-NetFirewallRule -DisplayName "Trading Bot Dashboard" -Direction Inbound `
    -Protocol TCP -LocalPort 8501 -Action Allow

# Also check your VPS provider's firewall/security group settings
# (AWS Security Groups, Azure NSG, etc.) and allow TCP 8501 inbound
```

**Bot stops unexpectedly:**
```powershell
# Check the audit database event log
python -c "
from monitoring.audit_log import AuditLogger
a = AuditLogger()
events = a.get_recent_signals(limit=20)
[print(e) for e in events]
"
# Or open data\audit.db in DB Browser for SQLite (free download)
```

**UnicodeEncodeError when printing results:**
```powershell
# Set UTF-8 output for the session
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```
