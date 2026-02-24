# RP Profits AI Trading Bot

An automated implementation of the RP Profits liquidity scalping strategy for ES/NQ futures,
targeting prop firm accounts (Topstep) via Tradovate API.

## Strategy Overview

Three setups traded daily between 9:00–11:30am EST:

| Setup | Logic |
|---|---|
| Break & Retest | Price breaks 8am range midpoint by 5+pts, retests midpoint, continues |
| Rejection | Price taps untested high/low, fails 2+ closes beyond it, reverses |
| Bounce | Price taps untested low/high, reclaim candle fires, continuation |

Risk rules: 5–10pt stop, 15–25pt target, min 1:3 R:R, max 3 trades/day, $200 daily loss limit.

## Project Structure

```
trading_bot/
├── config/settings.yaml        All parameters — edit this first
├── src/
│   ├── session_engine.py       Timezone and trading day logic
│   ├── level_builder.py        8am range + untested highs/lows
│   ├── setup_detector.py       Three setup implementations
│   ├── feature_builder.py      ML feature engineering
│   ├── risk_manager.py         All risk rules + kill-switch
│   ├── execution_engine.py     Tradovate WebSocket order management
│   ├── ml_filter.py            Phase 2: XGBoost meta-label filter
│   └── bot_runner.py           Main live bot orchestrator
├── backtest/
│   ├── harness.py              Replay backtester with cost model
│   └── walk_forward.py         Walk-forward evaluation framework
├── monitoring/
│   ├── audit_log.py            SQLAlchemy trade/signal logging
│   └── dashboard.py            Streamlit live dashboard
├── data/
│   └── download_historical.py  Databento data downloader
└── tests/
    └── test_setup_detector.py  24 unit tests (all passing)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your Databento API key and Tradovate credentials
```

### 3. Edit config

Open `config/settings.yaml` and review:
- `instrument.symbol` — `ES` or `NQ`
- `execution.paper_mode` — keep `true` until backtesting is complete
- Risk parameters

### 4. Download historical data

```bash
# Requires DATABENTO_API_KEY in .env
python data/download_historical.py --start 2022-01-03 --end 2024-12-31 --symbols ES NQ
```

### 5. Run backtest

```bash
python backtest/harness.py --symbol ES --start 2022-01-03 --end 2024-06-30 --save
```

### 6. Walk-forward evaluation

```bash
python backtest/walk_forward.py --symbol ES --start 2022-01-03 --end 2024-12-31
```

**Gate: Only proceed to live if 2+ of 3 windows are profitable after costs.**

### 7. Run tests

```bash
python -m pytest tests/ -v
```

### 8. Paper trading

```bash
# Ensure execution.paper_mode = true in settings.yaml
python src/bot_runner.py
```

### 9. Launch dashboard

```bash
streamlit run monitoring/dashboard.py
```

Open http://localhost:8501

### 10. Go live (after 30 profitable paper trading days)

1. Set `execution.paper_mode: false` in `settings.yaml`
2. Confirm Topstep account is funded and automation is permitted
3. Start with 1 MES/MNQ contract only
4. Run: `python src/bot_runner.py`

## Phase 2: ML Filter

After the rules-first engine is proven profitable live:

```bash
# Label historical signals
python src/ml_filter.py label

# Train XGBoost filter
python src/ml_filter.py train

# Evaluate improvement
python src/ml_filter.py eval

# Enable in config if it improves P&L:
# ml.enabled: true
```

## Key Risk Rules

- **Never go live without completing the 4-phase roadmap**
- **Check your prop firm's automation policy before going live**
- **Topstep allows bots but will not help you troubleshoot them**
- **Daily loss limit (-$200) is a hard stop — do not override it**
- **Kill-switch fires on connection loss > 60s — bot flattens and halts**

## Monthly Costs (Live Phase)

| Item | Cost |
|---|---|
| Tradovate Monthly Plan | $99 |
| Tradovate API add-on | $29.99 |
| CME data (ES + NQ) | ~$12 |
| VPS (optional) | $20–$40 |
| **Total** | **~$160–$180/mo** |

## Disclaimer

This software is for educational purposes only. Trading futures involves substantial risk of loss.
Past performance (backtested or live) does not guarantee future results. You are solely responsible
for all trading decisions and any losses incurred.
# charged-up-profits-pro
