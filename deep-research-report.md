iturn25image0turn29image0turn29image2turn29image4
# RP Profits and an AI-Driven Trading Bot Blueprint

## Executive summary

RP Profits is a trading content creator whose public content heavily emphasizes “simple,” repeatable scalping—often framed as a liquidity/levels approach—and he markets a broader education/community product (Profit Insider) alongside claims of large trading profits in video titles and promotional materials. citeturn5search1turn3search13turn14view0

Reconstructing his “replicable” method from publicly available materials, the clearest, most codifiable core is a three-setup structure around (a) an “8 a.m. candle / opening range” level (including a midpoint), plus (b) “untested” highs/lows as liquidity targets, with (c) entries that require a specific reaction (break/retest, rejection, or bounce) and tight stops with larger fixed targets (often described as ~5–10 points risk for ~15–25 points reward). Because some of the most important parts are described in discretionary terms (e.g., “strength/weakness,” “trend”), a faithful automation attempt must (1) define those discretionary clauses precisely, and (2) treat the first automated version as “rules-first,” with machine learning used only as a **filter** (meta-label) rather than as the primary signal generator. fileciteturn0file0

On the engineering side, this is not high-frequency trading in the strict sense; it can be implemented with 1-minute and 15-minute bars plus robust order management (brackets/OCO), but you still need professional-grade discipline: verified historical data, realistic slippage/commission modeling, walk-forward testing, and careful monitoring to reduce “backtest overfitting” risk. citeturn23search0turn23search8turn21search0turn21search19

A practical “move forward” path is: build the deterministic strategy engine first; prove it in a clean backtest with conservative assumptions; paper trade; then go live with micro size and strict kill-switches. Only after that should you add ML components to reduce false positives or adapt thresholds by regime. citeturn21search0turn30search35turn20search31

## Public footprint, biography signals, and business ecosystem

The most accessible, primary-footprint indicators show RP Profits operating primarily on **entity["company","YouTube","video platform"]** with a channel listing roughly **34.6K subscribers and ~162 videos** (as surfaced in platform snippets), and he positions the channel around trading education plus lifestyle content. citeturn5search1

On **entity["company","Instagram","social platform"]**, the handle commonly associated with the brand is presented as “Reece Phillips” (public profile naming) and shows a large follower count (hundreds of thousands) in search snippets; this is consistent with a “trading influencer” growth pattern where Instagram is the main acquisition channel and YouTube is long-form education/authority. citeturn6search2  
On **entity["company","X","social platform rebrand"]**, direct viewing is constrained, but third-party mirrors/snippets commonly show a smaller account footprint (low-thousands followers), which is typical of creators who prioritize Instagram + YouTube over text-first platforms. citeturn6search9

The central monetization layer appears to be **entity["organization","Profit Insider","trading community brand"]**, marketed as a community with analysis/signals/education and routed through **entity["company","Whop","digital product marketplace"]** and **entity["company","Discord","chat platform"]**. The public pricing shown is **$99/month** for the premium tier (plus a free access tier). citeturn14view0turn16view0turn16view1

Profit Insider’s public landing pages also make strong performance-adjacent claims (e.g., “over $3,000,000+ in member profits,” high ratings, and large “trusted by”/community-size language). Treat these as marketing claims unless independently verified. citeturn14view0  
Their disclaimer explicitly states the content is for educational purposes, that members are responsible for decisions, and that trading involves significant risk and losses; this aligns with, but does not validate, any performance claims. citeturn15view0

Public reviews show a mixed sentiment distribution: high average ratings alongside visible negative reviews/complaints about refunds, deleted reviews, or mismatch between expectations and delivery. These are not proof either way, but they are relevant reputational risk indicators if you plan to “replicate the approach” commercially. citeturn15view2turn5search4turn19search4

### Content themes and claimed performance

RP Profits’ public video titling/positioning repeatedly emphasizes:
- “one candle” / opening-range style scalping, sometimes with “proven” or “backtested” framing citeturn3search4turn3search13turn3search20turn13search21  
- large profit numbers in titles (e.g., weekly, monthly, yearly claims) which should be interpreted as marketing claims unless corroborated by broker statements or audited track records (not found in the accessible primary sources here). citeturn3search13turn4search14turn13search8  
- a “liquidity levels” framing (often minimizing indicator reliance). citeturn13search8turn13search6turn19search5  

In addition, the script you provided explicitly discusses building/using algorithms and claims multi-year backtest coverage and thousands of trades for at least one algorithm variant—again, as claims within content, not independently verified. fileciteturn0file0

## Trading strategy reconstruction from public materials

This section reconstructs what appears to be RP Profits’ “core method” from the transcript you supplied and cross-checks the surrounding public framing via his video catalog snippets. Where details are not explicitly stated, they are marked **unspecified**.

### Instruments, sessions, and timeframes

**Instruments (publicly indicated):**
- The transcript repeatedly references trading “the S&P 500” and shows futures-style point targets/stops consistent with ES-style scalping; other video titles explicitly reference trading $ES futures. fileciteturn0file0 citeturn6search18turn4search17  
- Profit Insider marketing mentions “Stock/Options” and “Futures/Crypto” signals, implying multi-asset scope at the community level, even if the “one candle” method is futures-centric. citeturn14view0  

**Session/time anchors (publicly indicated):**
- The transcript’s method centers on an “8 a.m. candle” and then focuses on the main US session as “the important time to trade,” with additional time windows referenced (e.g., waiting until 9:30). fileciteturn0file0  
- Exact timezone is **not explicitly specified** in the accessible snippets; however, the context implies US market hours (commonly Eastern Time for US index futures discussion). Treat timezone handling as an implementation assumption you must confirm. fileciteturn0file0

**Timeframes (publicly indicated):**
- One key candle/range is referenced (the “8 a.m. candle”) and the script discusses going to lower timeframes for execution/confirmation; this matches a top-down: 15m anchor + 1m/5m execution pattern. fileciteturn0file0  
- Additional strategy variants are mentioned (including 1-minute scalping and a higher-timeframe bot), but the precise rules for those alternate bots are **unspecified** publicly. fileciteturn0file0

### Core concepts to encode

The transcript describes a “boring, repetitive” structure: mark levels in the morning, then trade the same setups repeatedly. fileciteturn0file0  
The codable primitives are:

**A. Morning anchor range (“8 a.m. candle”)**
- Mark the candle’s high/low and a midpoint; the midpoint later acts as a key decision level for at least one setup. fileciteturn0file0  

**B. Untested highs/lows (“liquidity levels”)**
- Mark session highs/lows that have “not yet been tested,” with the explicit rule-of-thumb that untested levels are “more powerful” than levels already touched. fileciteturn0file0  

**C. Reaction-based entries**
- The transcript says the strategy has *three setups*: break-and-retest, rejection, and bounce. fileciteturn0file0

### Setups, entry/exit rules, and what is unspecified

Below is the most faithful “engineering translation” of the publicly described rules. Any rule not explicitly described is **unspecified** and should not be invented if your goal is strict replication.

| Setup (public name) | Public trigger logic (translated) | Entry condition (codable) | Stop / target (public ranges) | Discretionary elements |
|---|---|---|---|---|
| Break & retest | Price breaks a key level (commonly the 8 a.m. midpoint) and then retests it before continuing | “Break” must be defined (e.g., close beyond, or N points beyond); “retest” must be defined (touch ± tolerance) fileciteturn0file0 | Tight stop and larger target are described (risk:reward framing ~1:3). fileciteturn0file0 | What constitutes a valid “break” vs noise; trend filter specifics **unspecified** fileciteturn0file0 |
| Rejection (off high/low) | Price taps a key high/low and fails to accept beyond it (“failing to close above”) | One or more closes that fail beyond level; then enter in opposite direction fileciteturn0file0 | Script references point targets like ~24 points and R:R framing fileciteturn0file0 | How many candles must “fail”; volume/impulse criteria **partly specified** (volume mentioned) fileciteturn0file0 |
| Bounce (off low/high) | The mirror image of rejection: price hits a key level and “bounces” | Confirmation via a reclaim/close back away from level; then enter with invalidation behind swing fileciteturn0file0 | Tight stop behind invalidation; larger target (exact numbers vary in transcript) fileciteturn0file0 | Exact “strength” confirmation is discretionary unless defined (wick patterns, close location, etc.) fileciteturn0file0 |

**Risk management (publicly stated at a high level):**
- The method emphasizes tight stops, larger targets, and explicit risk-to-reward framing (e.g., “1 to three”). fileciteturn0file0  
- The strategy’s rule-based framing is highlighted as “no subjectivity,” but the transcript still contains discretionary language (strength/weakness, trend, volume “dying down”), so automation must turn that language into deterministic checks or accept that the bot will diverge from the human method. fileciteturn0file0

**Filters / “don’ts” (publicly stated):**
- The transcript includes the idea that the “8 a.m.” range should be below a threshold (referenced as “under 20 points”). fileciteturn0file0  
- It also warns against trading “tested” levels and against trading against the trend. Exact trend definition is **unspecified**. fileciteturn0file0

### Notable automation-related claims in the public script

The transcript claims:
- Institutional traders use algorithms, and competing requires automation. fileciteturn0file0  
- At least one algorithm is presented with “results” over multiple years and “2,700 trades” (explicitly stated). fileciteturn0file0  

These claims help you infer the *intended automation style* (rules-first, consistent execution), but they do not provide the proprietary implementation details needed to literally clone a commercial bot. fileciteturn0file0

## Automation requirements for a bot that mirrors this approach

This section turns the reconstructed strategy into concrete engineering requirements: data, connectivity, broker/execution controls, latency, and operational reliability.

### Market data requirements

**Minimum viable data (to match the transcript rules):**
- 1-minute OHLCV bars (for “close above/below,” volume checks, and entry confirmation) plus 15-minute bars (for the “8 a.m. candle” range/midpoint). fileciteturn0file0  
- A trading calendar/timezone model appropriate for index futures so the “8 a.m.” candle is unambiguous. (Timezone is not explicitly stated in sources; confirm via direct video transcript or on-chart examples before coding.) fileciteturn0file0

**Recommended data (for realism and future ML filtering):**
- Tick-level trades (or 1-second bars) to model intrabar touches of levels and to build better slippage simulation; CME futures data feeds can provide event-level depth-of-book, but you may not need full depth if you are not doing order-flow logic. citeturn23search0turn23search8turn23search25

**Primary data vendor examples (futures):**
- **entity["company","Databento","market data vendor"]** offers CME Globex MDP 3.0 (“GLBX.MDP3”) with historical coverage and high-resolution timestamps, including full depth-of-book. citeturn23search0turn23search8turn23search29  
- CME infrastructure reality (Aurora matching engine location) matters if you’re optimizing latency; third-party explainers and CME’s own Aurora-related announcements support this. citeturn20search6turn20search22

### Brokerage and execution interfaces

To automate safely, you need a broker/API path that supports:
- placing bracket orders (entry + stop + take-profit), or at least OCO behavior
- reliable real-time market data subscription (or vendor data + broker for execution)
- robust order state tracking (fills, partial fills, rejects)

Common automation-capable routes include:

- **entity["company","Interactive Brokers","brokerage"]** APIs (TWS API / Client Portal API) support many order types, including bracket orders and OCA group logic. citeturn20search3turn20search31turn20search7turn20search19  
- **entity["company","Tradovate","futures brokerage platform"]** exposes API access and supports real-time data via WebSockets; their platform materials highlight access to tick/chart/DOM data and complex order capabilities. citeturn23search3turn23search11turn23search7turn23search28  
- **entity["company","Rithmic","dma execution platform"]** provides programmatic trading interfaces with market data and order routing, including brackets and OCOs (as described in their API overview). citeturn23search2turn23search10

### Latency and VPS/colocation considerations

For an “8 a.m. candle + 1m confirmation” strategy, you are not competing in microsecond latency arbitration. However, latency still impacts fill quality and slippage on fast moves, especially around the US open. citeturn20search6turn20search18

Practical guidance:
- If trading CME index futures, placing your server geographically close to Aurora can reduce network latency relative to a random region VPS. citeturn20search6turn20search18turn20search22  
- Do not build a fragile system: exchange/data-center incidents happen. Recent CME outages tied to data-center cooling issues illustrate operational risk; your bot should have a “safe mode,” timeouts, and kill-switches. citeturn20news43turn20news44turn20search26

### Prop-firm and platform rule constraints

If your goal is to “scale prop firms” with automation, the constraint is often not engineering—it’s **rules**.

Examples from primary rule pages:
- Some firms explicitly prohibit automation/AI/bots (Apex’s own support documentation states automation is prohibited). citeturn30search5  
- Others allow automated strategies with disclaimers (Topstep states automated strategies can be used, but it won’t help you set them up or troubleshoot). citeturn30search7  
- Many firms also prohibit trade copying or resource sharing patterns (device/IP/account-sharing style enforcement). citeturn30search1turn30search4  

This means a “replica bot” may be technically feasible but operationally unusable on specific prop programs unless you choose a program that explicitly permits automation and you remain compliant. citeturn30search5turn30search7

## System architecture for an AI-assisted replica bot

A “replica” approach should be treated as **two layers**:

- **Deterministic strategy engine** (replicates publicly described rules)
- **ML filter** (optional; reduces false positives, sizes risk, or adapts thresholds)

This minimizes the risk of building an opaque ML system that “fits the past” but fails live. The finance literature strongly warns about backtest overfitting and false discoveries when you search/optimize too aggressively. citeturn21search0turn21search19turn21search5

### Proposed module breakdown

**Signal engine (rules-first)**
- Build the day’s levels:
  - `range_8am_high`, `range_8am_low`, `range_mid`
  - “untested highs/lows” definition (you must define the lookback window: prior day session? overnight? multiple sessions—publicly not fully specified) fileciteturn0file0
- Detect setups:
  - break+retest pattern around the midpoint (or other key levels)
  - rejection pattern: touch level → fail to close beyond → reverse
  - bounce pattern: touch level → reclaim/confirm → continuation fileciteturn0file0

**Risk manager**
- Hard daily loss limit
- Max trades per day
- Per-trade risk sizing (contracts) based on stop distance and risk budget
- “No trade” filters (e.g., 8 a.m. range too large; low liquidity windows; volatility regime) fileciteturn0file0

**Execution engine**
- Places orders as bracket/OCO where supported (entry + stop + TP)
- Reconciles order states; handles rejects; retries safely; cancels all if feed disconnects citeturn20search31turn23search2turn23search28

**Logging + analytics**
- Store every decision: levels, detected setup, features, order actions, fills, slippage
- Produce daily “blame-free” diagnostics (“missed because filter X,” “stopped because volatility spike”) citeturn30search35

### Machine learning component ideas

Use ML only after you have a baseline rules engine.

**What ML should do here (good fit):**
- Predict *probability of success* for a detected setup, so you only take the best 20–40% of signals (“meta-labeling”). citeturn21search21turn21search27  
- Predict expected slippage regime or “chop risk” to reduce trades in unstable conditions (often the real killer of simple scalpers). citeturn21search0turn21search5  

**Labeling approach**
- A practical label is: given an entry point (on a detected setup), did price hit TP before SL within a time limit?  
- This matches “triple barrier” style labeling (profit-take barrier, stop barrier, and time barrier), which is widely used in financial ML to create realistic classification labels. citeturn21search21turn21search10

**Candidate models**
- Gradient boosting (XGBoost/LightGBM style) for tabular features is often a strong first choice; keep the model simple and interpretable before trying deep learning. (Model choice guidance here is engineering inference; the *overfitting risk* and the need for careful evaluation is the key point.) citeturn21search0turn21search5  
- If you later move to sequence models (CNN/LSTM/Transformers), only do so after you have enough labeled samples and a strong leakage-proof evaluation harness. citeturn21search0turn21search19

### Feature engineering ideas tailored to RP Profits’ described logic

All of these features are directly tied to the strategy’s public language (“range,” “midpoint,” “untested levels,” “volume,” “trend”):

- 8 a.m. range size (points), and normalized range size (range / ATR) fileciteturn0file0  
- Distance from current price to midpoint, high, low (in points and in ATR units) fileciteturn0file0  
- “Acceptance” measures near level: number of closes beyond level in last N minutes; wick ratio; time spent within tolerance band fileciteturn0file0  
- Volume at the level vs baseline (rolling median), since volume is referenced as an input fileciteturn0file0  
- Trend proxy (must be defined): e.g., higher timeframe MA slope, or prior session VWAP slope (trend definition itself is **unspecified** publicly; pick one definition and treat it as your assumption) fileciteturn0file0  

### Architecture diagram (rules + ML filter)

```mermaid
flowchart TB
  A[Market Data Ingest<br/>1m + 15m bars (and optional ticks)] --> B[Session Engine<br/>timezones, trading day boundaries]
  B --> C[Level Builder<br/>8am range high/low/mid<br/>untested highs/lows]
  C --> D[Setup Detector<br/>break+retest / rejection / bounce]
  D --> E[Feature Builder<br/>range stats, proximity, volume, trend proxy]
  E --> F{ML Filter?<br/>probability-of-success}
  F -->|No| G[Risk Manager<br/>position sizing, max loss, kill-switch]
  F -->|Yes| G
  G --> H[Execution Engine<br/>brackets/OCO, retries, reconcile fills]
  H --> I[Broker API]
  H --> J[Audit Log + Metrics Store]
  J --> K[Backtest/Sim Harness<br/>slippage, fees, walk-forward]
```

## Development stacks and reusable tools

Below is a practical comparison table for building a system like this. Costs are ballpark and exclude trading losses; they emphasize *engineering + data + hosting*. “Suitability” is for a strategy anchored on futures session levels (like the described “one candle” method). citeturn21search2turn22search3turn22search4turn23search2turn23search29

| Stack | Approx cost profile | Pros | Cons | Suitability for “RP Profits style” |
|---|---:|---|---|---|
| Python + Backtrader + IBKR API | Low–medium | Mature backtesting/trading framework; known broker integrations; fast to prototype citeturn22search4turn20search3 | Backtesting realism depends on your modeling; not built specifically for futures microstructure | Good for MVP; solid if you stay 1m/15m |
| Python + NautilusTrader + (Rithmic/IBKR) | Medium | Event-driven, high-performance, designed for backtest/live parity citeturn22search3turn22search7 | Heavier learning curve; integrations vary; more engineering upfront | Very good if you want production-quality logging and scale |
| C# / Python + Lean engine (self-host) | Medium | Institutional-style event engine; supports research→live workflow; open-source core citeturn21search2turn21search4 | Integration work; futures data plumbing can be nontrivial | Good if you want a full “platform” feel |
| Python + direct Rithmic protocol | Medium–high | Full control; can get low-latency and advanced order handling; Rithmic supports brackets/OCO citeturn23search2turn23search13turn23search35 | More engineering risk; testing harness must be strong | Strong for futures execution if you can engineer safely |
| Python + Freqtrade (crypto variant) | Low–medium | Batteries-included bot; backtesting + ML tooling built in (for crypto) citeturn22search2turn22search10 | Crypto-focused; doesn’t map cleanly to CME futures session logic | Only if you adapt the idea to crypto, not a strict replica |

### Open-source references worth reusing

If you want to reduce build time, the best “reuse” components are usually:
- a battle-tested backtest/live engine (Lean, NautilusTrader, Backtrader) citeturn21search2turn22search3turn22search4  
- robust data access + normalization tools (Databento clients, continuous futures handling) citeturn23search4turn23search18  
- proven cautionary frameworks for avoiding overfitting and selection bias in research workflows citeturn21search0turn21search19turn21search5  

## Data labeling, backtesting, evaluation, and realism controls

### Dataset sizing and labeling practicality

A single “setup detector” on ES can generate many candidate events per day (especially if “untested highs/lows” are defined broadly). Your labeling pipeline will likely produce:
- **100s–1,000s** of labeled events per year (depending on strictness)  
- enough for a first-pass meta-labeling model after 6–18 months of data, but still very easy to overfit if you tune too aggressively citeturn21search0turn21search5

A robust approach is:
- Start with 3–5 years of 1-minute data (the transcript itself references multi-year history as meaningful) fileciteturn0file0  
- Create labels using triple-barrier framing (TP, SL, time limit), because it matches how the strategy is described (tight stop, larger TP, trade must “work” quickly). citeturn21search21turn21search10

### Backtesting methodology you should treat as non-negotiable

Finance backtests are extremely prone to false discoveries if you iterate too freely. The literature documents how easily “best backtests” fail out-of-sample due to multiple testing and overfitting. citeturn21search0turn21search5turn21search19

Minimum standard:
- Walk-forward evaluation (train on earlier period, test on later period)  
- No leakage: ensure “future information” (like session high/low after the fact) does not enter features for decisions earlier in the day  
- Conservative slippage assumptions (especially during 9:30 open volatility)  
- Stability checks: performance by month, by volatility regime, by day-of-week citeturn21search0turn21search5

Recommended metrics:
- Max drawdown, average drawdown duration
- Profit factor, expectancy, win rate, avg win/avg loss
- Sharpe/Sortino (with caution) and “deflated Sharpe ratio” style thinking to avoid overstating significance citeturn21search19turn21search0

### Slippage, fees, and order modeling

You should model:
- commissions + exchange fees (broker-specific)
- “spread/slippage” as a function of volatility and time-of-day
- partial fills if using limits at fast levels  

Even if you execute with market orders, you still need to estimate adverse selection. This is one of the main reasons simple scalpers look great in naive backtests and fail live. citeturn21search0turn21search5

## Compliance, ethics, cost estimate, and a step-by-step roadmap

### Compliance and ethical constraints

Key points if you ever share results publicly or sell access:
- Futures promotional standards can require prominent “past performance not indicative” statements and impose conditions on performance presentation. citeturn20search4turn20search0turn20search8  
- If you fall under investment adviser marketing rules (context-dependent), hypothetical/backtested performance is heavily regulated and conditioned. citeturn20search1turn20search5  
- Even for personal systems, you should keep the same discipline as regulated entities: audit logs, risk disclosures, and honest reporting. citeturn15view0turn20search4

Prop-firm ethics/rules:
- Some firms explicitly prohibit automation/AI; violating rules can lead to account closure and forfeiture. citeturn30search5turn30search1  
- “Scale with bots across many accounts” can cross into prohibited behavior depending on the firm’s definitions and monitoring. Always treat the firm’s own rule pages as source of truth. citeturn30search1turn30search4turn30search7

### Estimated costs and timeline

Two cost envelopes are useful: (A) an MVP for personal use; (B) a production-grade system.

**Assumptions (explicit):**
- You already have a VPS (you previously mentioned paying ~$30/mo; not re-verified here).  
- You trade futures and need data + an execution API.  
- You want a rules-first engine plus optional ML filtering, not a full HFT stack. citeturn23search29turn20search3turn23search2

#### MVP cost outline (prototype to paper trading)

| Item | Monthly | One-time | Notes / drivers |
|---|---:|---:|---|
| VPS | $0–$100 | $0 | $0 if existing; higher if specialized low-latency hosting citeturn20search18turn20search6 |
| Historical market data | $0–$300+ | $50–$500+ | Depends on vendor, depth, and how much you download; Databento advertises “historical from $0.50/GB” for some datasets citeturn23search29turn23search4 |
| Live data | $0–$200+ | $0 | Broker entitlements or vendor plan dependent citeturn23search25turn23search11 |
| Backtesting engine + infra | $0 | $0 | Open-source if you self-build (Lean, Backtrader, NautilusTrader) citeturn21search2turn22search4turn22search3 |
| Dev time (your time) | — | — | Biggest cost is engineering hours—especially testing + monitoring |

#### Production cost outline (live, monitored, resilient)

Add:
- redundant monitoring/alerting (PagerDuty-style)
- backup connectivity and automated shutdown logic
- stricter data pipeline (roll adjustments, session calendars)
- compliance-ready logging if you publish results

Operational reliability matters because exchange/data-center incidents and connectivity faults are real. citeturn20news43turn23search2turn20search26

### Step-by-step implementation roadmap with testing gates

Below is a pragmatic path that matches your constraint (“Cursor does the brunt work”) while keeping you from deploying an untested bot.

#### Roadmap milestones

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| Requirements lock | A written spec of “8 a.m. candle,” untested levels, and the three setups | No ambiguous terms remain without a definition (especially “trend,” “retest,” “fail to close”) fileciteturn0file0 |
| Data pipeline | Clean 1m + 15m bars with correct session boundaries | Spot-check 20 random days: levels match chart reality; no timezone drift citeturn23search0turn23search18 |
| Strategy engine v1 | Deterministic signals + bracket order plan | Unit tests for each setup; simulated fills; no lookahead leakage citeturn21search0turn20search31 |
| Backtest harness | Slippage + fees + walk-forward | Performance is stable across periods; not concentrated in one month citeturn21search0turn21search5 |
| Paper trading | Live feed + execution in sim | 2–4 weeks of stable ops; low error rate; logs complete citeturn30search35turn23search11 |
| Live micro | Smallest size, strict kill switch | 30 trading days without catastrophic failure; disciplined shutdown rules citeturn20news43turn15view0 |
| ML filter v1 | Meta-label model (optional) | Improves drawdown or hit rate out-of-sample, not just in-sample citeturn21search27turn21search0 |

#### 30-day minimal viable experiment

Goal: answer one question—**“Does the rules-first version have any edge after costs?”**—before you spend months on ML.

**Week 1**
- Implement level builder (8 a.m. range + midpoint, untested highs/lows) and strict definitions. fileciteturn0file0  
- Write replay/backtest harness on 6–12 months of data with conservative slippage. citeturn21search0turn23search0  

**Week 2**
- Implement only *one setup* (break+retest) + full risk manager + bracket orders. fileciteturn0file0  
- Validate: no trading when 8 a.m. range exceeds the stated threshold; no trading outside your defined time window. fileciteturn0file0  

**Week 3**
- Add setups 2 and 3 (rejection and bounce) only if backtest quality is stable. fileciteturn0file0  
- Paper trade in real time; alert on every signal; do not auto-execute yet.

**Week 4**
- Turn on auto-execution in simulation with full logging.
- Produce a post-mortem: where did it fail? (false signals, slippage, session confusion, “trend” filter issues)

If, after 30 days, the rules-first engine is unprofitable net of conservative costs, do **not** “fix it with ML.” Instead, tighten the rule definitions, reduce trading frequency, or reassess whether the public description omits critical discretionary context that cannot be inferred. citeturn21search0turn21search5

### Risks, failure modes, and mitigations

| Risk | What it looks like | Mitigation |
|---|---|---|
| Hidden discretion | Human trader avoids trades the bot takes | Encode explicit filters; add “no-trade states”; use ML meta-labeling only after baseline works citeturn21search27turn21search0 |
| Backtest overfitting | Great equity curve, then immediate live failure | Walk-forward testing, limit parameter search, track PBO/DSR thinking citeturn21search0turn21search19turn21search25 |
| Slippage sensitivity | Small edge disappears net of costs | Conservative slippage modeling; prefer limit/controlled entries if strategy allows citeturn21search0turn20search31 |
| Operational incidents | Feed disconnects; exchange outage; stuck orders | Kill switch, timeouts, reconciliation loops, alerting; design for CME-like outages citeturn20news43turn23search2turn20search26 |
| Prop firm noncompliance | Account closure, forfeiture | Choose firms that explicitly allow automation; follow their rules as primary source citeturn30search5turn30search7turn30search1 |

## Primary-source index

The user asked for links to all primary sources found. URLs are provided verbatim below.

```text
YouTube channel
https://www.youtube.com/@RP.Profits1

Key RP Profits videos referenced by title/snippet
https://www.youtube.com/watch?v=sXtmM_KQYiM
https://www.youtube.com/watch?v=jbtkrsOCXis
https://www.youtube.com/watch?v=zXasoSKXvLo
https://www.youtube.com/watch?v=sM56cLgQCxQ
https://www.youtube.com/watch?v=zI7235VZxGc

RP Profits X profile (primary, but may be difficult to view without login)
https://x.com/rp_profits?lang=en

RP Profits Instagram profile (primary, but may be difficult to view in some contexts)
https://www.instagram.com/rp.profits/?hl=en

Profit Insider website and disclaimer (owned brand pages)
https://profit-insiders.com/
https://profit-insiders.com/disclaimer

Whop pages for Profit Insider (product + reviews + checkout)
https://whop.com/profit-insider/reviews/
https://whop.com/checkout/plan_FZM56HjBplXox/?d2c=true

Linktree for Profit Insider
https://linktr.ee/ProfitInsider
```

