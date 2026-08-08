# PROJECT_GOAL.md

## Executive Summary
The goal of `day-trading-analysis` is to build an end-to-end, quantitative research and strategy testing framework that evaluates the statistical validity of Inner Circle Trader (ICT) day trading patterns. 

Rather than building a speculative live trading execution script, this project approaches mechanical trading rules through data science, machine learning, and quantitative portfolio management. It tests whether popular price-action patterns hold actual predictive edge across multiple asset classes and under varying market conditions.

---

## 1. Asset Coverage & Data Layer
The framework ingests high-frequency intraday and daily price data across 5 core global markets to ensure cross-asset validity:

* **Equity Indices:** S&P 500 (`^GSPC`), Nasdaq (`^IXIC`)
* **Commodities:** Gold (`GC=F`), Silver (`SI=F`)
* **Foreign Exchange:** EUR/USD (`EURUSD=X`)

### Market Noise & Regime Filtering
Before evaluating trading signals, price data is passed through market regime filters to separate trending price movement from random noise:
* **Average True Range (ATR):** Measures relative volatility regimes.
* **Kaufman Efficiency Ratio / Hurst Exponent:** Quantifies trend efficiency versus random walk/consolidation phases.

---

## 2. ICT Pattern Detection Engine
The system programmatically identifies **Bullish** and **Bearish** occurrences for four core ICT trading setups:

1. **Fair Value Gaps (FVG):** Imbalances defined across 3-candle sequences where outer wicks do not overlap.
2. **Order Blocks (OB):** The final opposing candle prior to an explosive directional price movement.
3. **Optimal Trade Entry (OTE):** Price retracements targeting the 62%–79% Fibonacci zone following a displacement impulse.
4. **Liquidity Sweeps ("Turtle Soup"):** Price spikes beyond key structural highs/lows that immediately reverse back inside the prior range.

Each detected setup automatically generates a standardized signal dictionary with defined **Entry**, **Stop Loss (SL)**, and **Take Profit (TP)** parameters.

---

## 3. Machine Learning Classification Layer
To determine if pattern success can be predicted beforehand, machine learning classifiers (e.g., Random Forest, XGBoost) are trained on historical pattern setups.

* **Target Variable:** Binary outcome (1 if trade hits TP first, 0 if trade hits SL first).
* **Features:** 
  * Structural metrics (gap width, displacement candle magnitude, distance to key levels).
  * Macro context (time of day/session, market volatility regime, ATR ratio).
  * Cross-pattern presence (confluence indicators).

---

## 4. Multi-Scenario Backtesting Engine
The backtesting engine evaluates execution performance across four distinct operational modes to isolate the impact of ICT rules, pattern confluence, and machine learning filtering:

| Scenario | Mode | Execution Rule | Objective |
| :--- | :--- | :--- | :--- |
| **Scenario 1** | **Pure ICT** | Take every valid single-pattern signal detected. | Establish baseline performance of un-filtered ICT patterns. |
| **Scenario 2** | **ICT Confluence** | Take trades only when 2+ patterns align in the same direction within a short time window (e.g., FVG + OB). | Test if pattern overlap increases win rate and risk-to-reward ratio. |
| **Scenario 3** | **ML-Filtered ICT** | Take a single-pattern trade *only if* the ML model predicts a positive outcome (1). | Measure the statistical value added by Machine Learning filters. |
| **Scenario 4** | **ML-Filtered Confluence** | Take trades *only when* multiple confluent patterns occur AND the ML model predicts success. | Evaluate performance under maximum conviction setups. |

---

## 5. Portfolio & Risk Management Engine
All backtesting scenarios route trade volume through an institutional risk management layer based on quantitative portfolio theory:

* **Volatility Parity Sizing:** Dynamically adjusts trade sizing so higher-volatility assets (e.g., Silver) do not disproportionately dominate overall portfolio risk compared to lower-volatility assets.
* **Confidence-Weighted Sizing:** Scales position sizes proportional to the prediction probability outputted by the ML model.
* **Drawdown Limits & Risk Capital:** Enforces maximum risk per trade (e.g., 1–2% portfolio equity) and hard stop rules for maximum account drawdown.

