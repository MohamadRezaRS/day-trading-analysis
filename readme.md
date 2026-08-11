# Quantitative Trading Architecture & ICT Pattern Analysis

## Overview
The objective of this repository is to systematically evaluate structural price action concepts—specifically ICT (Inner Circle Trader) patterns—using machine learning and strict out-of-sample simulation.

## System Architecture & Pipeline

### 1. Data Ingestion & Resampling
I engineered a data pipeline to aggregate 1-minute historical tick data across a four-year period. To completely eliminate the risk of temporal desynchronization across higher timeframes, I programmatically constructed the 5-minute, 15-minute, 1-hour, and 1-day timeframes directly from the raw 1-minute base layer.

### 2. Pattern Detection & Vectorized Simulation
I built a detection module capable of identifying four specific market structures across five distinct markets and four timeframes. For execution simulation, I utilized Pure NumPy Vectorized Forward Simulation. By leveraging binary search (`searchsorted`) and `argmax`, the engine achieves instant trade execution modeling. During this detection phase, the system also extracted and stored critical structural features to be used as predictive inputs.

### 3. Hybrid EDA & Feature Engineering System
Dealing with a massive volume of isolated datasets required a hybrid, looping approach between Exploratory Data Analysis (EDA) and Feature Engineering (FE):
*   **Quality Control:** I established a strict statistical threshold. Any dataset containing fewer than 500 instances after dropping NaNs and duplicate rows was immediately discarded.
*   **Data Transformation:** Because certain extracted features were categorical objects, I routed the data into the FE module for transformation before feeding it back into the EDA pipeline for anomaly detection and correlation checks.
*   **Temporal Encoding:** To ensure flexibility when training various models, I scaled and encoded the data. A specific challenge was scaling continuous temporal variables (hours and days of the week). I solved this by applying cyclic encoding using sine and cosine transformations to preserve their cyclical nature.

### 4. Statistical Discovery: The Liquidity Sweep Anomaly
Before training the models, I conducted a baseline win-rate check for every isolated pattern. During this process, I discovered a significant anomaly regarding the Liquidity Sweep pattern: it exhibited a consistent win rate of under 15% across all datasets when traded in its native direction. This structural failure allowed me to mechanically invert the signal in the testing phases to weaponize the trap.

### 5. Machine Learning & Probability Thresholding
Training classification models on financial data inherently involves severe class imbalance. A naive model could simply predict '0' (failure) for every setup and achieve over 65% accuracy. 
*   To counter this, I shifted the evaluation metric to PR-AUC (Precision-Recall Area Under Curve) and relied heavily on `predict_proba` for granular probability thresholding. 
*   Models that failed to outperform the baseline pattern win rate were discarded. 
*   The surviving models underwent rigorous hyperparameter tuning across multiple configurations to maximize out-of-sample predictive performance.

### 6. Out-of-Sample Backtesting (2025 - Mid 2026)
I built a secondary data ingestion pipeline to gather strict, unseen out-of-sample data spanning from January 2025 to mid-2026. I then developed a custom event-driven backtesting engine to evaluate the architecture across four distinct scenarios:
*   **Scenario 1:** A raw, unfiltered baseline.
*   **Scenario 2:** A restricted environment utilizing a Confirmation Pool (one active trade per market).
*   **Scenario 3:** A pure Machine Learning prediction filter.
*   **Scenario 4:** The finalized system combining ML filters, the Confirmation Pool, and dynamic Half-Kelly position sizing.

---

### Disclaimer
The financial market is an environment where institutional giants and quantitative scientists compete using highly advanced infrastructure. While the models in this repository may display profitable out-of-sample equity curves, algorithms of this nature almost certainly fail in live market conditions due to latency, slippage, and shifting macroeconomic regimes. 

This project is not a recommendation for live trading. Its sole purpose was to conduct a rigorous, data-driven analysis of ICT patterns and algorithmic day trading strategies.