# Trading Signal Prediction Project Documentation

**Last Updated:** July 2025  

**Guide: ** |: With or 'Or' _depending on context

           ->: Update to
	   =>: Because | Result _depending on context



**Objective:** Develop a predictive system to generate profitable trading signals from EUR/USD forex data

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Architecture](#project-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Models Implemented](#models-implemented)
6. [Experimental Results](#experimental-results)
7. [Key Findings & Insights](#key-findings--insights)
8. [Code Organization](#code-organization)
9. [Next Steps](#next-steps)

---

## Executive Summary

### Primary Goal
Determine a predictive way to generate trading signals from historical EUR/USD data that produces consistent profitability in MetaTrader backtesting.

### Key Metrics
- **Primary**: Composite Score = (buy_precision + sell_precision)/2 - 0.25 * |buy_precision - sell_precision|
- **Secondary**: P&L, Win Rate -> Sharpe Ratio
- **Backtesting**: MetaTrader | 0.1 lot size & R/R = (1/2, 1/3)

### Current Status
- Best composite score: 0.37 (LGBM, inconsistent)
- Multiple approaches tested: Classification, Regression, Ensemble
- Core issue: Models optimize accuracy, not profitability

---

## Project Architecture

### Pipeline
```
Raw Data (1-min bars) 
    ↓
Feature Engineering
    ↓
Signal Generation (TP/SL Logic)
    ↓
Model Training
    ↓
Predictions
    ↓
MetaTrader Backtest
```

### Signal Logic Evolution

#### 1. Fixed Horizon (Original)
- Question: "Will price move 10+ pips in next N hours?"
- Result: Time-dependent, brittle

#### 2. TP/SL Logic (Current Best)
- Question: "Will price hit TP before SL?"
- Parameters: 24h lookahead, various R:R ratios tested
- Pros: More robust, learns price structure.
- Cons: Static -> Dynamic 


---

## Data Pipeline

### 1. Raw Data
- **Source**: EUR/USD 1-minute bars
- **Period**: 5 years (expandable to 20 years)
- **Format**: Time, Open, High, Low, Close (Volume dropped) => Inaccurate

### 2. Data Cleaning
- **Weekend Handling**: Initially forward-filled, now removed => No noticeable difference.
- **Entry Logic**: Corrected to enter at NEXT candle's open price => reflects real-world trading.
- **Near-Miss Analysis**: ~8% of losses nearly hit TP first

### 3. Signal Generation Scripts
- `canonical_data_script.py`: Generates TP/SL signals with proper entry logic
- `1b_generate_data_tp_sl.py`: Original TP/SL implementation

### Key Data Insights
- **24h lookahead optimal**: Best balance of signal quantity and quality
- **R:R = 1:2 (20:10 pips)**: Trader requirement
- **No inherent class imbalance**: Balanced signals in 24h window

---

## Feature Engineering

### Base Features 
```python
- OHLC price data 
- Returns (open, high, low, close)
- Body size, High-Low range
- Time features (hour/day sin/cos)
```

### Advanced Features
```python
# Multi-timeframe (5m, 15m, 30m, 1h, 4h)
- Rolling means, std, ranges
- Price position within range

# Volatility
- ATR (14, 30, 60, 240 periods)
- Realized volatility
- Volatility regime changes

# Market Microstructure
- Time since significant move
- Average pip movement per hour
- Session overlaps (London, NY, Tokyo)
- Tick counts as volume proxy

# Support/Resistance
- Rolling highs/lows
- Distance to levels
- Pivot points
```

### Feature Importance Findings
- **4h lookahead**: hour_cos dominates (time of day critical)
- **24h lookahead**: high, close, day_sin most important (day of week critical)

---

## Models Implemented

### 1. LightGBM (Baseline)
- **Best Score**: 0.37 (once, not reproducible)
- **Typical Score**: 0.20-0.25
- **Key Issue**: Overfits to 'keep' class

### 2. GTN (Gated Transformer Network)
- **Architecture**: Dual tower (features × time)
- **Best Score**: 0.36 (weighted cross-entropy)
- **Improvements**: Time embeddings > sin/cos encoding

### 3. CatBoost with Sharpe Optimization
- **Innovation**: Optimize Sharpe ratio, not accuracy
- **Features**: Confidence thresholding, Kelly sizing
- **Status**: New implementation

### 4. Regression Approach
- **Targets**: Expected pips, optimal R:R, success probability
- **Innovation**: Multi-stage decisions, volatility-adaptive
- **Status**: New implementation

### 5. Ensemble with Regime Detection
- **Regimes**: Trending up/down, ranging, volatile
- **Features**: Dynamic TP/SL, regime-specific models
- **Status**: New implementation

### 6. Direct P&L Optimization
- **Method**: Neural network with custom P&L loss
- **Features**: Kelly criterion, position sizing
- **Status**: New implementation

### Inconclusive Experiments
- **Rocket Classifier**: Overfitted badly to keep
- **MoE (Mixture of Experts)**: Model too small to benefit
- **SwiGLU Activation**: No improvement
- **Two-Stage LGBM Model**: Filter Model -> Expert Model => Inconsistent back-testing results 

---

## Key Findings & Insights

### Strategic Findings
1. **TP/SL Logic > Fixed Horizon**: The TP/SL signal logic is more robust and realistic.
2. **24h Lookahead is Optimal**: This timeframe provides the best balance of signal quantity and quality for the TP/SL logic.
3. **Composite Score ≠ Profitability**: The primary offline metric (Composite Score) has not proven to be a reliable predictor of real-world P&L in MetaTrader backtesting. This is one of the core challenges of the project.
4. **Inconsistent Results**: Similar model configurations can produce different results across runs, indicating sensitivity to initial conditions or other stochastic factors.
5. **Models Overfit to "Keep"**: A persistent challenge is the models' tendency to favor the majority "keep" class, leading to a low number of trade signals. -> Tackled with weighted cross-entropy & custom cost matrix with penalty & focal_loss & undersampling -> custom cost matrix with weighted cross-entropy > other methods.
6. **MQ5 CSV Encoding**: A critical discovery was that MetaTrader requires CSV signal files to be saved with `UTF-16-LE` encoding, not the standard `UTF-8`.
7. **LabelEncoder Behavior**: The `sklearn.LabelEncoder` sorts class names alphabetically (e.g., 'buy', 'keep', 'sell' becomes `[0, 1, 2]`). Therefore, Code must explicitly map these integer predictions back to the desired `(1, 0, -1)` format for MQ5.
8. **Entry Logic Correction**: The signal generation was corrected to use the *next* candle's open price for trade entry, reflecting a realistic trading scenario.
9. **Major error discovered: Sequence creation was unnecessary for tree-based models like CatBoost, LGBMs, causing hours-long delays.**

---

## Next Steps

### Medium-term 
1. **Ensemble Best Models**: Combine top 2-3 approaches
2. **Add Market Data**: Economic calendar, sentiment
3. **Optimize for Drawdown**: Not just returns
4. **Walk-Forward Analysis**: Test on rolling windows

### Long-term 
1. **Reinforcement Learning**: Full trading agent

### Key Questions to Answer

1. Can we achieve positive Sharpe with any approach?
2. Is 1-minute data too noisy for reliable signals?
3. Should we trade less frequently but with higher confidence?

---

### Experiment Protocol
Before each experiment:
1. State hypothesis clearly
2. Isolate single variable
3. Define success metric
4. Estimate cost/benefit
5. Document results immediately

---

