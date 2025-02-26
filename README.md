# Overfitting
# Project Plan: Python Backtesting Tool for Crypto Trading for Futures Trading

## 1. Project Overview
- **Purpose:** A user-friendly backtesting tool that allows **Python developers** to fetch K-line data from multiple exchanges and simulate trading strategies.
- **Target Users:**
  - Beginner to intermediate Python developers
  - Traders who want to validate their strategies

---

## 2. Core Features
### Data Pipeline (Market Data Collection)
- **Fetch K-line (OHLCV) data** from multiple exchanges via APIs.
- **Support for multiple exchanges** (e.g., Binance, Bybit, OKX, KuCoin).
- **Historical & Live data fetching**.
- **Flexible data storage options** (CSV, Pandas DataFrame).

### 🎯 Backtesting Engine
- **Simulate trading strategies** on historical data.
- **Support multiple trading pairs & timeframes**.
- **Custom strategy implementation via Python functions** User should be able to implement their own indicator.

### 📈 Performance Metrics & Analysis
- **PnL Calculation** (profit & loss tracking).
- **Sharpe Ratio, Max Drawdown, Win Rate calculations**.
- **Plotting results using matplotlib**.

---

## 3. Tech Stack
- **Programming Language:** Python
- **Libraries:**  
  - `pandas` → Data handling & analysis  
  - `numpy` → Mathematical operations  
  - `matplotlib` → Visualization  
  - `csv` → Data storage  

---

## 4. Project Architecture
```
📂 Overfitting/
 ├── 📂 data_pipeline/        # Fetching & storing K-line data
 │   ├── fetch_data.py        # Fetch K-line data from exchanges
 │   ├── store_data.py        # Save data (CSV, SQLite, Pandas)
 │   ├── exchange_config.py   # API keys, exchange settings
 │
 ├── 📂 backtesting_engine/   # Strategy simulation & analysis
 │   ├── broker.py        # Broker class to handle order and position objects and calculate liquidation, trading results etc...
 │   ├── error.py         # Error class
 │   ├── order.py         # Order class
 │   ├── position.py      # Futures position class
 │   ├── strategy.py      # Backtesting class which user will be importing to run backtesing
 │   ├── 📂 functions/   # Strategy simulation & analysis
 │   │   ├── data.py         # Candles data
 │   │   ├── type.py         # Utils for enumerators etc
 │   ├── 📂 plot/   # Strategy simulation & analysis
 │   │   ├── graph.py         # 
 │   │   ├── plot.py          # 
 │   │   ├── benchmark.py     # 
 │
 ├── requirements.txt          # Python dependencies
 ├── setup.cfg                 # Python dependencies
 ├── README.md                 # Project documentation
```

---

## 5. Development Plan
| Phase  | Tasks | Expected Time |
|--------|-------|--------------|
| **1 Data Pipeline** | Implement K-line fetching & storage | 3-5 days |
| **2 Backtesting Engine** | Develop core simulation logic | 4-7 days |
| **2 Plot functions** | Refactor Plot functions | 4-7 days |
| **3 Documentation & Release** | Write docs, create examples, publish | 3-5 days |

---
