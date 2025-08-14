# Overfitting

A robust and modular backtesting engine designed for crypto futures trading strategies.  
Built for speed, simplicity, and accuracy. Overfitting simulates a realistic crypto trading environment — including **liquidation**, **margin**, and **leverage** — for stress-testing your strategies.

## Prerequisites

Before using **Overfitting**, you’ll need to provide your own historical data.  
The engine is designed to work with **crypto futures price data**, with **OHLCV format**.

### Required Columns

Your dataset must be a CSV or DataFrame that includes at least the following columns:
- timestamp, open, high, low, close
  - `timestamp` should be a **UNIX timestamp in seconds or milliseconds**

## Installation
    $ pip install overfitting


## Usage
```python
import pandas as pd
from overfitting import Strategy

def load_data():
    df = pd.read_csv('./data/BTCUSDT.csv') # You will need to have your own DATA!
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df.loc['2023-01-01':]

    df['sma_short'] = df['close'].rolling(window=20).mean().shift()
    df['sma_long'] = df['close'].rolling(window=50).mean().shift()
    return df

class MyStrategy(Strategy):
    def init(self):
        self.asset = 'BTC'
        self.set_leverage(self.asset, 1)

    def next(self, i):
        short = self.data.sma_short[i]
        long = self.data.sma_long[i]

        if pd.isna(short) or pd.isna(long):
            return

        prev_short = self.data.sma_short[i - 1]
        prev_long = self.data.sma_long[i - 1]
        if pd.isna(prev_short) or pd.isna(prev_long):
            return

        price = self.data.open[i]
        lot_size = self.get_balance() // price
        p = self.get_position(self.asset)

        if prev_short <= prev_long and short > long and p.qty == 0:
            self.limit_order(self.asset, lot_size, price)

        if prev_short >= prev_long and short < long and p.qty > 0:
            self.market_order(self.asset, -p.qty)

data = load_data()
strategy = MyStrategy(data)
returns = strategy.run()
strategy.plot(returns)
```

Results
-------
```text
Performance Summary
Number of Years               1.70000000
Start Date           2023-01-01 00:00:00
End Date             2024-08-29 00:00:00
Initial Balance         100,000.00000000
Final Balance           202,802.51658000
CAGR                          0.51576326
Cumulative Return             2.02802517
Sharpe Ratio                  1.22963908
Sortino Ratio                 3.50674547
Max Drawdown                 -0.27312998
Daily Value At Risk          -0.04143807
Skew                          0.31909418
Kurtosis                      2.60022470
Total Trades                181.00000000
Winning Trades               68.00000000
Losing Trades               113.00000000
Win Rate (%)                 37.56906077
Gross Profit            391,161.01938000
Gross Loss             -288,358.50280000
Net Profit              102,802.51658000
Avg Return (%)                0.38386677
Avg Profit (%)                3.53708812
Avg Loss (%)                 -1.51364697
  Net drawdown in %  Peak date Valley date Recovery date Duration
0         27.312998 2024-03-13  2024-06-30           NaT      NaN
1         19.678014 2023-03-20  2023-09-07    2023-10-26      159
2          6.297244 2023-12-07  2024-01-24    2024-02-14       50
3          5.585429 2023-01-22  2023-02-14    2023-02-17       20
4          3.898568 2023-02-17  2023-03-11    2023-03-15       19
5          3.336877 2023-11-12  2023-11-18    2023-12-07       19
6          2.699556 2024-02-20  2024-02-26    2024-03-01        9
7          0.767196 2024-03-01  2024-03-03    2024-03-06        4
8          0.324161 2023-01-03  2023-01-07    2023-01-18       12
9          0.019817 2023-11-03  2023-11-04    2023-11-07        3
```

## Performance Visualizations Examples

![Cumulative Returns](https://raw.githubusercontent.com/dohyunkjuly/overfitting/main/documents/culmulative_returns.png)
![Daily Drawdowns](https://raw.githubusercontent.com/dohyunkjuly/overfitting/main/documents/daily_drawdowns.png)
![Monthly Heat Maps](https://raw.githubusercontent.com/dohyunkjuly/overfitting/main/documents/monthly_heat_maps.png)
![Rolling Sharpe Ratio](https://raw.githubusercontent.com/dohyunkjuly/overfitting/main/documents/rolling_sharpe_ratio.png)

## Liquidation Handling

Unlike many basic backtesting engines, **overfitting** simulates realistic crypto futures trading, including **forced liquidation** based on margin conditions.

The liquidation logic is based on **isolated margin mode** (similar to Binance Futures):

- **Initial Margin** = Entry Price × Quantity / Leverage  
- **Maintenance Margin** = Entry Price × Quantity × Maintenance Margin Rate − Maintenance Amount  
- **Liquidation Price** is then calculated based on whether the position is long or short.

When the price crosses the calculated liquidation level, the position is force-closed and the **entire margin is lost**, just like in real crypto markets.

### Liuqidation Calculation

```python
# For long positions
liquid_price = entry_price - (initial_margin - maintenance_margin)

# For short positions
liquid_price = entry_price + (initial_margin - maintenance_margin)
```

## Types Of Orders
Supports four order types: LIMIT, MARKET, STOP LIMIT, and STOP MARKET. Each behaves according to standard trading conventions.

[NOTE] For MAKRET Orders, the system will automatically execute the trade with "open" price.

**The Rules for Stop Order to Trigger is:** <br>
LONG: Price (High) >= Stop Price <br>
SHORT: Price (low) <= Stop Price
```python
# For Long qty > 0 for short qty < 0
# Example 1. if qty == -1. This means Short
# Example 2. if qty == 1. This means Long
limit_order(symbol: str, qty: float, price: float)
market_order(symbol: str, qty: float)
stop_limit_order(symbol: str, qty: float, price: float, stop_price: float)
stop_market_order(symbol: str, qty: float, stop_price: float)
```

### Stop Order Immediate Rejection Rule
If a STOP LIMIT or STOP MARKET order would trigger immediately upon creation (because the current price already breaches the stop price), the system rejects the order with "STOP order would Immediately Trigger" message.

## Upcoming Features

- **Parameter Optimizer**  
  A simple optimizer to help find the best-performing strategy parameters (like SMA windows, thresholds, etc.) based on backtest results.

- **Improved Slippage Modeling**  
  Dynamic slippage models based on volume, volatility, or order size.

> 💡 Got feedback or suggestions? Feel free to open an issue or contribute via pull request.
