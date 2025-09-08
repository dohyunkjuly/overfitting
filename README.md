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
    df = pd.read_csv('./data/BTCUSDT.csv')
    benchamrk_df = pd.read_csv('./data/BTCUSDT.csv') # BTC buy and Hold
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    start_time = pd.to_datetime('2023-01-01 00:00:00')
    df = df.loc[start_time:]
    # Compute short and long SMAs
    df['sma_short'] = df['close'].rolling(window=20).mean().shift(1)
    df['sma_long'] = df['close'].rolling(window=50).mean().shift(1)

    return df, benchamrk_df

class MyStrategy(Strategy):
    def init(self):
        self.asset = 'BTC'
        self.set_leverage(self.asset, 1)

    def next(self, i):
        if i == 0:
            return

        sma_short = self.val(self.asset, i, "sma_short")
        sma_long = self.val(self.asset, i, "sma_long")
        previous_sma_short = self.val(self.asset, i - 1, "sma_short") 
        previous_sma_long = self.val(self.asset, i - 1, "sma_long")

        # Also skip if values are not available
        if (pd.isna(sma_short) or pd.isna(sma_long) or 
            pd.isna(previous_sma_short) or pd.isna(previous_sma_long)):
            return

        # Fetch the current position
        position = self.get_position(self.asset)

        # Golden cross (entry)
        if previous_sma_short <= previous_sma_long and sma_short > sma_long and position.qty == 0:
            # First fetch current open price which is the target Price
            open_price = self.open(self.asset, i)
            # Determine Lot Size
            lot_size = self.get_balance() // open_price
            # Create LIMIT ORDER
            self.limit_order(self.asset, lot_size, open_price)

        # Death cross (exit)
        if previous_sma_short >= previous_sma_long and sma_short < sma_long and position.qty > 0:
            self.market_order(self.asset, -position.qty)

backtest_data, benchmark_data = load_data()
strategy = MyStrategy(
    data=backtest_data,
    benchmark=benchmark_data, # Default = None Optional
    initial_capital=100_000, # Default Optional
    commission_rate=0.0002, # Default Optional
    maint_maring_rate=0.005, # Default Optional
    maint_amount=50  # Default Optional
)
returns = strategy.run()
strategy.plot(returns)
```

Results
-------
```text
Performance Summary
Number of Years               1.66000000
Start Date           2023-01-01 00:00:00
End Date             2024-08-29 00:00:00
Initial Balance         100,000.00000000
Final Balance           205,328.91120000
CAGR                          0.52684228
Cumulative Return             2.05328911
Sharpe Ratio                  1.24678659
Sortino Ratio                 3.54979579
Max Drawdown                 -0.26332695
Daily Value At Risk          -0.04147282
Skew                          0.44515551
Kurtosis                      2.66444346
Total Trades                182.00000000
Winning Trades               69.00000000
Losing Trades               113.00000000
Win Rate (%)                 37.91208791
Gross Profit            399,044.19246000
Gross Loss             -293,715.28126000
Net Profit              105,328.91120000
Avg Return (%)                0.38834383
Avg Profit (%)                3.54140613
Avg Loss (%)                 -1.53697740
  Net drawdown in %  Peak date Valley date Recovery date Duration
0         26.332695 2024-03-13  2024-06-30           NaT      NaN
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

## Supported Order Types
Supports four order types: LIMIT, MARKET, STOP LIMIT, and STOP MARKET. Each behaves according to standard trading conventions.

[NOTE] For MAKRET Orders, the system will automatically execute the trade with "open" price.

**The Rules for Stop Order to Trigger is:** <br>
LONG: Price (High) >= Stop Price <br>
SHORT: Price (low) <= Stop Price
```python
# For Long qty > 0 for short qty < 0
# Example 1. if qty == -1. This means Position is Short
# Example 2. if qty == 1. This means Position is Long
limit_order(symbol: str, qty: float, price: float)
market_order(symbol: str, qty: float)
stop_limit_order(symbol: str, qty: float, price: float, stop_price: float)
stop_market_order(symbol: str, qty: float, stop_price: float)
```

### Stop Order Immediate Rejection Rule
If a STOP LIMIT or STOP MARKET order would trigger immediately upon creation (because the current price already breaches the stop price), the system rejects the order with "STOP order would Immediately Trigger" message.

## Multiple Currency Backtesting
You can simply test multiple currencies by passing data as dict[str, pd.DataFrame]. Here key value should be the name of the currency.

### **[IMPORTANT]** When you are running simulations in multi currency mode please make sure that "timestamp" are identical for every symbols

## Helper Functions for Strategy Definitions
```python
class MyStrategy(Strategy):
    def init(self):
        self.asset = 'BTC'
        
    def next(self, i):
        # Fetch the indicator or other custom column from data
        val = self.val(self.asset, i, "the indicator value") 
        # OHLV data
        open  = self.open(self.asset, i)
        high  = self.high(self.asset, i)
        low   = self.low(self.asset, i)
        close = self.close(self.asset, i)
        o, h, l, c = self.bars(self.asset, i)
        # ACCOUNT data
        position = self.get_position(self.asset)
        balance = self.get_balance()
        open_orders = self.get_open_orders() # returns all open orders regardless of symbols
```

## Upcoming Features

- **Parameter Optimizer**  
  A simple optimizer to help find the best-performing strategy parameters (like SMA windows, thresholds, etc.) based on backtest results.

- **Improved Slippage Modeling**  
  Dynamic slippage models based on volume, volatility, or order size.

> 💡 Got feedback or suggestions? Feel free to open an issue or contribute via pull request.
