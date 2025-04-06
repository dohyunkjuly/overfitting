# SMA Crossover Strategy (Hourly BTC)
# ============
#
# This notebook demonstrates a simple trading strategy that uses
# **Simple Moving Average (SMA) crossovers** to make trading decisions.
# 
# Specifically:
# - The strategy buys when the 20-period SMA crosses **above** the 50-period SMA ("golden cross").
# - It exits the position when the 20-period SMA crosses **below** the 50-period SMA ("death cross").
#
# We'll apply this logic to hourly BTCUSDT data, using a simple backtesting framework
# and visualize the result.
#
# Before we begin, we load and process our data, and calculate the necessary indicators.

# +
import pandas as pd
from overfitting import Strategy  # Your custom backtesting framework

def load_data():
    df = pd.read_csv('./data/binance_futures_BTCUSDT_1h 2019-09-09-2024-08-29.csv')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    start_time = pd.to_datetime('2023-01-01 00:00:00')
    df = df.loc[start_time:]

    # Compute short and long SMAs
    df['sma_short'] = df['close'].rolling(window=20).mean().shift()
    df['sma_long'] = df['close'].rolling(window=50).mean().shift()

    return df

backtest_data = load_data()
print(backtest_data.head())
# -

# The strategy works as follows:
#
# - **Buy Entry (Golden Cross)**:
#     - When 20-SMA crosses above 50-SMA
#     - And there's no existing position
# - **Exit (Death Cross)**:
#     - When 20-SMA crosses below 50-SMA
#     - And a long position is open
#
# The strategy trades the BTCUSDT pair on hourly data with full capital allocation at each signal.

# +
class MyStrategy(Strategy):
    def init(self):
        self.asset = 'BTC'
        self.set_leverage(self.asset, 1)

    def next(self, i):
        short = self.data.sma_short[i]
        long = self.data.sma_long[i]

        # Skip if SMA values aren't available yet
        if pd.isna(short) or pd.isna(long):
            return

        prev_short = self.data.sma_short[i - 1]
        prev_long = self.data.sma_long[i - 1]

        # Also skip if previous values are not available
        if pd.isna(prev_short) or pd.isna(prev_long):
            return

        price = self.data.open[i]
        lot_size = self.get_balance() // price
        p = self.get_position(self.asset)

        # Golden cross (entry)
        if prev_short <= prev_long and short > long and p.qty == 0:
            self.limit_order(self.asset, lot_size, price)

        # Death cross (exit)
        if prev_short >= prev_long and short < long and p.qty > 0:
            self.market_order(self.asset, -p.qty)
# -

# Let's run the strategy and plot the results.

# +
strategy = MyStrategy(backtest_data)
returns = strategy.run()
strategy.plot(returns)
# -

# You can also fetch the trade history for analysis.

strategy.fetch_trades()
