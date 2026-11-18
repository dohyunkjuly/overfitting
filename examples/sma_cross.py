# SMA Crossover Strategy (Hourly BTC)
# ============
#
# This strategy demonstrates a simple trading strategy that uses
# **Simple Moving Average (SMA) crossovers** to make trading decisions.
# 
# Specifically:
# - The strategy buys when the 20-period SMA crosses **above** the 50-period SMA ("golden cross").
# - It exits the position when the 20-period SMA crosses **below** the 50-period SMA ("death cross").
#
# We'll apply this logic to hourly BTCUSDT data, using a simple backtesting framework
# and visualize the result.
#
# Before we begin, we load and process the data, and calculate the necessary indicators.

# +
import pandas as pd
from overfitting import Strategy
from overfitting.indicators import sma, crossover, crossunder

def load_data():
    df = pd.read_csv('./data/BTCUSDT.csv')
    benchamrk_df = pd.read_csv('./data/BTCUSDT.csv') # BTC buy and Hold
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    start_time = pd.to_datetime('2023-01-01 00:00:00')
    df = df.loc[start_time:]

    return df, benchamrk_df

price_df, benchmark_data = load_data()
backtest_data = {"BTC": price_df}
print(price_df.head())
# -

# The Strategy Conditions:
#
# - **Buy Entry (Golden Cross)**:
#     - When 20-SMA crosses above 50-SMA
#     - And there's no existing position
# - **Exit (Death Cross)**:
#     - When 20-SMA crosses below 50-SMA
#     - And a long position is open
#
# Now let's code the strategy logic

# +
class MyStrategy(Strategy):
    def init(self):
        self.asset = 'BTC'
        self.set_leverage(self.asset, 1)

        close = self.data[self.asset].close
        self.fast = self.I(sma, close, 20)
        self.slow = self.I(sma, close, 50)

    def next(self):
        position = self.get_position(self.asset)

        # Golden cross (entry)
        if crossover(self.fast, self.slow) and position.qty == 0:
            price = self.price(self.asset)
            lot_size = self.get_balance() // price
            self.limit_order(self.asset, lot_size, price)

        # Death cross (exit)
        elif crossunder(self.fast, self.slow) and position.qty > 0:
            self.market_order(self.asset, -position.qty)
# -

# Let's run the strategy and plot the results.

# +
strategy = MyStrategy(
    data=backtest_data,
    benchmark=benchmark_data,
    initial_capital=100_000,
    commission_rate=0.0002,
    maint_margin_rate=0.005,
    maint_amount=50,
    slippage=0.001,   # every fill is 0.1% worse than the order price
)
returns = strategy.run()
strategy.plot(returns)
# -

# You can also fetch the trade history for analysis.

strategy.fetch_trades()
