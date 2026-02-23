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
from overfitting import Strategy, Slippage

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

backtest_data, benchmark_data = load_data()
print(backtest_data.head())
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
    slippage_model=Slippage.FixedPercent(f=0.001)
)
returns = strategy.run()
strategy.plot(returns)
# -

# You can also fetch the trade history for analysis.

strategy.fetch_trades()
