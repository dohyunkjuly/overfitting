# Multiple Currency SMA Crossover (Hourly)
# ============
#
# This strategy demonstrates a simple SMA crossover trading two different assets (BTC, ETH)
# 
# Strategy Logic:
# - The strategy buys when the 20-period SMA crosses **above** the 50-period SMA ("golden cross").
# - It exits the position when the 20-period SMA crosses **below** the 50-period SMA ("death cross").
# - Asset Allocation**: 50% BTC 50% ETH
#
# Before we begin, we need to load the data (OHLCV) and process the indicator

# +
import pandas as pd
from typing import Dict
from overfitting import Strategy

# Constants
DATA_PATHS={
    "BTC": "./data/BTCUSDT.csv",
    "ETH": "./data/ETHUSDT.csv"
}
START_TIME="2023-01-01 00:00:00"
END_TIME="2024-08-29 00:00:00"


def load_data(data_path: Dict, start_time, end_time) -> Dict[str, pd.DataFrame]:
    
    data: Dict[str, pd.DataFrame] = {}
    
    for asset, path in data_path.items(): 
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.drop(columns=["volume"], axis=1)
        
        # Currently for every symbols timestamps has to be identical
        start_time = pd.to_datetime(start_time)
        end_Time = pd.to_datetime(end_time)
        df = df.loc[start_time:end_Time]

        # Compute short and long SMAs
        df['sma_short'] = df['close'].rolling(window=20).mean().shift(1)
        df['sma_long'] = df['close'].rolling(window=50).mean().shift(1)

        data[asset] = df
    
    return data

backtest_data = load_data(DATA_PATHS, START_TIME, END_TIME)
print(backtest_data['ETH'].head())

# Strategy Definition
class MyStrategy(Strategy):
    def init(self):
        self.universe = ['BTC', 'ETH']
        # Set Every symbol leverage to 1
        for asset in self.universe:
            self.set_leverage(asset, 1)

    def next(self, i):
        for asset in self.universe:
            if i == 0:
                return

            sma_short = self.val(asset, i, "sma_short")
            sma_long = self.val(asset, i, "sma_long")
            previous_sma_short = self.val(asset, i - 1, "sma_short") 
            previous_sma_long = self.val(asset, i - 1, "sma_long")

            # Also skip if values are not available
            if (pd.isna(sma_short) or pd.isna(sma_long) or 
                pd.isna(previous_sma_short) or pd.isna(previous_sma_long)):
                return

            # Fetch the current position
            position = self.get_position(asset)

            # Golden cross (entry)
            if previous_sma_short <= previous_sma_long and sma_short > sma_long and position.qty == 0:
                # First fetch current open price which is the target Price
                open_price = self.open(asset, i)
                # Determine Lot Size - 50% BTC 50% ETH
                lot_size = (self.get_balance() / 2) // open_price
                # Create LIMIT ORDER
                self.limit_order(asset, lot_size, open_price)

            # Death cross (exit)
            if previous_sma_short >= previous_sma_long and sma_short < sma_long and position.qty > 0:
                self.market_order(asset, -position.qty)

# Run the strategy and plot the results.
strategy = MyStrategy(data=backtest_data)
returns = strategy.run()
strategy.plot(returns)

# Fetch the trade history for analysis.
strategy.fetch_trades()