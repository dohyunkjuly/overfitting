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
from overfitting.indicator import SMA

# Constants
DATA_PATHS={
    "BTC": "./data/BTCUSDT.csv",
    "ETH": "./data/ETHUSDT.csv"
}
START_TIME="2023-01-01 00:00:00"
END_TIME="2024-08-29 00:00:00"


def load_data(data_paths: Dict[str, str], start_time: str, end_time: str) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    for asset, path in data_paths.items():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        df = df.loc[start_time:end_time]
        data[asset] = df

    return data

backtest_data = load_data(DATA_PATHS, START_TIME, END_TIME)

# Strategy Definition
class MyStrategy(Strategy):
    def init(self):
        self.universe = list(self.data.symbols)

        for asset in self.universe:
            self.set_leverage(asset, 1)

        self.sma_short = {}
        self.sma_long = {}

        for asset in self.universe:
            self.sma_short[asset] = SMA(self, asset, source="close", window=20)
            self.sma_long[asset] = SMA(self, asset, source="close", window=50)

    def next(self, i):
        if i == 0:
            return

        for asset in self.universe:
            sma_short = self.sma_short[asset][i]
            sma_long = self.sma_long[asset][i]
            previous_sma_short = self.sma_short[asset][i - 1]
            previous_sma_long = self.sma_long[asset][i - 1]

            if (
                pd.isna(sma_short) or pd.isna(sma_long) or
                pd.isna(previous_sma_short) or pd.isna(previous_sma_long)
            ):
                continue

            position = self.get_position(asset)

            # Golden cross (entry)
            if previous_sma_short <= previous_sma_long and sma_short > sma_long and position.qty == 0:
                open_price = self.open(asset, i)
                lot_size = (self.get_balance() / 2) // open_price

                if lot_size > 0:
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