# Donchian Breakout Strategy (Custom Indicator Example)
# ============
#
# Writes a `channel` indicator as a plain function, then uses it in a
# Turtle-style breakout on hourly BTC.

# +
import pandas as pd
from overfitting import Strategy

# +
# Define a custom indicator: any function that returns one value per bar.
def channel(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    return pd.DataFrame({
        "upper": upper,
        "lower": lower,
        "mid":   (upper + lower) / 2,
    })
# -


# +
def load_data():
    df = pd.read_csv('./data/BTCUSDT.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    start_time = pd.to_datetime('2023-01-01 00:00:00')
    df = df.loc[start_time:]
    return df


price_df = load_data()
backtest_data = {"BTC": price_df}
# -


# +
class BreakoutStrategy(Strategy):
    def init(self):
        self.asset = 'BTC'
        self.window = 20
        self.set_leverage(self.asset, 1)

        bars = self.data[self.asset]
        self.dc = self.I(channel, bars.high, bars.low, window=self.window)
        self.closes = self.I(bars.close)

    def next(self):
        # [-1] is the last completed bar. The channel is read one bar further
        # back ([-2]), so the bar that just closed is compared with the range
        # of the bars before it. Otherwise its own high would be inside the
        # window and a breakout could never happen.
        last_close = self.closes[-1]
        upper = self.dc.upper[-2]
        lower = self.dc.lower[-2]
        position = self.get_position(self.asset)

        if last_close > upper and position.qty == 0:
            lot_size = self.get_balance() // self.price(self.asset)
            if lot_size > 0:
                self.market_order(self.asset, lot_size)

        elif last_close < lower and position.qty > 0:
            self.market_order(self.asset, -position.qty)
# -


# +
strategy = BreakoutStrategy(data=backtest_data, initial_capital=100_000)
returns = strategy.run()
strategy.plot(returns)
# -

# Inspect trades
strategy.fetch_trades()
