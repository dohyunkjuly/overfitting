# Donchian Breakout Strategy (Custom Indicator Example)
# ============
#
# Writes a `Donchian` indicator by subclassing `overfitting.indicators.Indicator`,
# then uses it in a Turtle-style breakout on hourly BTC.

# +
import pandas as pd
from overfitting import Strategy
from overfitting.indicators import Indicator

# +
# Define Custom Indicator
class Donchian(Indicator):
    def __init__(self, high: pd.Series, low: pd.Series, window: int = 20):
        self.high = high
        self.low = low
        self.window = window
        super().__init__()

    def compute(self):
        # shift(2) so the channel at bar i excludes bar i-1; lets the strategy
        # compare close[i-1] vs upper[i] meaningfully (with shift(1), bar i-1's
        # high would be inside the window and the comparison is impossible).
        upper = self.high.rolling(self.window).max().shift(2)
        lower = self.low.rolling(self.window).min().shift(2)
        self._values = pd.DataFrame({
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

        self.dc = self.indicator(Donchian, self.asset, window=self.window)

    def next(self, i):
        if i < 2:
            return

        upper = self.dc.upper[i]
        lower = self.dc.lower[i]
        if pd.isna(upper) or pd.isna(lower):
            return

        # Use close[i-1], not close[i]: at the start of bar i, close[i] hasn't
        # happened yet — reading it would be look-ahead bias.
        prev_close = self.close(self.asset, i - 1)
        position = self.get_position(self.asset)

        if prev_close > upper and position.qty == 0:
            open_price = self.open(self.asset, i)
            lot_size = self.get_balance() // open_price
            if lot_size > 0:
                self.market_order(self.asset, lot_size)

        elif prev_close < lower and position.qty > 0:
            self.market_order(self.asset, -position.qty)
# -


# +
strategy = BreakoutStrategy(data=backtest_data, initial_capital=100_000)
returns = strategy.run()
strategy.plot(returns)
# -

# Inspect trades
strategy.fetch_trades()
