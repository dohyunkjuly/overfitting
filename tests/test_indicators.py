"""Indicator functions, Line indexing, and declaring indicators with Strategy.I."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from overfitting import Strategy
from overfitting.indicators import (
    Line, Lines, atr, bollinger, crossover, crossunder, donchian, ema, macd, rsi, sma,
    stochastic, wma,
)

approx = pytest.approx
CLOSE = pd.Series([10.0, 11, 12, 11, 13, 14, 13, 15, 16, 15])
HIGH, LOW = CLOSE + 1, CLOSE - 1


# ---------------------------------------------------------------- functions ---

def test_sma():
    out = sma(CLOSE, 3)
    assert out.isna().sum() == 2
    assert out.iloc[2] == approx(11) and out.iloc[-1] == approx((15 + 16 + 15) / 3)


def test_ema_weights_recent_bars_more():
    out = ema(CLOSE, 3)   # alpha = 0.5
    assert out.iloc[0] == 10
    assert out.iloc[1] == approx(10.5)
    assert out.iloc[2] == approx(11.25)


def test_wma():
    assert wma(CLOSE, 3).iloc[2] == approx((10 * 1 + 11 * 2 + 12 * 3) / 6)


def test_rsi_extremes_and_range():
    rising = pd.Series(np.arange(1.0, 40))
    falling = rising[::-1].reset_index(drop=True)

    assert rsi(rising, 14).iloc[-1] == approx(100)
    assert rsi(falling, 14).iloc[-1] == approx(0)
    assert rsi(CLOSE, 3).dropna().between(0, 100).all()


def test_macd_columns_and_identity():
    out = macd(CLOSE, 3, 6, 2)
    assert list(out.columns) == ["macd", "signal", "hist"]
    assert (out["hist"] == out["macd"] - out["signal"]).all()
    assert out["macd"].iloc[-1] == approx(ema(CLOSE, 3).iloc[-1] - ema(CLOSE, 6).iloc[-1])


def test_stochastic_is_100_at_the_top_of_the_range():
    close = pd.Series([1.0, 2, 3, 4, 5, 6])
    out = stochastic(close, close, close, k_window=3, d_window=2, smooth_k=1)
    assert list(out.columns) == ["k", "d"]
    assert out["k"].iloc[-1] == approx(100)


def test_bollinger_bands_are_symmetric_around_the_average():
    out = bollinger(CLOSE, 4, 2)
    row = out.iloc[-1]
    assert row["middle"] == approx(CLOSE.iloc[-4:].mean())
    assert row["upper"] - row["middle"] == approx(row["middle"] - row["lower"])
    assert row["upper"] - row["middle"] == approx(2 * CLOSE.iloc[-4:].std())
    assert row["bandwidth"] == approx((row["upper"] - row["lower"]) / row["middle"])


def test_atr_of_constant_range_bars():
    flat = pd.Series([100.0] * 30)
    assert atr(flat + 2, flat - 2, flat, 5).iloc[-1] == approx(4)


def test_donchian_channel():
    out = donchian(HIGH, LOW, 3)
    assert out["upper"].iloc[-1] == 17 and out["lower"].iloc[-1] == 14
    assert out["middle"].iloc[-1] == approx(15.5)


@pytest.mark.parametrize("function", [sma, ema, wma, rsi])
def test_functions_do_not_shift_their_output(function):
    # The value at a bar may use that bar: changing the last close changes the last value.
    bumped = CLOSE.copy()
    bumped.iloc[-1] += 5
    assert function(CLOSE, 3).iloc[-1] != function(bumped, 3).iloc[-1]
    assert function(CLOSE, 3).iloc[-2] == function(bumped, 3).iloc[-2]


# -------------------------------------------------------------------- lines ---

class Cursor:
    def __init__(self, i):
        self.i = i


def test_line_shows_only_completed_bars():
    cursor = Cursor(0)
    line = Line([1.0, 2.0, 3.0, 4.0], cursor)

    assert len(line) == 0 and np.isnan(line[-1])      # nothing has completed yet
    cursor.i = 1
    assert line[-1] == 1.0 and np.isnan(line[-2])
    cursor.i = 3
    assert line[-1] == 3.0 and line[-2] == 2.0 and line[0] == 1.0
    assert len(line) == 3


def test_line_cannot_read_the_current_or_future_bars():
    line = Line([1.0, 2.0, 3.0, 4.0], Cursor(2))

    assert np.isnan(line[2]) and np.isnan(line[3])
    assert list(line[:]) == [1.0, 2.0]
    assert list(line[-5:]) == [1.0, 2.0]


def test_line_slices_ready_and_values():
    line = Line([np.nan, 2.0, 3.0, 4.0], Cursor(1), "sma")

    assert not line.ready
    line._cursor.i = 4
    assert line.ready
    assert list(line[-2:]) == [3.0, 4.0]
    assert len(line.values) == 4                      # the whole series stays available
    assert "sma" in repr(line)


def test_lines_expose_each_column():
    frame = pd.DataFrame({"upper": [3.0, 4.0], "lower": [1.0, 2.0]})
    lines = Lines(frame, Cursor(2))

    assert lines.upper[-1] == 4.0 and lines["lower"][-1] == 2.0
    assert [line.name for line in lines] == ["upper", "lower"]
    assert list(lines.values.columns) == ["upper", "lower"]


def test_crossover_and_crossunder():
    cursor = Cursor(2)
    fast = Line([1.0, 3.0, 1.0], cursor)
    slow = Line([2.0, 2.0, 2.0], cursor)

    assert crossover(fast, slow) and not crossunder(fast, slow)
    cursor.i = 3
    assert crossunder(fast, slow) and not crossover(fast, slow)


def test_crossover_against_a_constant_and_with_missing_values():
    cursor = Cursor(2)
    line = Line([29.0, 31.0, 28.0], cursor)

    assert crossover(line, 30) and not crossunder(line, 30)
    cursor.i = 3
    assert crossunder(line, 30)
    cursor.i = 1
    assert not crossover(line, 30) and not crossunder(line, 30)   # no previous bar yet


def test_touching_is_not_a_cross():
    cursor = Cursor(3)
    assert not crossover(Line([1.0, 2.0, 2.0], cursor), Line([2.0, 2.0, 2.0], cursor))


# --------------------------------------------------------------- Strategy.I ---

BARS = [(c, c + 1, c - 1, c) for c in CLOSE]


def strategy_with(init, next_=None, bars=BARS, **kwargs):
    namespace = {"init": init, "next": next_ or (lambda self: None)}
    return type("Case", (Strategy,), namespace)(
        data={"BTC": make_frame(bars)}, initial_capital=10_000, commission_rate=0.0, **kwargs)


def test_I_wraps_a_function_of_price_arrays():
    def init(self):
        self.fast = self.I(sma, self.data["BTC"].close, 3)

    strategy = strategy_with(init)
    assert isinstance(strategy.fast, Line) and strategy.fast.name == "sma"
    assert strategy.fast.values.iloc[-1] == approx(sma(CLOSE, 3).iloc[-1])


def test_I_accepts_keywords_frames_tuples_and_raw_series():
    def init(self):
        bars = self.data["BTC"]
        self.bands = self.I(bollinger, bars.close, window=4, num_std=2)
        self.pair = self.I(lambda c: (c + 1, c - 1), bars.close)
        self.closes = self.I(bars.close)

    strategy = strategy_with(init)
    assert isinstance(strategy.bands, Lines)
    assert strategy.bands.upper.values.iloc[-1] == approx(bollinger(CLOSE, 4, 2)["upper"].iloc[-1])
    assert len(strategy.pair) == 2 and isinstance(strategy.pair[0], Line)
    assert list(strategy.closes.values) == list(CLOSE)


def test_indicators_can_be_built_on_other_indicators():
    def init(self):
        self.fast = self.I(sma, self.data["BTC"].close, 2)
        self.smooth = self.I(sma, self.fast, 2)

    strategy = strategy_with(init)
    assert strategy.smooth.values.iloc[-1] == approx(sma(sma(CLOSE, 2), 2).iloc[-1])


def test_I_rejects_output_of_the_wrong_length():
    def init(self):
        self.bad = self.I(lambda c: c.iloc[:-1], self.data["BTC"].close)

    with pytest.raises(ValueError, match="one per bar"):
        strategy_with(init)


def test_next_without_an_index_sees_completed_bars_only():
    seen = []

    def init(self):
        self.closes = self.I(self.data["BTC"].close)

    def next_(self):
        seen.append((self.i, len(self.closes), self.closes[-1], self.price("BTC")))

    strategy_with(init, next_).run()

    assert seen[0][:2] == (0, 0) and np.isnan(seen[0][2])     # first bar: nothing completed
    assert seen[1] == (1, 1, 10.0, 11.0)                      # last close 10, about to trade at 11
    assert seen[-1] == (9, 9, 16.0, 15.0)


def test_next_with_an_index_still_works():
    seen = []

    def next_(self, i):
        seen.append(i)

    strategy_with(lambda self: None, next_).run()
    assert seen == list(range(10))


def test_new_style_strategy_trades_on_crossovers():
    # Falls to 6, then rises: the 2-bar average crosses above the 4-bar one on bar 6
    # (7.5 against 7.0). That is known once bar 6 has closed, so the buy fills at
    # the open of bar 7.
    closes = [10.0, 9, 8, 7, 6, 7, 8, 9, 10, 11]

    def init(self):
        close = self.data["BTC"].close
        self.fast = self.I(sma, close, 2)
        self.slow = self.I(sma, close, 4)

    def next_(self):
        position = self.get_position("BTC")
        if crossover(self.fast, self.slow) and position.qty == 0:
            self.market_order("BTC", 1)
        elif crossunder(self.fast, self.slow) and position.qty > 0:
            self.market_order("BTC", -1)

    strategy = strategy_with(init, next_, bars=[(c, c + 1, c - 1, c) for c in closes])
    strategy.run()
    fills = strategy.fetch_trades()

    assert len(fills) == 1
    assert fills.loc[0, "executed_at"] == pd.Timestamp("2024-01-01 07:00")
    assert fills.loc[0, "executed_price"] == 9
