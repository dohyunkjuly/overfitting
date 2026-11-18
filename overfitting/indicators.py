"""
Indicators.

An indicator is a plain function: it takes one or more price series and
returns a series (or a DataFrame, for indicators with several outputs) of the
same length.

    from overfitting.indicators import sma, crossover

    class MyStrategy(Strategy):
        def init(self):
            close = self.data["BTC"].close
            self.fast = self.I(sma, close, 20)
            self.slow = self.I(sma, close, 50)

        def next(self):
            if crossover(self.fast, self.slow):
                ...

`Strategy.I` wraps the result in a `Line`. Inside `next()` a Line only shows
bars that have finished: `line[-1]` is the last completed bar, `line[-2]` the
one before. The bar being traded and everything after it cannot be read, so an
indicator never needs to shift its output to avoid looking ahead.

A custom indicator is any function with the same shape:

    def donchian_width(high, low, window=20):
        return high.rolling(window).max() - low.rolling(window).min()

    self.width = self.I(donchian_width, bars.high, bars.low, window=20)
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "wma", "rsi", "macd", "stochastic", "bollinger", "atr", "donchian",
    "crossover", "crossunder", "Line", "Lines",
]


# ---------------------------------------------------------------- functions ---

def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def wma(series: pd.Series, window: int = 20) -> pd.Series:
    """Linearly weighted moving average: the newest bar weighs most."""
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index (Wilder's smoothing), 0 to 100."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD. Columns: macd, signal, hist."""
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14,
               d_window: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """Stochastic oscillator. Columns: k, d."""
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    k = (100 * (close - lowest) / (highest - lowest)).rolling(smooth_k).mean()
    return pd.DataFrame({"k": k, "d": k.rolling(d_window).mean()})


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands. Columns: middle, upper, lower, bandwidth."""
    middle = series.rolling(window).mean()
    deviation = series.rolling(window).std()
    upper, lower = middle + num_std * deviation, middle - num_std * deviation
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower,
                         "bandwidth": (upper - lower) / middle})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range (Wilder's smoothing)."""
    previous = close.shift(1)
    true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()],
                           axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Donchian channel: highest high and lowest low of the window. Columns: upper, lower, middle."""
    upper, lower = high.rolling(window).max(), low.rolling(window).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": (upper + lower) / 2})


# -------------------------------------------------------------------- lines ---

class Line:
    """
    One indicator output, read relative to the current bar.

        line[-1]     value at the last completed bar
        line[-2]     the bar before that
        line[-5:]    the last five completed bars, as a numpy array
        len(line)    how many bars have completed
        line.ready   True once line[-1] has a value
        line.values  the whole series (for plotting or analysis after the run)

    Out-of-range indexes give NaN instead of raising, so a strategy can read
    `line[-2]` on its first bars without guarding.
    """

    def __init__(self, values, cursor, name: str | None = None):
        self._series = pd.Series(np.asarray(values, dtype=float), name=name)
        self._array = self._series.to_numpy()
        self._cursor = cursor
        self.name = name

    @property
    def values(self) -> pd.Series:
        return self._series

    @property
    def ready(self) -> bool:
        return not np.isnan(self[-1])

    def __len__(self) -> int:
        return self._cursor.i

    def __getitem__(self, key):
        completed = self._array[:self._cursor.i]
        if isinstance(key, slice):
            return completed[key]
        try:
            return float(completed[key])
        except IndexError:
            return float("nan")

    def __repr__(self) -> str:
        return f"Line({self.name or 'values'}, last={self[-1]})"


class Lines:
    """
    Several named outputs of one indicator. Each is a `Line`:

        bands = self.I(bollinger, close, 20)
        bands.upper[-1], bands.lower[-1]
        bands["upper"][-1]            # the same
    """

    def __init__(self, frame: pd.DataFrame, cursor):
        self._lines = {str(name): Line(frame[name], cursor, str(name)) for name in frame.columns}
        for name, line in self._lines.items():
            setattr(self, name, line)

    @property
    def values(self) -> pd.DataFrame:
        return pd.DataFrame({name: line.values for name, line in self._lines.items()})

    def __getitem__(self, name: str) -> Line:
        return self._lines[name]

    def __iter__(self):
        return iter(self._lines.values())

    def __repr__(self) -> str:
        return f"Lines({', '.join(self._lines)})"


def _last_two(x) -> tuple[float, float]:
    """(previous, latest) for a Line, or the same number twice for a constant."""
    if isinstance(x, Line):
        return x[-2], x[-1]
    return float(x), float(x)


def crossover(a, b) -> bool:
    """True if `a` moved from at-or-below `b` to above it on the last completed bar."""
    a_prev, a_now = _last_two(a)
    b_prev, b_now = _last_two(b)
    return bool(a_prev <= b_prev and a_now > b_now)   # any NaN makes this False


def crossunder(a, b) -> bool:
    """True if `a` moved from at-or-above `b` to below it on the last completed bar."""
    a_prev, a_now = _last_two(a)
    b_prev, b_now = _last_two(b)
    return bool(a_prev >= b_prev and a_now < b_now)


# =========================================================================
# Deprecated class-based indicators
# =========================================================================
#
# The classes below are the pre-0.6 API, used with Strategy.indicator() and
# read with an absolute bar index (`indicator[i]`). Each shifts its output by
# one bar so that the value at bar i only uses bars before i. They still work
# and will be removed in 1.0. Use the functions above with Strategy.I().

class Indicator(ABC):
    """
    Base class for indicators. Subclasses take pandas Series as inputs and
    write their result to ``self._values`` inside ``compute()``.

    Single-output indicators set ``self._values`` to a Series.
    Multi-output indicators set it to a DataFrame; each column is exposed
    as an attribute (e.g. ``macd.signal``).
    """

    def __init__(self):
        self._values: pd.Series | pd.DataFrame | None = None
        self.compute()
        if isinstance(self._values, pd.DataFrame):
            for col in self._values.columns:
                setattr(self, col, self._values[col])

    @property
    def values(self) -> pd.Series | pd.DataFrame | None:
        return self._values

    @abstractmethod
    def compute(self):
        pass

    def __getitem__(self, i: int):
        if self._values is None:
            raise ValueError(f"{self.__class__.__name__} has not computed yet")
        return self._values.iloc[i]

    def ready(self, i: int) -> bool:
        """True if the indicator has a non-NaN value at index ``i``."""
        if self._values is None or i < 0 or i >= len(self._values):
            return False
        row = self._values.iloc[i]
        if isinstance(row, pd.Series):
            return not row.isna().any()
        return not pd.isna(row)


class SMA(Indicator):
    def __init__(self, series: pd.Series, window: int = 20):
        self.series, self.window = series, window
        super().__init__()

    def compute(self):
        self._values = sma(self.series, self.window).shift(1)


class EMA(Indicator):
    def __init__(self, series: pd.Series, span: int = 20):
        self.series, self.span = series, span
        super().__init__()

    def compute(self):
        self._values = ema(self.series, self.span).shift(1)


class WMA(Indicator):
    def __init__(self, series: pd.Series, window: int = 20):
        self.series, self.window = series, window
        super().__init__()

    def compute(self):
        self._values = wma(self.series, self.window).shift(1)


class RSI(Indicator):
    def __init__(self, series: pd.Series, window: int = 14):
        self.series, self.window = series, window
        super().__init__()

    def compute(self):
        self._values = rsi(self.series, self.window).shift(1)


class MACD(Indicator):
    def __init__(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        self.series, self.fast, self.slow, self.signal = series, fast, slow, signal
        super().__init__()

    def compute(self):
        self._values = macd(self.series, self.fast, self.slow, self.signal).shift(1)


class Stochastic(Indicator):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14,
                 d_window: int = 3, smooth_k: int = 3):
        self.high, self.low, self.close = high, low, close
        self.k_window, self.d_window, self.smooth_k = k_window, d_window, smooth_k
        super().__init__()

    def compute(self):
        self._values = stochastic(self.high, self.low, self.close, self.k_window,
                                  self.d_window, self.smooth_k).shift(1)


class BollingerBands(Indicator):
    def __init__(self, series: pd.Series, window: int = 20, num_std: float = 2.0):
        self.series, self.window, self.num_std = series, window, num_std
        super().__init__()

    def compute(self):
        self._values = bollinger(self.series, self.window, self.num_std).shift(1)


class ATR(Indicator):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        self.high, self.low, self.close, self.window = high, low, close, window
        super().__init__()

    def compute(self):
        self._values = atr(self.high, self.low, self.close, self.window).shift(1)
