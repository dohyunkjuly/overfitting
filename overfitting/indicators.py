import pandas as pd
from abc import ABC, abstractmethod


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
        self.series = series
        self.window = window
        super().__init__()

    def compute(self):
        self._values = self.series.rolling(self.window).mean().shift(1)


class EMA(Indicator):
    def __init__(self, series: pd.Series, span: int = 20):
        self.series = series
        self.span = span
        super().__init__()

    def compute(self):
        self._values = self.series.ewm(span=self.span, adjust=False).mean().shift(1)


class WMA(Indicator):
    def __init__(self, series: pd.Series, window: int = 20):
        self.series = series
        self.window = window
        super().__init__()

    def compute(self):
        weights = pd.Series(range(1, self.window + 1), dtype=float)

        def _wma(x):
            return (x * weights).sum() / weights.sum()

        self._values = self.series.rolling(self.window).apply(_wma, raw=False).shift(1)


class RSI(Indicator):
    def __init__(self, series: pd.Series, window: int = 14):
        self.series = series
        self.window = window
        super().__init__()

    def compute(self):
        delta = self.series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.window, adjust=False, min_periods=self.window).mean()
        avg_loss = loss.ewm(alpha=1 / self.window, adjust=False, min_periods=self.window).mean()

        rs = avg_gain / avg_loss
        self._values = (100 - (100 / (1 + rs))).shift(1)


class MACD(Indicator):
    def __init__(
        self,
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        self.series = series
        self.fast = fast
        self.slow = slow
        self.signal = signal
        super().__init__()

    def compute(self):
        fast_ema = self.series.ewm(span=self.fast, adjust=False).mean()
        slow_ema = self.series.ewm(span=self.slow, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        hist = macd - signal

        self._values = pd.DataFrame({
            "macd":   macd.shift(1),
            "signal": signal.shift(1),
            "hist":   hist.shift(1),
        })


class Stochastic(Indicator):
    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
        smooth_k: int = 3,
    ):
        self.high = high
        self.low = low
        self.close = close
        self.k_window = k_window
        self.d_window = d_window
        self.smooth_k = smooth_k
        super().__init__()

    def compute(self):
        lowest_low = self.low.rolling(self.k_window).min()
        highest_high = self.high.rolling(self.k_window).max()

        raw_k = 100 * (self.close - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(self.smooth_k).mean()
        d_line = k.rolling(self.d_window).mean()

        self._values = pd.DataFrame({
            "k": k.shift(1),
            "d": d_line.shift(1),
        })


class BollingerBands(Indicator):
    def __init__(
        self,
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ):
        self.series = series
        self.window = window
        self.num_std = num_std
        super().__init__()

    def compute(self):
        mid = self.series.rolling(self.window).mean()
        std = self.series.rolling(self.window).std()

        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        self._values = pd.DataFrame({
            "middle":    mid.shift(1),
            "upper":     upper.shift(1),
            "lower":     lower.shift(1),
            "bandwidth": ((upper - lower) / mid).shift(1),
        })


class ATR(Indicator):
    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ):
        self.high = high
        self.low = low
        self.close = close
        self.window = window
        super().__init__()

    def compute(self):
        prev_close = self.close.shift(1)

        tr1 = self.high - self.low
        tr2 = (self.high - prev_close).abs()
        tr3 = (self.low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / self.window, adjust=False, min_periods=self.window).mean()

        self._values = atr.shift(1)
