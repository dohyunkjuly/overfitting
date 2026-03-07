import pandas as pd
from abc import ABC, abstractmethod
from overfitting import Strategy

class Indicator(ABC):
    def __init__(self, strategy: Strategy, symbol: str):
        self.strategy: Strategy = strategy
        self.symbol: str = symbol
        self._values: pd.Series | pd.DataFrame | None = None
        self.compute()

    @property
    def values(self) -> pd.Series | pd.DataFrame | None:
        return self._values

    def get_series(self, name: str) -> pd.Series:
        d = self.strategy.broker._d(self.symbol)
        values = getattr(d, name, None)
        if values is None:
            raise AttributeError(f"Column '{name}' not found for symbol '{self.symbol}'")
        
        return pd.Series(values)

    @abstractmethod
    def compute(self):
        pass

    def value(self, i: int):
        if self._values is None:
            raise ValueError(f"{self.__class__.__name__} has not computed yet")
        return self._values.iloc[i]

    def __getitem__(self, i: int):
        return self.value(i)


class SMA(Indicator):
    def __init__(self, strategy: Strategy, symbol: str, source: str = "close", window: int = 20):
        self.source = source
        self.window = window
        super().__init__(strategy, symbol)

    def compute(self):
        s = self.get_series(self.source)
        self._values = s.rolling(self.window).mean().shift(1)


class EMA(Indicator):
    def __init__(self, strategy: Strategy, symbol: str, source: str = "close", span: int = 20):
        self.source = source
        self.span = span
        super().__init__(strategy, symbol)

    def compute(self):
        s = self.get_series(self.source)
        self._values = s.ewm(span=self.span, adjust=False).mean().shift(1)


class WMA(Indicator):
    def __init__(self, strategy: Strategy, symbol: str, source: str = "close", window: int = 20):
        self.source = source
        self.window = window
        super().__init__(strategy, symbol)

    def compute(self):
        s = self.get_series(self.source)
        weights = pd.Series(range(1, self.window + 1), dtype=float)

        def _wma(x):
            return (x * weights).sum() / weights.sum()

        self._values = s.rolling(self.window).apply(_wma, raw=False).shift(1)


class RSI(Indicator):
    def __init__(self, strategy: Strategy, symbol: str, source: str = "close", window: int = 14):
        self.source = source
        self.window = window
        super().__init__(strategy, symbol)

    def compute(self):
        s = self.get_series(self.source)
        delta = s.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.window, adjust=False, min_periods=self.window).mean()
        avg_loss = loss.ewm(alpha=1 / self.window, adjust=False, min_periods=self.window).mean()

        rs = avg_gain / avg_loss
        self._values = (100 - (100 / (1 + rs))).shift(1)


class MACD(Indicator):
    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        source: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        self.source = source
        self.fast = fast
        self.slow = slow
        self.signal = signal
        super().__init__(strategy, symbol)

    def compute(self):
        s = self.get_series(self.source)
        fast_ema = s.ewm(span=self.fast, adjust=False).mean()
        slow_ema = s.ewm(span=self.slow, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        hist = macd - signal

        self._values = pd.DataFrame({
            "macd": macd.shift(1),
            "signal": signal.shift(1),
            "hist": hist.shift(1),
        })


class Stochastic(Indicator):
    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        k_window: int = 14,
        d_window: int = 3,
        smooth_k: int = 3,
    ):
        self.k_window = k_window
        self.d_window = d_window
        self.smooth_k = smooth_k
        super().__init__(strategy, symbol)

    def compute(self):
        d = self.strategy.broker._d(self.symbol)
        high = d.high
        low = d.low
        close = d.close

        lowest_low = low.rolling(self.k_window).min()
        highest_high = high.rolling(self.k_window).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(self.smooth_k).mean()
        d_line = k.rolling(self.d_window).mean()

        self._values = pd.DataFrame({
            "%K": k.shift(1),
            "%D": d_line.shift(1),
        })


class BollingerBands(Indicator):
    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        source: str = "close",
        window: int = 20,
        num_std: float = 2.0,
    ):
        self.source = source
        self.window = window
        self.num_std = num_std
        super().__init__(strategy, symbol)

    def compute(self):
        s = self.get_series(self.source)
        mid = s.rolling(self.window).mean()
        std = s.rolling(self.window).std()

        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        self._values = pd.DataFrame({
            "middle": mid.shift(1),
            "upper": upper.shift(1),
            "lower": lower.shift(1),
            "bandwidth": ((upper - lower) / mid).shift(1),
        })


class ATR(Indicator):
    def __init__(self, strategy: Strategy, symbol: str, window: int = 14):
        self.window = window
        super().__init__(strategy, symbol)

    def compute(self):
        d = self.strategy.broker._d(self.symbol)
        prev_close = d.close.shift(1)

        tr1 = d.high - d.low
        tr2 = (d.high - prev_close).abs()
        tr3 = (d.low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / self.window, adjust=False, min_periods=self.window).mean()

        self._values = atr.shift(1)

