import pandas as pd
import numpy as np
from types import SimpleNamespace
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_float_dtype
from overfitting.errors import InitializationError

REQUIRED_OHLC = ("open", "high", "low", "close")

class Data(dict):
    """
    data = Data({"BTCUSDT": df_btc, "ETHUSDT": df_eth})
    data["BTCUSDT"].close[i]   # close price of BTC at step i
    data.index[i]              # the timestamp at step i
    data.symbols               # ("BTCUSDT", "ETHUSDT")
    len(data)                  # how many bars there are
    """
    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not isinstance(frames, dict) or not frames:
            raise InitializationError("Pass a non-empty dict mapping symbol -> DataFrame.")

        bars = {symbol: self._bars_from_frame(df, symbol) for symbol, df in frames.items()}
        self._check_aligned_timelines(bars)

        super().__init__(bars)
        self.symbols = tuple(bars)
        self.index = next(iter(bars.values())).timestamp

    def __len__(self) -> int:
        return len(self.index)


    def _bars_from_frame(self, df: pd.DataFrame, symbol: str) -> SimpleNamespace:
        """Turn one symbol's DataFrame into a namespace of numpy arrays."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise InitializationError(f"'{symbol}': expected a non-empty DataFrame.")

        missing = [c for c in REQUIRED_OHLC if c not in df.columns]
        if missing:
            raise InitializationError(
                f"'{symbol}': missing required columns {missing}. Got: {list(df.columns)}"
            )

        timestamps = self._pull_timestamps(df, symbol)

        arrays = {col: df[col].to_numpy() for col in df.columns}
        arrays["timestamp"] = timestamps
        return SimpleNamespace(**arrays)


    def _pull_timestamps(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Find the timestamp series — either a column or the DatetimeIndex — and normalise it."""
        if "timestamp" in df.columns:
            ts = df["timestamp"]
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name == "timestamp":
            ts = pd.Series(df.index, name="timestamp")
        else:
            raise InitializationError(
                f"'{symbol}': add a 'timestamp' column or use a DatetimeIndex."
            )
        return self._to_datetime_ns(ts).to_numpy()


    def _to_datetime_ns(self, ts: pd.Series) -> pd.Series:
        """Coerce whatever the user gave us into tz-naive datetime64[ns]."""
        if is_datetime64_any_dtype(ts):
            dt = pd.to_datetime(ts)
            try:
                return dt.tz_localize(None)
            except Exception:
                return dt

        if is_integer_dtype(ts) or is_float_dtype(ts):
            # Big numbers look like milliseconds; smaller ones look like seconds.
            unit = "ms" if pd.Series(ts).max() > 1e12 else "s"
            return pd.to_datetime(ts, unit=unit)

        return pd.to_datetime(ts, errors="raise")


    def _check_aligned_timelines(self, bars: dict[str, SimpleNamespace]) -> None:
        """All symbols must share the same timeline — for now."""
        # TODO: auto-fill / reindex so symbols with different timelines can co-exist.
        iterator = iter(bars.items())
        _, first = next(iterator)
        for symbol, b in iterator:
            if len(b.timestamp) != len(first.timestamp) or not np.array_equal(b.timestamp, first.timestamp):
                raise InitializationError(
                    f"'{symbol}' timeline doesn't match the other symbols."
                )
