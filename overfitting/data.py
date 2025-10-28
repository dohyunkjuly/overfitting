import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_float_dtype
from overfitting.utils.error import InitializationError

REQUIRED_OHLC = ("open", "high", "low", "close")

class Data(dict):
    
    open: np.ndarray          # [Required] Open prices 
    high: np.ndarray          # [Required] High prices
    low: np.ndarray           # [Required] Low prices
    close: np.ndarray         # [Required] Close prices
    timestamp: np.ndarray     # Candle timestamps
    index: np.ndarray         # Fast datetime64[ns] index
    columns: tuple[str, ...]  # Available columns
    n: int                    # Number of rows

    """
    Usage: data.open[i], data.high[i], data.timestamp[i]
    """
    
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise InitializationError("Data must be a non-empty pandas DataFrame.")

        # Validate OHLC
        missing = [c for c in REQUIRED_OHLC if c not in df.columns]
        if missing:
            raise InitializationError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

        # Resolve timestamp: column or index
        if "timestamp" in df.columns:
            ts = df["timestamp"]
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name == "timestamp":
            # normalize to a Series so downstream is uniform
            ts = pd.Series(df.index, index=df.index, name="timestamp")
        else:
            raise InitializationError("Provide a 'timestamp' column or use a DatetimeIndex.")

        # Coerce timestamp to datetime64[ns]
        ts = self._to_datetime_ns(ts)

        payload = {c: df[c].to_numpy() for c in df.columns}
        payload["timestamp"] = ts.to_numpy()

        super().__init__(payload)
        object.__setattr__(self, "columns", tuple(sorted(payload.keys(), key=lambda x: (x!='timestamp', x))))
        object.__setattr__(self, "index", ts.to_numpy())
        object.__setattr__(self, "n", len(df))

    @staticmethod
    def _to_datetime_ns(s: pd.Series) -> pd.Series:
        if is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s) 
            # drop tz if present
            try:
                dt = dt.tz_localize(None)
            except Exception:
                pass
            return dt

        if is_integer_dtype(s) or is_float_dtype(s):
            # Heuristic: ms since epoch vs seconds
            mx = pd.Series(s).max()
            unit = "ms" if mx > 1e12 else "s"
            return pd.to_datetime(s, unit=unit)

        # Strings or mixed -> let pandas parse
        return pd.to_datetime(s, errors="raise")

    def __getattr__(self, key):
        # attribute-style access for columns
        try:
            return self[key]
        except KeyError:
            raise AttributeError( f"'Data' has no field '{key}'. Available: {', '.join(self.columns)}")

    def __len__(self):
        return self.n

    def __setattr__(self, key, value):
        # keep the container read-only (avoids accidental mistakes)
        if key in {"index", "columns", "n"}:
            object.__setattr__(self, key, value)
        else:
            raise AttributeError("Data is read-only; modify the source DataFrame before wrapping.")
    
class MultiCurrency(dict):

    symbols: tuple[str, ...]  # List of symbols in this container
    index: np.ndarray         # datetime64[ns] index
    n: int                    # Number of rows per symbol

    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not isinstance(frames, dict) or not frames:
            raise InitializationError("Data must be non-empty dict Type - dict[str, pd.DataFrame]")

        payload = {}
        first_ts = None
        for symbol, df in frames.items():
            d = Data(df)
            if first_ts is None:
                first_ts = d.index
            else:
                # Requires identical timestamps for NOW
                # TODO IMNPLEMENT AUTO FILL for uniform timestamps
                if len(d.index) != len(first_ts) or not np.array_equal(d.index, first_ts):
                    raise InitializationError(
                        f"Len Timestamps for {symbol} are not equal with the other symbols.")
                
            payload[symbol] = d

        super().__init__(payload)
        object.__setattr__(self, "symbols", tuple(payload.keys()))
        object.__setattr__(self, "index", first_ts)
        object.__setattr__(self, "n", len(first_ts))

        def __len__(self) -> int:
            return self.n