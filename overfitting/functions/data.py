import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_float_dtype
from overfitting.error import InitializationError

REQUIRED_OHLC = ("open", "high", "low", "close")

class Data(dict):
    """
    Wrap a DataFrame and expose columns as NumPy arrays with attribute access.
    - Requires open/high/low/close
    - 'timestamp' can be a column OR the DataFrame index
    - Ensures timestamp is datetime64[ns]
    Usage: data.open[i], data.high[i], data.timestamp[i]
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise InitializationError("Data must be a non-empty pandas DataFrame.")

        # Validate OHLC
        missing = [c for c in REQUIRED_OHLC if c not in df.columns]
        if missing:
            raise InitializationError(
                f"Missing required columns: {missing}. Available: {list(df.columns)}"
            )

        # Resolve timestamp: column or index
        if "timestamp" in df.columns:
            ts = df["timestamp"]
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name == "timestamp":
            # normalize to a Series so downstream is uniform
            ts = pd.Series(df.index, index=df.index, name="timestamp")
        else:
            raise InitializationError("Provide a 'timestamp' column or use a DatetimeIndex - name it 'timestamp'.")

        # Coerce timestamp to datetime64[ns]
        ts = self._to_datetime_ns(ts)

        # Build payload: ALL columns as numpy arrays
        payload = {c: df[c].to_numpy() for c in df.columns}
        # Ensure we have 'timestamp' as well (from column or index)
        payload["timestamp"] = ts.to_numpy()

        super().__init__(payload)

        # Meta (read-only)
        object.__setattr__(self, "columns", tuple(sorted(payload.keys(), key=lambda x: (x!='timestamp', x))))
        object.__setattr__(self, "index", ts.to_numpy())  # fast datetime64[ns] index
        object.__setattr__(self, "n", len(df))

    @staticmethod
    def _to_datetime_ns(s: pd.Series) -> pd.Series:
        if is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s)  # already datetime-like; ensure ns
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
            raise AttributeError(
                f"'Data' has no field '{key}'. Available: {', '.join(self.columns)}"
            ) from None

    def __len__(self):
        return self.n

    def __setattr__(self, key, value):
        # keep the container read-only (avoids accidental mistakes)
        if key in {"index", "columns", "n"}:
            object.__setattr__(self, key, value)
        else:
            raise AttributeError("Data is read-only; modify the source DataFrame before wrapping.")