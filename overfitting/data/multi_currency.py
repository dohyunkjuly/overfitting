import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_float_dtype
from overfitting.error import InitializationError
from overfitting.data import Data 

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