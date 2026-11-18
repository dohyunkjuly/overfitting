"""Input validation and timestamp handling in Data."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from overfitting.data import Data
from overfitting.errors import InitializationError

BARS = [(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 101, 102)]


def test_exposes_columns_as_arrays():
    data = Data({"BTC": make_frame(BARS)})

    assert len(data) == 3
    assert data.symbols == ("BTC",)
    assert list(data["BTC"].close) == [100, 101, 102]
    assert data.index[0] == np.datetime64("2024-01-01T00:00")


def test_extra_columns_are_kept():
    frame = make_frame(BARS)
    frame["funding"] = [0.0001, 0.0002, 0.0003]
    data = Data({"BTC": frame})

    assert list(data["BTC"].funding) == [0.0001, 0.0002, 0.0003]


@pytest.mark.parametrize("timestamps", [
    [1704067200000, 1704070800000, 1704074400000],             # milliseconds
    [1704067200, 1704070800, 1704074400],                      # seconds
    ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
])
def test_timestamp_column_formats(timestamps):
    frame = make_frame(BARS).reset_index(drop=True)
    frame["timestamp"] = timestamps
    data = Data({"BTC": frame})

    assert pd.Timestamp(data.index[0]) == pd.Timestamp("2024-01-01 00:00")
    assert pd.Timestamp(data.index[2]) == pd.Timestamp("2024-01-01 02:00")


@pytest.mark.xfail(strict=True, reason="Data._to_datetime_ns calls Series.tz_localize, which fails and leaves the timezone on")
def test_timezone_is_dropped():
    frame = make_frame(BARS)
    frame.index = frame.index.tz_localize("UTC")
    data = Data({"BTC": frame})

    assert pd.Timestamp(data.index[0]) == pd.Timestamp("2024-01-01 00:00")


def test_multiple_symbols_share_one_timeline():
    data = Data({"BTC": make_frame(BARS), "ETH": make_frame(BARS)})

    assert data.symbols == ("BTC", "ETH")
    assert len(data) == 3


@pytest.mark.parametrize("bad", [None, {}, [make_frame(BARS)]])
def test_rejects_anything_but_a_non_empty_dict(bad):
    with pytest.raises(InitializationError):
        Data(bad)


def test_rejects_empty_frame():
    with pytest.raises(InitializationError):
        Data({"BTC": make_frame(BARS).iloc[0:0]})


def test_rejects_missing_ohlc_column():
    with pytest.raises(InitializationError):
        Data({"BTC": make_frame(BARS).drop(columns=["low"])})


def test_rejects_frame_without_timestamps():
    with pytest.raises(InitializationError):
        Data({"BTC": make_frame(BARS).reset_index(drop=True)})


def test_rejects_misaligned_timelines():
    other = make_frame(BARS, start="2024-02-01")
    with pytest.raises(InitializationError):
        Data({"BTC": make_frame(BARS), "ETH": other})


def test_rejects_different_lengths():
    with pytest.raises(InitializationError):
        Data({"BTC": make_frame(BARS), "ETH": make_frame(BARS[:2])})
