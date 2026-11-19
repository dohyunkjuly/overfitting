"""
The data pipeline: exchange connectors, the Parquet store and load().

The network is replaced by fake servers that follow each exchange's real
API rules (inclusive or exclusive bounds, newest-first or oldest-first
pages, page size limits), so these tests run offline. Set
OVERFITTING_LIVE=1 to also run one real download per exchange.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import overfitting.market.exchanges as ex
import overfitting.market.loader as loader
from overfitting import Strategy
from overfitting.errors import ExchangeError
from overfitting.market import downloads, load, parse_symbol, stored
from overfitting.market.store import Store

HOUR = ex.HOUR
T0 = int(pd.Timestamp("2024-01-01").value // 1_000_000)
NOW = T0 + 30 * 24 * HOUR + 30 * 60_000          # 2024-01-31 00:30


def price(t):
    return 100.0 + (t // HOUR) % 50


class FakeServer:
    """Answers like the real APIs, from a deterministic price series."""

    def __init__(self, listed=None, missing=(), funding_jitter=8, fail_after=None):
        self.listed = dict(listed or {})        # base -> first bar time
        self.missing = set(missing)             # bar times the exchange lacks
        self.jitter = funding_jitter
        self.fail_after = fail_after
        self.calls = []

    # bars and funding that exist
    def _bars(self, base, step, lo, hi):
        first = self.listed.get(base, T0 - 400 * 24 * HOUR)
        t = max(lo, first)
        t = -(-t // step) * step
        out = []
        while t <= hi:
            if t not in self.missing:
                p = price(t) * (2 if base == "ETH" else 1)
                out.append((t, p, p + 1, p - 1, p + 0.5, 10.0))
            t += step
        return out

    def _funding(self, base, lo, hi):
        first = self.listed.get(base, T0 - 400 * 24 * HOUR)
        t = -(-max(lo - self.jitter, first) // (8 * HOUR)) * (8 * HOUR)
        out = []
        while t + self.jitter <= hi:
            out.append((t + self.jitter, 0.0001 * (1 + (t // (8 * HOUR)) % 3)))
            t += 8 * HOUR
        return [r for r in out if lo <= r[0] <= hi]

    def __call__(self, url, params, timeout=30.0, method="GET"):
        self.calls.append((url, dict(params)))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise ConnectionError("network down")
        if "binance" in url:
            return self._binance(url, params)
        if "bybit" in url:
            return self._bybit(url, params)
        return self._okx(url, params)

    def _binance(self, url, p):
        base = p["symbol"].removesuffix("USDT")
        if base == "NOPE":
            return 400, {"code": -1121, "msg": "Invalid symbol."}
        if "klines" in url:
            step = ex.TIMEFRAMES[p["interval"]]
            rows = self._bars(base, step, p["startTime"], p["endTime"])[: p["limit"]]
            return 200, [[t, str(o), str(h), str(lo), str(c), str(v), t + step - 1] for t, o, h, lo, c, v in rows]
        rows = self._funding(base, p["startTime"], p["endTime"])[: p["limit"]]
        return 200, [{"symbol": p["symbol"], "fundingTime": t, "fundingRate": str(r)} for t, r in rows]

    def _bybit(self, url, p):
        base = p["symbol"].removesuffix("USDT")
        if base == "NOPE":
            return 200, {"retCode": 10001, "retMsg": "params error: Symbol Is Invalid", "result": {}}
        if "kline" in url:
            step = {v: ex.TIMEFRAMES[k] for k, v in ex.Bybit.timeframes.items()}[p["interval"]]
            rows = self._bars(base, step, p["start"], p["end"])
            rows = rows[::-1][: p["limit"]]                      # newest first, newest kept
            return 200, {"retCode": 0, "result": {"list": [[str(x) for x in r] for r in rows]}}
        rows = self._funding(base, p["startTime"], p["endTime"])[::-1][: p["limit"]]
        return 200, {"retCode": 0, "result": {"list": [
            {"fundingRate": str(r), "fundingRateTimestamp": str(t)} for t, r in rows]}}

    def _okx(self, url, p):
        base = p["instId"].split("-")[0]
        if base == "NOPE":
            return 200, {"code": "51001", "msg": "Instrument ID doesn't exist.", "data": []}
        lo, hi = p["before"] + 1, p["after"] - 1                # both bounds exclusive
        if "candles" in url:
            step = {v: ex.TIMEFRAMES[k] for k, v in ex.OKX.timeframes.items()}[p["bar"]]
            rows = self._bars(base, step, lo, hi)[::-1][: p["limit"]]
            return 200, {"code": "0", "data": [
                [str(t), str(o), str(h), str(l_), str(c), "999", str(v), "0", "1"]
                for t, o, h, l_, c, v in rows]}
        rows = self._funding(base, lo, hi)[::-1][: p["limit"]]
        return 200, {"code": "0", "data": [{"fundingTime": str(t), "fundingRate": str(r)} for t, r in rows]}


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(ex, "_get_json", fake)
    monkeypatch.setattr(ex.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(loader.time, "time", lambda: NOW / 1000)
    return fake


@pytest.fixture
def store_dir(tmp_path):
    return tmp_path / "store"


def quiet_load(*args, **kwargs):
    kwargs.setdefault("progress", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load(*args, **kwargs)


# ------------------------------------------------------------------ symbols ---

@pytest.mark.parametrize("text, expected", [
    ("BTC", ("BTC", "USDT")), ("btc", ("BTC", "USDT")), ("BTCUSDT", ("BTC", "USDT")),
    ("BTC/USDT", ("BTC", "USDT")), ("BTC-USDT", ("BTC", "USDT")), ("BTC/USDT:USDT", ("BTC", "USDT")),
    ("BTC-USDT-SWAP", ("BTC", "USDT")), ("ETHUSDC", ("ETH", "USDC")), ("1000PEPE", ("1000PEPE", "USDT")),
])
def test_parse_symbol(text, expected):
    assert parse_symbol(text) == expected


def test_bare_symbol_takes_the_quote_argument():
    assert parse_symbol("SOL", quote="usdc") == ("SOL", "USDC")


# ---------------------------------------------------------- each exchange ---

@pytest.mark.parametrize("exchange", ["binance", "bybit", "okx"])
def test_load_returns_every_closed_bar(server, store_dir, exchange):
    data = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-11", exchange=exchange,
                      data_dir=store_dir, funding=True)

    frame = data["BTC"]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "funding_rate"]
    assert len(frame) == 240
    assert frame.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert frame.index[-1] == pd.Timestamp("2024-01-10 23:00")
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    expected = [price(int(t.value // 1_000_000)) for t in frame.index]
    assert np.allclose(frame["open"], expected)
    assert (frame["volume"] == 10.0).all()         # OKX: base volume, not contracts


@pytest.mark.parametrize("exchange", ["binance", "bybit", "okx"])
def test_pages_follow_the_exchange_limit(server, store_dir, exchange):
    quiet_load("BTC", "1m", start="2024-01-01", end="2024-01-03", exchange=exchange, data_dir=store_dir)

    limit = ex.EXCHANGES[exchange].candle_limit
    assert len(server.calls) == -(-2 * 24 * 60 // limit)
    assert all(params.get("limit") == limit for _, params in server.calls)


@pytest.mark.parametrize("exchange", ["binance", "bybit", "okx"])
def test_unknown_symbol_raises_the_exchange_message(server, store_dir, exchange):
    with pytest.raises(ExchangeError, match="(?i)invalid|doesn't exist"):
        quiet_load("NOPE", "1h", start="2024-01-01", exchange=exchange, data_dir=store_dir)


# --------------------------------------------------------------- caching ---

def test_second_load_makes_no_requests(server, store_dir):
    first = quiet_load(["BTC", "ETH"], "1h", start="2024-01-01", end="2024-01-20", data_dir=store_dir)
    calls = len(server.calls)
    second = quiet_load(["BTC", "ETH"], "1h", start="2024-01-01", end="2024-01-20", data_dir=store_dir)

    assert len(server.calls) == calls
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_a_narrower_range_is_served_from_the_store(server, store_dir):
    quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-20", data_dir=store_dir)
    calls = len(server.calls)
    frame = quiet_load("BTC", "1h", start="2024-01-05", end="2024-01-06", data_dir=store_dir)["BTC"]

    assert len(server.calls) == calls
    assert len(frame) == 24 and frame.index[0] == pd.Timestamp("2024-01-05")


def test_extending_the_range_downloads_only_the_new_part(server, store_dir):
    quiet_load("BTC", "1h", start="2024-01-10", end="2024-01-20", data_dir=store_dir)
    server.calls.clear()
    frame = quiet_load("BTC", "1h", start="2024-01-05", end="2024-01-25", data_dir=store_dir)["BTC"]

    old = (int(pd.Timestamp("2024-01-10").value // 1e6), int(pd.Timestamp("2024-01-20").value // 1e6))
    for _, params in server.calls:
        lo, hi = params["startTime"], params["endTime"]
        assert hi < old[0] or lo >= old[1]
    assert len(frame) == 20 * 24


def test_open_bar_is_never_returned_or_stored(server, store_dir):
    # NOW is 2024-01-31 00:30: the 00:00 bar is still open.
    frame = quiet_load("BTC", "1h", start="2024-01-30", data_dir=store_dir)["BTC"]

    assert frame.index[-1] == pd.Timestamp("2024-01-30 23:00")
    with Store(store_dir) as store:
        assert store.covered("binance", "candles:1h", "BTC/USDT")[-1][1] == NOW - 30 * 60_000


def test_bars_closed_since_the_last_load_are_fetched(server, store_dir, monkeypatch):
    quiet_load("BTC", "1h", start="2024-01-30", data_dir=store_dir)
    monkeypatch.setattr(loader.time, "time", lambda: (NOW + 3 * HOUR) / 1000)
    server.calls.clear()
    frame = quiet_load("BTC", "1h", start="2024-01-30", data_dir=store_dir)["BTC"]

    assert frame.index[-1] == pd.Timestamp("2024-01-31 02:00")
    assert server.calls and all(p["startTime"] >= NOW - 30 * 60_000 for _, p in server.calls)


def test_range_before_listing_is_not_asked_for_again(server, store_dir):
    server.listed["NEW"] = T0 + 5 * 24 * HOUR
    with pytest.warns(UserWarning, match="start at 2024-01-06 00:00, after the requested start"):
        frame = load("NEW", "1h", start="2024-01-01", end="2024-01-10", data_dir=store_dir,
                     progress=False)["NEW"]
    calls = len(server.calls)
    quiet_load("NEW", "1h", start="2024-01-01", end="2024-01-10", data_dir=store_dir)

    assert frame.index[0] == pd.Timestamp("2024-01-06")
    assert len(server.calls) == calls


def test_interrupted_download_resumes_where_it_stopped(server, store_dir):
    server.fail_after = 3
    with pytest.raises(ExchangeError, match="could not reach"):
        quiet_load("BTC", "1m", start="2024-01-01", end="2024-01-05", data_dir=store_dir)
    saved_windows = 3

    server.fail_after = None
    server.calls.clear()
    frame = quiet_load("BTC", "1m", start="2024-01-01", end="2024-01-05", data_dir=store_dir)["BTC"]

    assert len(frame) == 4 * 24 * 60
    assert len(server.calls) == -(-4 * 24 * 60 // 1500) - saved_windows


# ----------------------------------------------------------------- funding ---

def test_funding_lands_on_the_bar_it_is_paid_at(server, store_dir):
    frame = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-03", funding=True,
                       data_dir=store_dir)["BTC"]

    paid = frame[frame["funding_rate"] != 0]
    assert list(paid.index.hour) == [0, 8, 16] * 2      # payments stamped 8 ms late
    assert paid["funding_rate"].iloc[0] == pytest.approx(0.0001 * (1 + (T0 // (8 * HOUR)) % 3))


def test_daily_bars_add_up_the_days_funding(server, store_dir):
    hourly = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-05", funding=True,
                        data_dir=store_dir)["BTC"]
    daily = quiet_load("BTC", "1d", start="2024-01-01", end="2024-01-05", funding=True,
                       data_dir=store_dir)["BTC"]

    by_day = hourly["funding_rate"].resample("D").sum()
    assert np.allclose(daily["funding_rate"], by_day)


def test_funding_is_left_out_by_default(server, store_dir):
    frame = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-02", data_dir=store_dir)["BTC"]

    assert "funding_rate" not in frame
    assert not any("funding" in url.lower() for url, _ in server.calls)


def test_late_funding_history_warns(server, store_dir, monkeypatch):
    original = server._funding
    monkeypatch.setattr(server, "_funding",
                        lambda base, lo, hi: [r for r in original(base, lo, hi) if r[0] > T0 + 10 * 24 * HOUR])
    with pytest.warns(UserWarning, match="funding history .* starts at"):
        load("BTC", "1h", start="2024-01-01", end="2024-01-20", funding=True,
             data_dir=store_dir, progress=False)


# ---------------------------------------------------------- several symbols ---

def test_symbols_are_trimmed_to_the_bars_they_share(server, store_dir):
    server.listed["ETH"] = T0 + 2 * 24 * HOUR
    with pytest.warns(UserWarning) as caught:
        data = load(["BTC", "ETH"], "1h", start="2024-01-01", end="2024-01-11",
                    data_dir=store_dir, progress=False)

    messages = [str(w.message) for w in caught]
    assert any("ETH/USDT 1h candles start at 2024-01-03" in m for m in messages)
    assert any("Kept only the 192 bars" in m for m in messages)

    assert data["BTC"].index.equals(data["ETH"].index)
    assert data["BTC"].index[0] == pd.Timestamp("2024-01-03")


def test_keys_are_the_names_given(server, store_dir):
    data = quiet_load(["btc", "ETHUSDT"], "1h", start="2024-01-01", end="2024-01-02", data_dir=store_dir)
    assert list(data) == ["btc", "ETHUSDT"]
    assert data["ETHUSDT"]["open"].iloc[0] == 2 * data["btc"]["open"].iloc[0]


def test_missing_bars_warn(server, store_dir):
    server.missing = {T0 + 5 * HOUR, T0 + 6 * HOUR}
    with pytest.warns(UserWarning, match="2 bars are missing"):
        frame = load("BTC", "1h", start="2024-01-01", end="2024-01-02", data_dir=store_dir,
                     progress=False)["BTC"]
    assert len(frame) == 22


def test_loaded_data_runs_a_backtest_with_funding(server, store_dir):
    data = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-05", funding=True,
                      data_dir=store_dir)

    class Hold(Strategy):
        def init(self):
            self.set_leverage("BTC", 1)

        def next(self, i):
            if i == 0:
                self.market_order("BTC", 1)

    strategy = Hold(data, funding=True)
    strategy.run()
    assert len(strategy.fetch_funding()) == 4 * 3 - 1        # every payment after the entry bar


# ------------------------------------------------------------------ errors ---

@pytest.mark.parametrize("kwargs, message", [
    ({"timeframe": "7m"}, "Unknown timeframe"),
    ({"exchange": "nowhere"}, "Unknown exchange"),
    ({"start": None}, "Give a start date"),
    ({"start": "2024-01-02", "end": "2024-01-01"}, "No closed bars"),
    ({"start": "not a date"}, "cannot read"),
])
def test_bad_arguments(server, store_dir, kwargs, message):
    arguments = {"symbols": "BTC", "timeframe": "1h", "start": "2024-01-01", "data_dir": store_dir}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        quiet_load(**arguments)


def test_duplicate_symbols_are_rejected(server, store_dir):
    with pytest.raises(ValueError, match="twice"):
        quiet_load(["BTC", "BTC"], "1h", start="2024-01-01", data_dir=store_dir)


def test_throttling_is_retried(server, store_dir, monkeypatch):
    answers = iter([(429, None), (200, {"retCode": 10006, "retMsg": "Too many visits"})])

    def flaky(url, params, timeout=30.0, method="GET"):
        return next(answers, None) or server(url, params)

    monkeypatch.setattr(ex, "_get_json", flaky)
    frame = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-02", exchange="bybit", data_dir=store_dir)["BTC"]
    assert len(frame) == 24


def test_persistent_throttling_gives_up(server, store_dir, monkeypatch):
    monkeypatch.setattr(ex, "_get_json", lambda url, params, timeout=30.0, method="GET": (429, None))
    with pytest.raises(ExchangeError, match="429"):
        quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-02", data_dir=store_dir)


# ------------------------------------------------------------------- store ---

def test_missing_ranges(tmp_path):
    with Store(tmp_path) as store:
        store.mark_candles("x", "S", "1h", 10, 20)
        store.mark_candles("x", "S", "1h", 30, 40)
        assert store.missing("x", "candles:1h", "S", 0, 50) == [(0, 10), (20, 30), (40, 50)]
        assert store.missing("x", "candles:1h", "S", 12, 18) == []
        assert store.missing("x", "candles:1h", "S", 15, 35) == [(20, 30)]

        store.mark_candles("x", "S", "1h", 20, 30)        # fills the hole: ranges merge
        assert store.covered("x", "candles:1h", "S") == [(10, 40)]
        assert store.covered("x", "candles:5m", "S") == []


def test_store_survives_reopening_and_replaces_rows(tmp_path):
    t = T0
    with Store(tmp_path) as store:
        store.save_candles("x", "BTC/USDT", "1h", [[t, 1, 2, 0.5, 1.5, 10], [t + HOUR, 2, 3, 1, 2.5, 20]])
        store.mark_candles("x", "BTC/USDT", "1h", t, t + 2 * HOUR)
    with Store(tmp_path) as store:
        assert store.covered("x", "candles:1h", "BTC/USDT") == [(t, t + 2 * HOUR)]
        store.save_candles("x", "BTC/USDT", "1h", [[t + HOUR, 9, 9, 9, 9, 99]])      # a corrected bar
        store.mark_candles("x", "BTC/USDT", "1h", t + HOUR, t + 2 * HOUR)
    with Store(tmp_path) as store:
        rows = store.candles("x", "BTC/USDT", "1h", t, t + 2 * HOUR)
    assert rows[:, 0].tolist() == [t, t + HOUR]
    assert rows[1].tolist() == [t + HOUR, 9, 9, 9, 9, 99]


def test_files_are_monthly_below_one_hour_and_yearly_above(tmp_path):
    december, january = T0 - 24 * HOUR, T0                  # 2023-12-31 and 2024-01-01
    with Store(tmp_path) as store:
        for timeframe in ("1m", "1h"):
            store.save_candles("x", "BTC/USDT", timeframe, [[december, 1, 1, 1, 1, 1], [january, 1, 1, 1, 1, 1]])
        store.save_funding("x", "BTC/USDT", [[december, 0.0001], [january, 0.0002]])

    names = lambda *parts: sorted(p.name for p in tmp_path.joinpath("x", *parts).glob("*.parquet"))
    assert names("candles", "1m", "BTC-USDT") == ["2023-12.parquet", "2024-01.parquet"]
    assert names("candles", "1h", "BTC-USDT") == ["2023.parquet", "2024.parquet"]
    assert names("funding", "BTC-USDT") == ["2023.parquet", "2024.parquet"]
    with Store(tmp_path) as store:
        assert store.funding("x", "BTC/USDT", december, january + 1)[:, 1].tolist() == [0.0001, 0.0002]


def test_coverage_is_only_written_with_its_rows(tmp_path):
    """A range is never recorded as downloaded unless its rows reached disk."""
    store = Store(tmp_path)
    store.save_candles("x", "BTC/USDT", "1h", [[T0, 1, 1, 1, 1, 1]])
    store.mark_candles("x", "BTC/USDT", "1h", T0, T0 + HOUR)
    assert Store(tmp_path).covered("x", "candles:1h", "BTC/USDT") == []     # nothing flushed yet
    store.close()
    assert Store(tmp_path).covered("x", "candles:1h", "BTC/USDT") == [(T0, T0 + HOUR)]


# ------------------------------------------------- files changed by hand ---

def btc_folder(store_dir, timeframe="1h"):
    return store_dir / "binance" / "candles" / timeframe / "BTC-USDT"


def requested(server):
    """(start, end) of every candle request, as Timestamps."""
    return [(pd.Timestamp(p["startTime"], unit="ms"), pd.Timestamp(p["endTime"] + 1, unit="ms"))
            for url, p in server.calls if "klines" in url]


def test_deleted_file_is_downloaded_again(server, store_dir):
    quiet_load("BTC", "1h", start="2023-06-01", end="2024-01-20", data_dir=store_dir)
    (btc_folder(store_dir) / "2023.parquet").unlink()
    server.calls.clear()

    frame = quiet_load("BTC", "1h", start="2023-06-01", end="2024-01-20", data_dir=store_dir)["BTC"]

    assert len(frame) == (pd.Timestamp("2024-01-20") - pd.Timestamp("2023-06-01")) // pd.Timedelta("1h")
    assert requested(server)[0][0] == pd.Timestamp("2023-06-01")
    assert all(end <= pd.Timestamp("2024-01-01") for _, end in requested(server))   # 2024 untouched
    assert downloads(store_dir)["event"].tolist()[:2] == ["download", "missing"]


def test_downloads_shows_a_deletion_straight_away(server, store_dir):
    quiet_load("BTC", "1h", start="2023-06-01", end="2024-01-20", data_dir=store_dir)
    (btc_folder(store_dir) / "2023.parquet").unlink()

    log = downloads(store_dir)

    assert log["event"].iloc[0] == "missing"
    assert log["files"].iloc[0] == "2023.parquet"


def test_file_put_in_by_hand_is_used_without_downloading(server, store_dir, tmp_path):
    elsewhere = tmp_path / "other"
    quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-10", data_dir=elsewhere)
    btc_folder(store_dir).mkdir(parents=True)
    (btc_folder(elsewhere) / "2024.parquet").rename(btc_folder(store_dir) / "2024.parquet")
    server.calls.clear()

    frame = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-10", data_dir=store_dir)["BTC"]

    assert len(frame) == 9 * 24
    assert server.calls == []
    log = downloads(store_dir)
    assert log["event"].tolist() == ["import"]
    assert log["rows"].iloc[0] == 9 * 24


def test_hand_made_file_with_timestamps_and_any_order_is_accepted(server, store_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    times = pd.date_range("2024-01-01", periods=48, freq="h")[::-1]          # newest first
    table = pa.table({"time": pa.array(times.to_numpy(), pa.timestamp("ns")),
                      "open": [1.0] * 48, "high": [2.0] * 48, "low": [0.5] * 48,
                      "close": [1.5] * 48, "volume": [7.0] * 48})
    btc_folder(store_dir).mkdir(parents=True)
    pq.write_table(table, btc_folder(store_dir) / "2024.parquet")

    frame = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-03", data_dir=store_dir)["BTC"]

    assert server.calls == []
    assert frame.index.is_monotonic_increasing and len(frame) == 48
    assert (frame["volume"] == 7.0).all()


@pytest.mark.parametrize("name, rows, reason", [
    ("btc_2024.parquet", None, "name must be YYYY.parquet"),
    ("2023.parquet", None, "outside the period"),
    ("2024.parquet", [[T0 + 1, 1, 2, 0.5, 1.5, 1]], "not on 1h bar boundaries"),
    ("2024.parquet", [[T0, 1, 0.9, 0.5, 1.5, 1]], "high below"),
])
def test_bad_hand_made_file_is_set_aside(server, store_dir, name, rows, reason):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = rows or [[T0, 1, 2, 0.5, 1.5, 1]]
    columns = ["time", "open", "high", "low", "close", "volume"]
    table = pa.table({c: [int(r[0]) if c == "time" else float(r[j]) for r in rows]
                      for j, c in enumerate(columns)})
    btc_folder(store_dir).mkdir(parents=True)
    pq.write_table(table, btc_folder(store_dir) / name)

    with pytest.warns(UserWarning, match=reason):
        frame = load("BTC", "1h", start="2024-01-01", end="2024-01-02", data_dir=store_dir, progress=False)["BTC"]

    assert (btc_folder(store_dir) / (name + ".invalid")).exists()
    assert len(frame) == 24 and np.allclose(frame["open"], [price(T0 + k * HOUR) for k in range(24)])
    assert "rejected" in downloads(store_dir)["event"].tolist()


def test_file_changed_by_hand_keeps_only_what_it_holds(server, store_dir):
    import pyarrow.parquet as pq

    quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-11", data_dir=store_dir)
    path = btc_folder(store_dir) / "2024.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 5 * 24), path)                # keep the first 5 days only
    server.calls.clear()

    frame = quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-11", data_dir=store_dir)["BTC"]

    assert len(frame) == 10 * 24
    assert requested(server)[0][0] == pd.Timestamp("2024-01-06")
    assert downloads(store_dir)["event"].tolist()[:2] == ["download", "changed"]


def test_empty_range_before_listing_survives_a_deleted_file(server, store_dir):
    server.listed["NEW"] = int(pd.Timestamp("2024-01-05").value // 1e6)
    quiet_load("NEW", "1h", start="2023-06-01", end="2024-01-10", data_dir=store_dir)
    (store_dir / "binance" / "candles" / "1h" / "NEW-USDT" / "2024.parquet").unlink()
    server.calls.clear()

    quiet_load("NEW", "1h", start="2023-06-01", end="2024-01-10", data_dir=store_dir)

    assert requested(server) and all(start >= pd.Timestamp("2024-01-01") for start, _ in requested(server))


def test_download_log_and_inventory(server, store_dir):
    quiet_load(["BTC", "ETH"], "1h", start="2024-01-01", end="2024-01-03", funding=True,
               data_dir=store_dir)

    log = downloads(store_dir)
    assert set(log["event"]) == {"download"}
    assert len(log) == 4                                        # candles and funding, two symbols
    candles = log[log["dataset"] == "candles:1h"]
    assert (candles["rows"] == 48).all() and (candles["files"] == "2024.parquet").all()
    assert (log["start"] == pd.Timestamp("2024-01-01")).all()
    assert log["version"].notna().all()

    table = stored(store_dir)
    assert len(table) == 4 and (table["files"] == 1).all() and (table["size_mb"] > 0).all()
    assert table.loc[table["dataset"] == "candles:1h", "rows"].tolist() == [48, 48]


def test_data_dir_from_environment(server, tmp_path, monkeypatch):
    monkeypatch.setenv("OVERFITTING_DATA_DIR", str(tmp_path / "env"))
    quiet_load("BTC", "1h", start="2024-01-01", end="2024-01-02", funding=True)

    assert (tmp_path / "env" / "binance" / "candles" / "1h" / "BTC-USDT" / "2024.parquet").exists()
    table = stored()
    assert set(table["dataset"]) == {"candles:1h", "funding"}
    assert (table["symbol"] == "BTC/USDT").all()


# ------------------------------------------- recorded exchange responses ---
# tests/data/exchanges/<name>_candles.json and <name>_funding.json hold real
# responses (trimmed to a few rows) saved when each connector was written.

RESPONSES = Path(__file__).parent / "data" / "exchanges"


def recorded(kind):
    return sorted(p.name.removesuffix(f"_{kind}.json") for p in RESPONSES.glob(f"*_{kind}.json"))


def offline(name):
    """The connector, answering its contract-size lookup from the saved response."""
    exchange = ex.EXCHANGES[name]()
    contract = RESPONSES / f"{name}_contract.json"
    if contract.exists():
        exchange.current = exchange.symbol("BTC", exchange.default_quote)
        exchange._request = lambda url, params: json.loads(contract.read_text())
    return exchange


@pytest.mark.parametrize("name", recorded("candles"))
def test_recorded_candles_parse(name):
    exchange = offline(name)
    rows = exchange.parse_candles(json.loads((RESPONSES / f"{name}_candles.json").read_text()))

    assert rows and all(len(row) == 6 for row in rows)
    for t, o, h, lo, c, v in rows:
        assert isinstance(t, int) and t > 1_500_000_000_000          # milliseconds, not seconds
        assert all(isinstance(x, float) for x in (o, h, lo, c, v))
        assert lo <= min(o, c) <= max(o, c) <= h and v >= 0
        assert 1_000 < c < 1_000_000                                 # a BTC price, not scaled


@pytest.mark.parametrize("name", recorded("funding"))
def test_recorded_funding_parse(name):
    exchange = offline(name)
    rows = exchange.parse_funding(json.loads((RESPONSES / f"{name}_funding.json").read_text()))

    assert rows and all(len(row) == 2 for row in rows)
    for t, rate in rows:
        assert isinstance(t, int) and t > 1_500_000_000_000
        assert isinstance(rate, float) and abs(rate) < 0.05          # per payment, not annualized


# -------------------------------------------------------------------- live ---

@pytest.mark.skipif(not os.environ.get("OVERFITTING_LIVE"), reason="set OVERFITTING_LIVE=1 to hit real exchanges")
@pytest.mark.parametrize("exchange", sorted(ex.EXCHANGES))
def test_live_download(tmp_path, exchange):
    """The last two full days of BTC hourly bars, checked against Binance."""
    end = pd.Timestamp.now(tz="UTC").floor("D").tz_localize(None)
    start = end - pd.Timedelta(days=2)
    frame = quiet_load("BTC", "1h", start=start, end=end, exchange=exchange, funding=True,
                       data_dir=tmp_path)["BTC"]
    binance = quiet_load("BTC", "1h", start=start, end=end, exchange="binance", data_dir=tmp_path)["BTC"]

    assert len(frame) == 48
    assert (frame["close"] / binance["close"] - 1).abs().max() < 0.01
