"""
load(): market data ready for a backtest, downloaded only when needed.

    from overfitting.market import load

    data = load(["BTC/USDT", "ETH/USDT"], "1h", start="2023-01-01")
    strategy = MyStrategy(data)

The first call downloads from the exchange and saves to the local store.
Later calls read from the store and only download what is missing, such as
bars that have closed since.
"""
from __future__ import annotations

import sys
import time
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from overfitting.errors import ExchangeError
from overfitting.market.exchanges import TIMEFRAMES, get_exchange
from overfitting.market.store import Store

QUOTES = ("USDT", "USDC", "USD")
# A bar counts as closed this long after its end, giving the exchange time to publish it.
SETTLE_MS = 30_000


def parse_symbol(text: str, default_quote: str = "USDT") -> tuple[str, str]:
    """
    (base, quote) from a symbol written BASE/QUOTE, such as "BTC/USDT", the way
    ccxt writes it. Case does not matter. Anything else raises ValueError with
    the BASE/QUOTE form to use, a bare name getting `default_quote`.
    """
    if isinstance(text, str):
        base, slash, quote = text.strip().upper().partition("/")
        if slash and base.isalnum() and quote.isalnum():
            return base, quote
    raise ValueError(f'Write symbols as BASE/QUOTE: "{_suggest(text, default_quote)}", '
                     f"not {text!r}.")


def _suggest(text, default_quote: str) -> str:
    """The BASE/QUOTE form someone probably meant by "BTC", "BTCUSDT" or "BTC-USDT-SWAP"."""
    s = str(text).strip().upper().split(":")[0]
    for suffix in ("-SWAP", "-PERP", "PERP"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
    s = s.strip("-_/")
    for separator in ("/", "-", "_"):
        base, found, quote = s.partition(separator)
        if found and base and quote:
            return f"{base}/{quote}"
    for quote in QUOTES:
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[: -len(quote)]}/{quote}"
    return f"{s or 'BTC'}/{default_quote}"


def _to_ms(value, name: str) -> int:
    """Milliseconds since the epoch from a date string, datetime or Timestamp (naive = UTC)."""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name}: cannot read {value!r} as a date.") from error
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return int(stamp.value // 1_000_000)


def _say(message: str, progress: bool) -> None:
    if progress:
        print(message, file=sys.stderr, flush=True)


def _day(ms: int) -> str:
    return f"{pd.Timestamp(ms, unit='ms'):%Y-%m-%d %H:%M}"


def _sync_candles(store, exchange, base, quote, timeframe, start, end, progress):
    key = f"{base}/{quote}"
    for gap_start, gap_end in store.missing(exchange.name, f"candles:{timeframe}", key, start, end):
        _say(f"Downloading {exchange.name} {key} {timeframe} candles "
             f"{_day(gap_start)} to {_day(gap_end)}", progress)
        count = 0
        for window_start, window_end, rows in exchange.fetch_candles(
                base, quote, timeframe, gap_start, gap_end):
            store.save_candles(exchange.name, key, timeframe, rows)
            store.mark_candles(exchange.name, key, timeframe, window_start, window_end)
            count += len(rows)
        _say(f"  {count:,} bars saved", progress)


def _sync_funding(store, exchange, base, quote, start, end, progress):
    key = f"{base}/{quote}"
    for gap_start, gap_end in store.missing(exchange.name, "funding", key, start, end):
        _say(f"Downloading {exchange.name} {key} funding rates "
             f"{_day(gap_start)} to {_day(gap_end)}", progress)
        count = 0
        for window_start, window_end, rows in exchange.fetch_funding(base, quote, gap_start, gap_end):
            store.save_funding(exchange.name, key, rows)
            store.mark_funding(exchange.name, key, window_start, window_end)
            count += len(rows)
        _say(f"  {count:,} funding payments saved", progress)


def _frame(candles: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(candles[:, 1:], columns=["open", "high", "low", "close", "volume"],
                         index=pd.to_datetime(candles[:, 0].astype(np.int64), unit="ms"))
    frame.index.name = "timestamp"
    return frame


def _funding_column(index: pd.DatetimeIndex, records: np.ndarray, step: int) -> np.ndarray:
    """
    Each payment on the bar whose open it falls in (exchanges stamp payments
    a few milliseconds late). Several payments in one bar, as on daily bars,
    are added up. Bars without a payment get 0.
    """
    column = pd.Series(0.0, index=index)
    if len(records):
        bars = pd.to_datetime((records[:, 0].astype(np.int64) // step) * step, unit="ms")
        rates = pd.Series(records[:, 1], index=bars).groupby(level=0).sum()
        rates = rates[rates.index.isin(index)]
        column.loc[rates.index] = rates.to_numpy()
    return column.to_numpy()


def load(symbols: str | Iterable[str], timeframe: str = "1h", start=None, end=None, *,
         exchange: str = "binance", funding: bool = False,
         data_dir=None, progress: bool = True) -> dict[str, pd.DataFrame]:
    """
    Candles for one or more perpetual futures, ready to pass to a Strategy.

    symbols     "BTC/USDT", or a list such as ["BTC/USDT", "ETH/USDT"]: BASE/QUOTE,
                as in ccxt. Every symbol is a linear perpetual.
    timeframe   one of 1m 3m 5m 15m 30m 1h 2h 4h 6h 12h 1d
    start, end  dates such as "2023-01-01" (UTC). `end` defaults to now; only
                bars that have closed are returned.
    exchange    a name from EXCHANGES, such as "binance", "bybit" or "okx"
    funding     also download funding rates and add a `funding_rate` column:
                the rate paid at each bar's open, 0 on bars without a payment
    data_dir    where the store lives; default ~/.overfitting/data, or the
                OVERFITTING_DATA_DIR environment variable
    progress    print what is being downloaded

    Returns {symbol: DataFrame} with columns open, high, low, close, volume
    (and funding_rate), indexed by bar open time, keyed by symbol ("BTC/USDT").
    With several symbols, only the bars all of them have are kept, so their
    timelines match.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {timeframe!r}. Use one of {', '.join(TIMEFRAMES)}.")
    if start is None:
        raise ValueError('Give a start date, e.g. load("BTC/USDT", "1h", start="2023-01-01").')
    connector = get_exchange(exchange)
    names = [symbols] if isinstance(symbols, str) else list(symbols)
    if not names:
        raise ValueError("Give at least one symbol.")
    pairs = [parse_symbol(name, connector.default_quote) for name in names]
    if len(set(pairs)) != len(pairs):
        raise ValueError("A symbol is listed twice.")

    step = TIMEFRAMES[timeframe]
    start_ms = -(-_to_ms(start, "start") // step) * step         # first bar opening at or after start
    last_closed = (int(time.time() * 1000) - SETTLE_MS) // step * step
    end_ms = last_closed if end is None else min(_to_ms(end, "end"), last_closed)
    if end_ms <= start_ms:
        raise ValueError("No closed bars between start and end.")

    frames = {}
    with Store(data_dir) as store:
        for base, q in pairs:
            key = f"{base}/{q}"
            try:
                _sync_candles(store, connector, base, q, timeframe, start_ms, end_ms, progress)
            except ExchangeError as error:
                hint = f'try "{base}/{connector.default_quote}"'
                if q == connector.default_quote or hint in str(error):
                    raise
                raise ExchangeError(f"{error} {connector.name} perpetuals are quoted in "
                                    f"{connector.default_quote}; {hint}.") from error
            candles = store.candles(connector.name, key, timeframe, start_ms, end_ms)
            if not len(candles):
                raise ValueError(f"{connector.name} has no {key} {timeframe} candles between "
                                 f"{_day(start_ms)} and {_day(end_ms)}.")
            frame = _frame(candles)

            if funding and connector.funding_url is None:
                frame["funding_rate"] = 0.0
                warnings.warn(f"{connector.name} does not provide funding rate history; "
                              "funding_rate is 0 throughout.", stacklevel=2)
            elif funding:
                _sync_funding(store, connector, base, q, start_ms, end_ms, progress)
                records = store.funding(connector.name, key, start_ms, end_ms)
                frame["funding_rate"] = _funding_column(frame.index, records, step)
                if not len(records):
                    warnings.warn(f"{connector.name} returned no funding rates for {key}; "
                                  "funding_rate is 0 throughout.", stacklevel=2)
                elif records[0, 0] - candles[0, 0] > 2 * 24 * 3_600_000:
                    warnings.warn(f"{connector.name} funding history for {key} starts at "
                                  f"{_day(int(records[0, 0]))}, after the first bar; earlier "
                                  "bars have funding_rate 0.", stacklevel=2)

            if candles[0, 0] - start_ms >= step:
                warnings.warn(f"{connector.name} {key} {timeframe} candles start at "
                              f"{_day(int(candles[0, 0]))}, after the requested start "
                              f"{_day(start_ms)} (listed later, or the exchange keeps "
                              "limited history).", stacklevel=2)

            expected = (candles[-1, 0] - candles[0, 0]) // step + 1
            if len(candles) < expected:
                warnings.warn(f"{key}: {int(expected - len(candles)):,} bars are missing between "
                              f"{_day(int(candles[0, 0]))} and {_day(int(candles[-1, 0]))} "
                              "(the exchange has no data for them).", stacklevel=2)
            frames[key] = frame

    if len(frames) > 1:
        common = None
        for frame in frames.values():
            common = frame.index if common is None else common.intersection(frame.index)
        if len(common) == 0:
            raise ValueError("The symbols have no bars in common.")
        trimmed = [name for name, frame in frames.items() if len(frame) != len(common)]
        if trimmed:
            warnings.warn(f"Kept only the {len(common):,} bars all symbols share, from "
                          f"{common[0]:%Y-%m-%d %H:%M}; {', '.join(trimmed)} had more.",
                          stacklevel=2)
        frames = {name: frame.loc[common] for name, frame in frames.items()}
    return frames


def stored(data_dir=None) -> pd.DataFrame:
    """
    What the local store holds: one row per exchange, dataset and symbol, with
    the date range downloaded, rows, number of files and size on disk.
    """
    with Store(data_dir) as store:
        return store.inventory()


def downloads(data_dir=None) -> pd.DataFrame:
    """
    The store's log, newest first. One row per event:

        download   rows fetched from an exchange and saved
        import     a Parquet file put in the store folder by hand, checked and used
        changed    a file changed outside the store; its contents were checked and kept
        missing    a file deleted by hand; its period will be downloaded again
        rejected   a file that failed the checks, renamed to *.invalid
    """
    with Store(data_dir) as store:
        return store.history()
