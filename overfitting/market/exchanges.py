"""
Exchange connectors: download candles and funding rates from public APIs.

Every exchange is a small class describing how its API differs: the URL,
the symbol format, the request parameters and the response layout. The
shared work (paging through history, retries, rate limits, error messages)
lives in `Exchange`. Only the Python standard library is used.

Adding an exchange means writing one subclass and registering it in
EXCHANGES at the bottom of this file. Tests replace `_get_json` with a fake
server, so they never touch the network.

All times are milliseconds since the epoch, UTC. A request for [start, end)
returns the bars whose open time t satisfies start <= t < end.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator

from overfitting.errors import ExchangeError

MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# Timeframes in milliseconds. Each one divides a day, so bars line up with
# midnight UTC on every exchange.
TIMEFRAMES = {
    "1m": MINUTE, "3m": 3 * MINUTE, "5m": 5 * MINUTE, "15m": 15 * MINUTE,
    "30m": 30 * MINUTE, "1h": HOUR, "2h": 2 * HOUR, "4h": 4 * HOUR,
    "6h": 6 * HOUR, "12h": 12 * HOUR, "1d": DAY,
}

USER_AGENT = "overfitting-data/1"
RETRIES = 5


def _get_json(url: str, params: dict, timeout: float = 30.0):
    """GET `url` with query `params`; return (HTTP status, decoded JSON or None)."""
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, None


class Exchange:
    """
    Base class. A subclass sets the class attributes and implements the
    `*_params` and `parse_*` methods; everything else is shared.

    name             key used in load(exchange=...) and in the local store
    candles_url      endpoint for candles
    timeframes       our timeframe -> the exchange's name for it
    candle_limit     most candles one request may return
    funding_url      endpoint for funding rate history, None if unsupported
    funding_limit    most funding records one request may return
    pause            seconds to wait between requests (rate limit)
    """
    name = ""
    candles_url = ""
    timeframes: dict[str, str] = {}
    candle_limit = 500
    funding_url: str | None = None
    funding_limit = 100
    pause = 0.1

    def __init__(self):
        self._last_request = 0.0

    # ---------------------------------------------------- per exchange ---

    def symbol(self, base: str, quote: str) -> str:
        """The exchange's name for the linear perpetual on `base`/`quote`."""
        raise NotImplementedError

    def candle_params(self, symbol: str, timeframe: str, start: int, end: int) -> dict:
        """Query parameters for the candles in [start, end)."""
        raise NotImplementedError

    def parse_candles(self, payload) -> list[list]:
        """Rows [time, open, high, low, close, volume], volume in the base asset."""
        raise NotImplementedError

    def funding_params(self, symbol: str, start: int, end: int) -> dict:
        """Query parameters for the funding records in [start, end)."""
        raise NotImplementedError

    def parse_funding(self, payload) -> list[list]:
        """Rows [time, rate]."""
        raise NotImplementedError

    def error_message(self, payload) -> str | None:
        """The error in a response body, or None if the body is a success."""
        return None

    def throttled(self, payload) -> bool:
        """True if the body says to slow down (some exchanges do this with HTTP 200)."""
        return False

    # ---------------------------------------------------------- shared ---

    def fetch_candles(self, base: str, quote: str, timeframe: str,
                      start: int, end: int) -> Iterator[tuple[int, int, list[list]]]:
        """
        Yield (window start, window end, rows) covering [start, end), oldest first.
        Each request asks for one window of `candle_limit` bars, so paging
        does not depend on the order or edges of the exchange's response.
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"{self.name} has no {timeframe!r} candles. "
                             f"Available: {', '.join(self.timeframes)}.")
        symbol = self.symbol(base, quote)
        step = TIMEFRAMES[timeframe] * self.candle_limit
        for window_start in range(start, end, step):
            window_end = min(window_start + step, end)
            payload = self._request(self.candles_url,
                                    self.candle_params(symbol, timeframe, window_start, window_end))
            rows = [r for r in self.parse_candles(payload) if window_start <= r[0] < window_end]
            rows.sort(key=lambda r: r[0])
            yield window_start, window_end, rows

    def fetch_funding(self, base: str, quote: str,
                      start: int, end: int) -> Iterator[tuple[int, int, list[list]]]:
        """Yield (window start, window end, rows) covering [start, end), oldest first."""
        if self.funding_url is None:
            raise ValueError(f"{self.name} does not provide funding rate history.")
        symbol = self.symbol(base, quote)
        # Funding is paid at most once an hour, so a window of `funding_limit`
        # hours never holds more records than one request returns.
        step = HOUR * self.funding_limit
        for window_start in range(start, end, step):
            window_end = min(window_start + step, end)
            payload = self._request(self.funding_url,
                                    self.funding_params(symbol, window_start, window_end))
            rows = [r for r in self.parse_funding(payload) if window_start <= r[0] < window_end]
            rows.sort(key=lambda r: r[0])
            yield window_start, window_end, rows

    def _request(self, url: str, params: dict):
        """One GET with rate limiting, retries on throttling and server errors."""
        delay = 1.0
        for attempt in range(RETRIES):
            wait = self.pause - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                status, payload = _get_json(url, params)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                if attempt == RETRIES - 1:
                    raise ExchangeError(f"{self.name}: could not reach {url} ({error}).") from error
                time.sleep(delay)
                delay *= 2
                continue

            slow_down = status in (418, 429) or (payload is not None and self.throttled(payload))
            if slow_down or status >= 500:                  # throttled or server trouble
                if attempt == RETRIES - 1:
                    raise ExchangeError(f"{self.name}: HTTP {status} after {RETRIES} tries.")
                time.sleep(delay)
                delay *= 2
                continue

            message = self.error_message(payload) if payload is not None else "empty response"
            if status != 200 or message:
                raise ExchangeError(f"{self.name}: {message or f'HTTP {status}'} "
                                    f"(request {url} {params}).")
            return payload
        raise ExchangeError(f"{self.name}: request failed.")      # not reached


# ======================================================================
# Exchanges
# ======================================================================

class Binance(Exchange):
    """Binance USD-M futures. https://developers.binance.com/docs/derivatives"""
    name = "binance"
    candles_url = "https://fapi.binance.com/fapi/v1/klines"
    timeframes = {tf: tf for tf in TIMEFRAMES}
    candle_limit = 1500
    funding_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    funding_limit = 1000
    pause = 0.1

    def symbol(self, base, quote):
        return f"{base}{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # endTime is inclusive, so stop one millisecond before `end`.
        return {"symbol": symbol, "interval": self.timeframes[timeframe],
                "startTime": start, "endTime": end - 1, "limit": self.candle_limit}

    def parse_candles(self, payload):
        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                for r in payload]

    def funding_params(self, symbol, start, end):
        return {"symbol": symbol, "startTime": start, "endTime": end - 1,
                "limit": self.funding_limit}

    def parse_funding(self, payload):
        return [[int(r["fundingTime"]), float(r["fundingRate"])] for r in payload]

    def error_message(self, payload):
        if isinstance(payload, dict) and "code" in payload:
            return payload.get("msg") or f"error {payload['code']}"
        return None


class Bybit(Exchange):
    """Bybit linear perpetuals. https://bybit-exchange.github.io/docs/v5/intro"""
    name = "bybit"
    candles_url = "https://api.bybit.com/v5/market/kline"
    timeframes = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
                  "2h": "120", "4h": "240", "6h": "360", "12h": "720", "1d": "D"}
    candle_limit = 1000
    funding_url = "https://api.bybit.com/v5/market/funding/history"
    funding_limit = 200
    pause = 0.2

    def symbol(self, base, quote):
        return f"{base}{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # start and end are both inclusive.
        return {"category": "linear", "symbol": symbol, "interval": self.timeframes[timeframe],
                "start": start, "end": end - 1, "limit": self.candle_limit}

    def parse_candles(self, payload):
        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                for r in payload["result"]["list"]]

    def funding_params(self, symbol, start, end):
        # Bybit rejects a startTime without an endTime.
        return {"category": "linear", "symbol": symbol, "startTime": start, "endTime": end - 1,
                "limit": self.funding_limit}

    def parse_funding(self, payload):
        return [[int(r["fundingRateTimestamp"]), float(r["fundingRate"])]
                for r in payload["result"]["list"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("retCode", 0) != 0:
            return payload.get("retMsg") or f"error {payload['retCode']}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("retCode") == 10006


class OKX(Exchange):
    """OKX USDT-margined swaps. https://www.okx.com/docs-v5/en/"""
    name = "okx"
    candles_url = "https://www.okx.com/api/v5/market/history-candles"
    # The plain 6H, 12H and 1D bars follow Hong Kong time; the "utc" ones start at
    # midnight UTC like every other exchange.
    timeframes = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H",
                  "2h": "2H", "4h": "4H", "6h": "6Hutc", "12h": "12Hutc", "1d": "1Dutc"}
    candle_limit = 100
    funding_url = "https://www.okx.com/api/v5/public/funding-rate-history"
    funding_limit = 100
    pause = 0.12        # 20 requests per 2 seconds

    def symbol(self, base, quote):
        return f"{base}-{quote}-SWAP"

    def candle_params(self, symbol, timeframe, start, end):
        # `before` and `after` are both exclusive: newer than `before`, older than `after`.
        return {"instId": symbol, "bar": self.timeframes[timeframe],
                "before": start - 1, "after": end, "limit": self.candle_limit}

    def parse_candles(self, payload):
        # Volume in the base asset is volCcy (index 6); index 5 counts contracts.
        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])]
                for r in payload["data"]]

    def funding_params(self, symbol, start, end):
        return {"instId": symbol, "before": start - 1, "after": end, "limit": self.funding_limit}

    def parse_funding(self, payload):
        return [[int(r["fundingTime"]), float(r["fundingRate"])] for r in payload["data"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("code", "0") != "0":
            return payload.get("msg") or f"error {payload['code']}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == "50011"


EXCHANGES: dict[str, type[Exchange]] = {cls.name: cls for cls in (Binance, Bybit, OKX)}


def get_exchange(name: str) -> Exchange:
    """A connector by name, e.g. "binance"."""
    try:
        return EXCHANGES[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown exchange {name!r}. Available: {', '.join(EXCHANGES)}.") from None
