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
from datetime import datetime, timezone
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


def _get_json(url: str, params: dict, timeout: float = 30.0, method: str = "GET"):
    """
    GET `url` with query `params` (or POST `params` as a JSON body);
    return (HTTP status, decoded JSON or None).
    """
    if method == "POST":
        request = urllib.request.Request(
            url, data=json.dumps(params).encode(), method="POST",
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"})
    else:
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
    method           "GET" (parameters in the URL) or "POST" (parameters as a JSON body)
    default_quote    quote currency for bare names like "BTC" (USDT, or USDC/USD
                     on exchanges that settle in those)

    While a download runs, `self.current` holds the exchange's symbol, so
    parse_* can look up per-symbol details such as contract size.
    """
    name = ""
    candles_url = ""
    timeframes: dict[str, str] = {}
    candle_limit = 500
    funding_url: str | None = None
    funding_limit = 100
    pause = 0.1
    method = "GET"
    default_quote = "USDT"

    def __init__(self):
        self._last_request = 0.0
        self.current: str | None = None

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

    def no_data(self, payload) -> bool:
        """
        True if an error body only means there is no data for the period (for example
        Gate's "too long ago"). The body is then passed to parse_*, which returns no
        rows, and the period is recorded as having nothing to download.
        """
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
        symbol = self.current = self.symbol(base, quote)
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
        symbol = self.current = self.symbol(base, quote)
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
        """One request with rate limiting, retries on throttling and server errors."""
        delay = 1.0
        for attempt in range(RETRIES):
            wait = self.pause - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                status, payload = _get_json(url, params, method=self.method)
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

            if payload is not None and self.no_data(payload):
                return payload
            message = self.error_message(payload) if payload is not None else "empty response"
            if status != 200 or message:
                raise ExchangeError(f"{self.name}: {message or f'HTTP {status}'} "
                                    f"(request {url} {params}).")
            return payload
        raise ExchangeError(f"{self.name}: request failed.")      # not reached


# ======================================================================
# Exchanges, in alphabetical order:
#   binance, aster (Binance's API), bingx, bitfinex, bitget, blofin, bybit, cryptocom,
#   deribit, dydx, gate, htx, hyperliquid, kraken, kucoin, mexc, okx, phemex, woo
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


class Aster(Binance):
    """
    Aster perpetuals, a Binance-futures-compatible API. https://github.com/asterdex/api-docs

    Requests, responses and errors match Binance USD-M, so everything but the URLs is
    inherited. Volume is in the base asset. BTCUSDT history starts 2021-09-01 (inherited
    from APX/ApolloX). Funding is per interval (8h for BTC; some symbols use 4h or 1h).
    Like Binance, an unknown symbol's funding history is an empty list, not an error.
    """
    name = "aster"
    candles_url = "https://fapi.asterdex.com/fapi/v1/klines"
    funding_url = "https://fapi.asterdex.com/fapi/v1/fundingRate"
    pause = 0.25            # 2400 weight a minute; a 1500-bar klines request weighs 10

class BingX(Exchange):
    """
    BingX USDT-M perpetual swaps. https://bingx-api.github.io/docs/#/en-us/swapV2/market-api.html

    Volume is in the base asset. History depth differs by timeframe (BTC, checked
    2026-11: 1m from 2025-11-10, 30m from 2025-02-26, 1h from 2024-04-27, 2h and longer
    from 2022-10-10); older windows come back empty. 3m, 5m and 15m keep only a few
    weeks, so they are left out. A page holds at most 1000 bars although `limit`
    accepts 1440. The funding history endpoint is not used: the same request returns
    a full page, a partial page or nothing from one call to the next.
    """
    name = "bingx"
    candles_url = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"
    timeframes = {tf: tf for tf in ("1m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")}
    candle_limit = 1000
    funding_url = None
    pause = 0.5         # code 100410 ("try again later") was seen at 2-3 requests a second

    def symbol(self, base, quote):
        return f"{base}-{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # endTime is inclusive; results come newest first.
        return {"symbol": symbol, "interval": self.timeframes[timeframe],
                "startTime": start, "endTime": end - 1, "limit": self.candle_limit}

    def parse_candles(self, payload):
        return [[int(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["volume"])] for r in payload["data"] or []]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("code", 0) != 0:
            return payload.get("msg") or f"error {payload['code']}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == 100410


class Bitfinex(Exchange):
    """
    Bitfinex perpetual futures (tBTCF0:USTF0). https://docs.bitfinex.com/reference/rest-public-candles

    Rows are [time, open, CLOSE, high, low, volume]; volume is in the base asset. Minutes
    without trades have no bar, so 1m and 5m series have gaps on quiet pairs. History
    starts at the listing (BTC: July 2019). Some assets have short codes (USDT is UST,
    ALGO is ALG), looked up in the public currency map. An unknown pair answers with an
    empty list, so names are checked against the futures list. There is no funding
    endpoint: status/deriv/hist gives a snapshot about once a minute, and the first
    snapshot after each 8-hourly funding event carries the rate set at that event
    (CURRENT_FUNDING, with NEXT_FUNDING_EVT 8 hours later). Errors come as HTTP 500.
    """
    name = "bitfinex"
    candles_url = "https://api-pub.bitfinex.com/v2/candles/trade:{}:{}/hist"
    timeframes = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
                  "6h": "6h", "12h": "12h", "1d": "1D"}
    candle_limit = 10000
    funding_url = "https://api-pub.bitfinex.com/v2/status/deriv/{}/hist"
    funding_limit = 72          # hours: about 4300 snapshots, under the 5000 cap
    pause = 2.1                 # candles: 30 requests a minute (status: 90)
    _pairs = None

    def symbol(self, base, quote):
        if self._pairs is None:
            url = "https://api-pub.bitfinex.com/v2/conf/"
            self._pairs = set(self._request(url + "pub:list:pair:futures", {})[0])
            self._codes = self._request(url + "pub:map:currency:sym", {})[0]

        def codes(asset):      # Bitfinex's own codes for an asset, e.g. USDT -> UST
            return [asset] + [c.removesuffix("F0") for c, sym in self._codes if sym.upper() == asset]

        for pair in (f"{b}F0:{q}F0" for b in codes(base) for q in codes(quote)):
            if pair in self._pairs:
                return f"t{pair}"
        raise ExchangeError(f"bitfinex: no perpetual for {base}/{quote}.")

    def fetch_candles(self, base, quote, timeframe, start, end):
        # The timeframe and pair are part of the URL path.
        if timeframe in self.timeframes:
            self.candles_url = type(self).candles_url.format(self.timeframes[timeframe],
                                                             self.symbol(base, quote))
        return super().fetch_candles(base, quote, timeframe, start, end)

    def fetch_funding(self, base, quote, start, end):
        self.funding_url = type(self).funding_url.format(self.symbol(base, quote))
        return super().fetch_funding(base, quote, start, end)

    def candle_params(self, symbol, timeframe, start, end):
        # start and end are both inclusive.
        return {"start": start, "end": end - 1, "limit": self.candle_limit, "sort": 1}

    def parse_candles(self, payload):
        return [[int(r[0]), float(r[1]), float(r[3]), float(r[4]), float(r[2]), float(r[5])]
                for r in payload]

    def funding_params(self, symbol, start, end):
        return {"start": start, "end": end - 1, "limit": 5000, "sort": 1}

    def parse_funding(self, payload):
        # One row per event: the event 8 hours before the next one, and the rate set then.
        rates = {int(r[7]) - 8 * HOUR: float(r[11]) for r in payload
                 if r[7] is not None and r[11] is not None}
        return [[t, rate] for t, rate in rates.items()]

    def error_message(self, payload):
        if isinstance(payload, list) and payload[:1] == ["error"]:
            return f"{payload[2]} (code {payload[1]})"
        return None

    def throttled(self, payload):
        return isinstance(payload, list) and payload[:2] == ["error", 11010]


class Bitget(Exchange):
    """
    Bitget USDT-M futures. https://www.bitget.com/api-doc/contract/market/Get-History-Candle-Data
    Volume is the base-coin volume (index 5). Funding history is only served by page
    number (no time filter) and reaches back about three months, so it is not supported.
    """
    name = "bitget"
    candles_url = "https://api.bitget.com/api/v2/mix/market/history-candles"
    # The plain 6H, 12H and 1D bars follow UTC+8; the "utc" ones start at midnight UTC.
    # There is no 2h bar.
    timeframes = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H",
                  "4h": "4H", "6h": "6Hutc", "12h": "12Hutc", "1d": "1Dutc"}
    # A request may return 200 bars but span at most 90 days, so 90 daily bars.
    candle_limit = 90
    funding_url = None
    pause = 0.1         # 20 requests per second

    def symbol(self, base, quote):
        return f"{base}{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # endTime is exclusive; the newest `limit` bars before it are returned.
        return {"symbol": symbol, "productType": "usdt-futures",
                "granularity": self.timeframes[timeframe],
                "startTime": start, "endTime": end, "limit": self.candle_limit}

    def parse_candles(self, payload):
        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                for r in payload["data"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("code", "00000") != "00000":
            return payload.get("msg") or f"error {payload['code']}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == "429"


class BloFin(Exchange):
    """BloFin USDT perpetuals (OKX-style API). https://docs.blofin.com/index.html"""
    name = "blofin"
    candles_url = "https://openapi.blofin.com/api/v1/market/candles"
    # 6H, 12H and 1D bars start at midnight UTC.
    timeframes = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H",
                  "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D"}
    candle_limit = 1440
    funding_url = "https://openapi.blofin.com/api/v1/market/funding-rate-history"
    funding_limit = 100
    pause = 0.25        # 1500 requests per 5 minutes per IP, else banned for an hour

    def symbol(self, base, quote):
        return f"{base}-{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # `before` and `after` are both exclusive: newer than `before`, older than `after`.
        return {"instId": symbol, "bar": self.timeframes[timeframe],
                "before": start - 1, "after": end, "limit": self.candle_limit}

    def parse_candles(self, payload):
        # Volume in the base asset is volCurrency (index 6); index 5 counts contracts.
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
        return isinstance(payload, dict) and payload.get("code") == "429"


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


class CryptoCom(Exchange):
    """
    Crypto.com Exchange perpetuals (BTCUSD-PERP). https://exchange-docs.crypto.com/exchange/v1/rest-ws/index.html

    Perpetuals are quoted in USD. Contract size is 1, so `v` is in the base asset.
    A page holds at most 300 bars; when a range holds more, the newest 300 come back.
    No 3m bars. Funding is paid hourly; get-valuations funding_hist gives the rate per
    hour. Both endpoints treat end_ts as exclusive.
    """
    name = "cryptocom"
    candles_url = "https://api.crypto.com/exchange/v1/public/get-candlestick"
    timeframes = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
                  "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D"}
    candle_limit = 300
    funding_url = "https://api.crypto.com/exchange/v1/public/get-valuations"
    funding_limit = 300
    pause = 0.1         # public market data: 100 requests per second per method
    default_quote = "USD"

    def symbol(self, base, quote):
        return f"{base}{quote}-PERP"

    def candle_params(self, symbol, timeframe, start, end):
        return {"instrument_name": symbol, "timeframe": self.timeframes[timeframe],
                "count": self.candle_limit, "start_ts": start, "end_ts": end}

    def parse_candles(self, payload):
        return [[int(r["t"]), float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]),
                 float(r["v"])] for r in payload["result"]["data"]]

    def funding_params(self, symbol, start, end):
        return {"instrument_name": symbol, "valuation_type": "funding_hist",
                "count": self.funding_limit, "start_ts": start, "end_ts": end}

    def parse_funding(self, payload):
        return [[int(r["t"]), float(r["v"])] for r in payload["result"]["data"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("code", 0) != 0:
            return payload.get("message") or f"error {payload['code']}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == 42901


class Deribit(Exchange):
    """
    Deribit USDC-margined linear perpetuals (BTC_USDC-PERPETUAL).
    https://docs.deribit.com/#public-get_tradingview_chart_data

    Volume is the API's `volume`, already in the base asset (cost / volume is the price).
    Bars with no trades are still returned, with volume 0. Deribit's 4h does not exist and
    its 6h, 12h and 1d bars start at 08:00 UTC, so they are left out. History starts at
    the listing (BTC: March 2022) for every timeframe.

    Funding accrues continuously. get_funding_rate_history has one record per hour whose
    `interest_1h` is the funding accrued over the hour ending at its timestamp, i.e. the
    rate per hour (`interest_8h` is the sum of the last eight). That hourly rate is used.
    """
    name = "deribit"
    candles_url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    timeframes = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
                  "2h": "120"}
    candle_limit = 5000
    funding_url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
    funding_limit = 720     # it returns at most the 744 newest records of a range
    pause = 0.2             # 20 requests per second for unauthenticated users
    default_quote = "USDC"

    def symbol(self, base, quote):
        return f"{base}_{quote}-PERPETUAL"

    def candle_params(self, symbol, timeframe, start, end):
        # Both timestamps are inclusive.
        return {"instrument_name": symbol, "resolution": self.timeframes[timeframe],
                "start_timestamp": start, "end_timestamp": end - 1}

    def parse_candles(self, payload):
        r = payload["result"]
        return [[int(t), float(o), float(h), float(l), float(c), float(v)] for t, o, h, l, c, v
                in zip(r["ticks"], r["open"], r["high"], r["low"], r["close"], r["volume"])]

    def funding_params(self, symbol, start, end):
        # start_timestamp is exclusive, end_timestamp inclusive.
        return {"instrument_name": symbol, "start_timestamp": start - 1, "end_timestamp": end - 1}

    def parse_funding(self, payload):
        return [[int(r["timestamp"]), float(r["interest_1h"])] for r in payload["result"]]

    def error_message(self, payload):
        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            reason = (error.get("data") or {}).get("reason")
            return f"{error.get('message')}: {reason}" if reason else str(error.get("message"))
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and (payload.get("error") or {}).get("code") == 10028


def _iso(ms):
    return f"{datetime.fromtimestamp(ms // 1000, timezone.utc):%Y-%m-%dT%H:%M:%S}.{ms % 1000:03d}Z"


def _ms(text):
    return round(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


class DYDX(Exchange):
    """
    dYdX v4 perpetuals, indexer API. https://docs.dydx.xyz/indexer-client/http

    Markets are named BTC-USD (settled in USDC); history starts at the v4 launch in
    late October 2023. Volume is `baseTokenVolume`, in the base asset. No 3m, 2h, 6h or
    12h bars. Funding is paid hourly and `rate` is the rate per hour; the funding endpoint
    only takes an upper bound, so each window asks for the newest `limit` records before
    its end. Timestamps come in ISO strings; funding is stamped up to ~1 s after the hour.
    """
    name = "dydx"
    candles_url = "https://indexer.dydx.trade/v4/candles/perpetualMarkets/{}"
    timeframes = {"1m": "1MIN", "5m": "5MINS", "15m": "15MINS", "30m": "30MINS",
                  "1h": "1HOUR", "4h": "4HOURS", "1d": "1DAY"}
    candle_limit = 1000
    funding_url = "https://indexer.dydx.trade/v4/historicalFunding/{}"
    funding_limit = 1000
    pause = 0.2             # indexer: about 100 requests per 10 seconds
    default_quote = "USD"

    def symbol(self, base, quote):
        return f"{base}-{quote}"

    def fetch_candles(self, base, quote, timeframe, start, end):
        # The market is part of the URL path, not a parameter.
        self.candles_url = type(self).candles_url.format(self.symbol(base, quote))
        return super().fetch_candles(base, quote, timeframe, start, end)

    def fetch_funding(self, base, quote, start, end):
        self.funding_url = type(self).funding_url.format(self.symbol(base, quote))
        return super().fetch_funding(base, quote, start, end)

    def candle_params(self, symbol, timeframe, start, end):
        # fromISO and toISO are both inclusive; results come newest first.
        return {"resolution": self.timeframes[timeframe], "fromISO": _iso(start),
                "toISO": _iso(end - 1), "limit": self.candle_limit}

    def parse_candles(self, payload):
        return [[_ms(r["startedAt"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["baseTokenVolume"])] for r in payload["candles"]]

    def funding_params(self, symbol, start, end):
        return {"effectiveBeforeOrAt": _iso(end - 1), "limit": self.funding_limit}

    def parse_funding(self, payload):
        return [[_ms(r["effectiveAt"]), float(r["rate"])] for r in payload["historicalFunding"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("errors"):
            return "; ".join(e.get("msg", str(e)) for e in payload["errors"])
        return None


class Gate(Exchange):
    """
    Gate USDT-settled perpetuals. https://www.gate.com/docs/developers/apiv4/#futures
    Only the most recent 10,000 bars of each timeframe are served (1m: about a week),
    and funding only for the last 180 days. Older windows are answered with an error
    ("too long ago", "180-day limit") that is read as an empty page. Volume is contracts times the contract's
    quanto_multiplier.
    """
    name = "gate"
    candles_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
    timeframes = {tf: tf for tf in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                                    "6h", "12h", "1d")}
    candle_limit = 2000
    funding_url = "https://api.gateio.ws/api/v4/futures/usdt/funding_rate"
    funding_limit = 1000
    pause = 0.1         # 200 requests per 10 seconds

    def __init__(self):
        super().__init__()
        self._multipliers = {}

    def symbol(self, base, quote):
        return f"{base}_{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # Times are seconds. `to` also returns an unfinished bar at `end`, which the
        # window filter drops.
        return {"contract": symbol, "interval": self.timeframes[timeframe],
                "from": start // 1000, "to": end // 1000}

    def parse_candles(self, payload):
        size = self._multiplier(self.current)      # also how an unknown symbol is reported
        if isinstance(payload, dict):               # "too long ago": nothing served there
            return []
        return [[int(r["t"]) * 1000, float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]),
                 float(r["v"]) * size] for r in payload]

    def _multiplier(self, symbol):
        if symbol not in self._multipliers:
            info = self._request(f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}", {})
            self._multipliers[symbol] = float(info["quanto_multiplier"])
        return self._multipliers[symbol]

    def funding_params(self, symbol, start, end):
        return {"contract": symbol, "from": start // 1000, "to": end // 1000 - 1,
                "limit": self.funding_limit}

    def parse_funding(self, payload):
        if isinstance(payload, dict):               # older than 180 days: nothing served
            return []
        # Older records are stamped one second past the hour; round down to the minute.
        return [[int(r["t"]) // 60 * 60_000, float(r["r"])] for r in payload]

    def error_message(self, payload):
        if isinstance(payload, dict) and "label" in payload:
            return payload.get("message") or payload["label"]
        return None

    def no_data(self, payload):
        # Candles: "Candlestick too long ago"; funding: "from time exceeds 180-day limit".
        message = str(payload.get("message", "")) if isinstance(payload, dict) else ""
        return "too long ago" in message or "180-day limit" in message

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("label") == "TOO_MANY_REQUESTS"


class HTX(Exchange):
    """
    HTX (Huobi) USDT-margined swaps. https://www.htx.com/en-us/opend/newApiPages/?id=8cb81024-77b5-11ed-9966-0242ac110003
    Volume is `amount`, already in the base coin. The 1day bar starts at 16:00 UTC
    (UTC+8) and there are no 2h/6h/12h bars, so those are left out. Funding history
    is only served by page number, so it is not supported.
    """
    name = "htx"
    candles_url = "https://api.hbdm.com/linear-swap-ex/market/history/kline"
    timeframes = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min",
                  "4h": "4hour"}
    candle_limit = 2000
    funding_url = None
    pause = 0.1

    def symbol(self, base, quote):
        return f"{base}-{quote}"

    def candle_params(self, symbol, timeframe, start, end):
        # Seconds, both bounds inclusive.
        return {"contract_code": symbol, "period": self.timeframes[timeframe],
                "from": start // 1000, "to": end // 1000 - 1}

    def parse_candles(self, payload):
        return [[int(r["id"]) * 1000, float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["amount"])] for r in payload["data"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("status") != "ok":
            return payload.get("err-msg") or f"error {payload.get('err-code')}"
        return None


class Hyperliquid(Exchange):
    """
    Hyperliquid perpetuals (USDC). https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint

    Only the most recent ~5000 candles of each interval are served (1m: about 3.5 days,
    1h: about 7 months, 4h: about 2.3 years); older windows come back empty. 12h and 1d
    reach back further, but rows before trading began (BTC: 2023-02-26) are placeholders
    with no trades and are dropped. Volume `v` is in the base asset. There is no 6h.
    Funding is paid hourly; `fundingRate` is the rate per hour (history from May 2023).
    An unknown coin makes the API answer HTTP 500, so names are checked against `meta`.
    """
    name = "hyperliquid"
    candles_url = "https://api.hyperliquid.xyz/info"
    timeframes = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
                  "2h": "2h", "4h": "4h", "12h": "12h", "1d": "1d"}
    candle_limit = 5000
    funding_url = "https://api.hyperliquid.xyz/info"
    funding_limit = 500
    # 1200 weight a minute per IP: a request costs 20, plus 1 per 60 candles or per 20
    # funding records returned, so a full funding page weighs 45.
    pause = 3.0
    method = "POST"
    default_quote = "USDC"
    _coins = None

    def symbol(self, base, quote):
        if quote != "USDC":
            raise ExchangeError(f"hyperliquid: perpetuals settle in USDC, not {quote}.")
        if self._coins is None:
            meta = self._request(self.candles_url, {"type": "meta"})
            self._coins = {c["name"].upper(): c["name"] for c in meta["universe"]}
        if base.upper() not in self._coins:
            raise ExchangeError(f"hyperliquid: unknown coin {base!r}.")
        return self._coins[base.upper()]       # names such as kPEPE are case-sensitive

    def candle_params(self, symbol, timeframe, start, end):
        # endTime is inclusive.
        return {"type": "candleSnapshot", "req": {"coin": symbol, "interval": self.timeframes[timeframe],
                                                  "startTime": start, "endTime": end - 1}}

    def parse_candles(self, payload):
        return [[int(r["t"]), float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"])]
                for r in payload if r["n"] > 0]

    def funding_params(self, symbol, start, end):
        return {"type": "fundingHistory", "coin": symbol, "startTime": start, "endTime": end - 1}

    def parse_funding(self, payload):
        return [[int(r["time"]), float(r["fundingRate"])] for r in payload]


class Kraken(Exchange):
    """
    Kraken Futures multi-collateral perpetuals (PF_XBTUSD). https://docs.kraken.com/api/docs/futures-api/charts/candles

    Markets are quoted in USD and BTC is called XBT. Contracts are one unit of the base
    asset, so `volume` is already in the base asset. 1m history starts at the listing
    (BTC: 2022-03-23). No 3m, 2h or 6h bars. An unknown market answers with an empty
    list, so names are checked against the market list. Funding is paid hourly;
    `relativeFundingRate` is the rate per payment (`fundingRate` is USD per contract).
    The funding endpoint only serves the last year, all of it in one response.
    """
    name = "kraken"
    candles_url = "https://futures.kraken.com/api/charts/v1/trade/{}/{}"
    timeframes = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
                  "4h": "4h", "12h": "12h", "1d": "1d"}
    candle_limit = 2000
    funding_url = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"
    funding_limit = 24 * 366    # one response holds the whole year it keeps
    pause = 0.2                 # no published limit for these endpoints
    default_quote = "USD"
    _markets = None

    def symbol(self, base, quote):
        name = f"PF_{'XBT' if base == 'BTC' else base}{quote}"
        if self._markets is None:
            self._markets = set(self._request("https://futures.kraken.com/api/charts/v1/trade", {}))
        if name not in self._markets:
            raise ExchangeError(f"kraken: no perpetual {name}.")
        return name

    def fetch_candles(self, base, quote, timeframe, start, end):
        # The market and resolution are part of the URL path.
        if timeframe in self.timeframes:
            self.candles_url = type(self).candles_url.format(self.symbol(base, quote),
                                                             self.timeframes[timeframe])
        return super().fetch_candles(base, quote, timeframe, start, end)

    def candle_params(self, symbol, timeframe, start, end):
        # Seconds; `to` is inclusive.
        return {"from": start // 1000, "to": (end - 1) // 1000}

    def parse_candles(self, payload):
        return [[int(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["volume"])] for r in payload["candles"]]

    def funding_params(self, symbol, start, end):
        return {"symbol": symbol}

    def parse_funding(self, payload):
        return [[int(datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp() * 1000),
                 float(r["relativeFundingRate"])] for r in payload["rates"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("result") == "error":
            errors = payload.get("errors") or [{"message": payload.get("error")}]
            return "; ".join(str(e.get("message")) for e in errors)
        return None


class KuCoin(Exchange):
    """
    KuCoin Futures USDT-margined perpetuals. https://www.kucoin.com/docs-new/rest/futures-trading/market-data/get-klines
    BTC is called XBT. Volume is lots times the contract multiplier. Unknown symbols
    return an empty success, so the contract lookup is what reports them.
    """
    name = "kucoin"
    candles_url = "https://api-futures.kucoin.com/api/v1/kline/query"
    timeframes = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120",
                  "4h": "240", "12h": "720", "1d": "1440"}
    candle_limit = 200
    funding_url = "https://api-futures.kucoin.com/api/v1/contract/funding-rates"
    funding_limit = 100
    pause = 0.1         # 2000 weight per 30 seconds, 3 per kline request

    def __init__(self):
        super().__init__()
        self._multipliers = {}

    def symbol(self, base, quote):
        return f"{'XBT' if base == 'BTC' else base}{quote}M"

    def candle_params(self, symbol, timeframe, start, end):
        # Bars from `from`, oldest first, at most 200.
        return {"symbol": symbol, "granularity": self.timeframes[timeframe],
                "from": start, "to": end - 1}

    def parse_candles(self, payload):
        size = self._multiplier(self.current)
        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]) * size]
                for r in payload["data"]]

    def _multiplier(self, symbol):
        if symbol not in self._multipliers:
            info = self._request(f"https://api-futures.kucoin.com/api/v1/contracts/{symbol}", {})
            self._multipliers[symbol] = float(info["data"]["multiplier"])
        return self._multipliers[symbol]

    def funding_params(self, symbol, start, end):
        return {"symbol": symbol, "from": start, "to": end - 1}

    def parse_funding(self, payload):
        return [[int(r["timepoint"]), float(r["fundingRate"])] for r in payload["data"] or []]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("code", "200000") != "200000":
            return payload.get("msg") or f"error {payload['code']}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == "429000"


class MEXC(Exchange):
    """
    MEXC USDT-M perpetuals (contract API). https://www.mexc.com/api-docs/futures/market-endpoints
    1m bars go back about 30 days, 5m and 15m about a year. Volume is contracts times
    the contract size. Funding history is only served by page number, so it is not supported.
    """
    name = "mexc"
    timeframes = {"1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30", "1h": "Min60",
                  "4h": "Hour4", "1d": "Day1"}
    candle_limit = 2000
    funding_url = None
    pause = 0.12        # 20 requests per 2 seconds

    def __init__(self):
        super().__init__()
        self._sizes = {}

    def symbol(self, base, quote):
        return f"{base}_{quote}"

    @property
    def candles_url(self):
        # The symbol goes in the path; `current` is set before the first request.
        return f"https://contract.mexc.com/api/v1/contract/kline/{self.current}"

    def candle_params(self, symbol, timeframe, start, end):
        # Seconds, both bounds inclusive.
        return {"interval": self.timeframes[timeframe], "start": start // 1000,
                "end": end // 1000 - 1}

    def parse_candles(self, payload):
        size = self._size(self.current)
        d = payload["data"]
        return [[int(t) * 1000, float(o), float(h), float(l), float(c), float(v) * size]
                for t, o, h, l, c, v in zip(d["time"], d["open"], d["high"], d["low"],
                                            d["close"], d["vol"])]

    def _size(self, symbol):
        if symbol not in self._sizes:
            info = self._request("https://contract.mexc.com/api/v1/contract/detail",
                                 {"symbol": symbol})
            self._sizes[symbol] = float(info["data"]["contractSize"])
        return self._sizes[symbol]

    def error_message(self, payload):
        if isinstance(payload, dict) and not payload.get("success", False):
            return payload.get("message") or f"error {payload.get('code')}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == 510


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


class Phemex(Exchange):
    """
    Phemex USDT-M perpetuals. https://phemex-docs.github.io/
    Rows hold real prices (the Rp fields, no Ep scaling) and volume in the base asset
    (one contract is one coin); history starts at listing (BTCUSDT: 2022-11-12).
    """
    name = "phemex"
    candles_url = "https://api.phemex.com/exchange/public/md/v2/kline/list"
    # 3m, 2h, 6h and 12h are not in the docs but are served and start at midnight UTC.
    timeframes = {"1m": "60", "3m": "180", "5m": "300", "15m": "900", "30m": "1800",
                  "1h": "3600", "2h": "7200", "4h": "14400", "6h": "21600",
                  "12h": "43200", "1d": "86400"}
    candle_limit = 1000     # the docs allow "less than 2000" bars per request
    funding_url = "https://api.phemex.com/api-data/public/data/funding-rate-history"
    funding_limit = 100
    pause = 0.6             # klines weigh 10; 5000 weight per 5 minutes per IP

    def __init__(self):
        super().__init__()
        self._known: set[str] = set()

    def symbol(self, base, quote):
        symbol = f"{base}{quote}"
        if symbol not in self._known:
            # Funding history answers an unknown symbol with no rows; the kline
            # endpoint rejects it ("Symbol not supported").
            self._request("https://api.phemex.com/exchange/public/md/v2/kline/last",
                          {"symbol": symbol, "resolution": "86400", "limit": 5})
            self._known.add(symbol)
        return symbol

    def candle_params(self, symbol, timeframe, start, end):
        # Seconds; `from` is inclusive and `to` exclusive.
        return {"symbol": symbol, "resolution": self.timeframes[timeframe],
                "from": start // 1000, "to": end // 1000}

    def parse_candles(self, payload):
        # [time s, interval, last close, open, high, low, close, volume, turnover, symbol]
        return [[int(r[0]) * 1000, float(r[3]), float(r[4]), float(r[5]), float(r[6]),
                 float(r[7])] for r in payload["data"]["rows"]]

    def funding_params(self, symbol, start, end):
        # The rate is published under ".BTCUSDTFR8H" whatever the interval, and is the
        # rate actually paid each interval. start and end are both inclusive.
        return {"symbol": f".{symbol}FR8H", "start": start, "end": end - 1,
                "limit": self.funding_limit}

    def parse_funding(self, payload):
        return [[int(r["fundingTime"]), float(r["fundingRate"])]
                for r in payload["data"]["rows"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("code", 0) != 0:
            return payload.get("msg") or f"error {payload['code']}"
        return None


class Woo(Exchange):
    """
    WOO X perpetuals. https://developer.woox.io/api-reference/introduction
    Volume is in the base asset. 1m, 1h and longer bars go back to listing (BTC: 2022-03),
    but shorter ones are kept for a limited time only (late 2026: 3m about 30 days,
    5m about 10 months, 15m about 15 months, 30m since 2024-06); older windows come back empty.
    """
    name = "woo"
    candles_url = "https://api-pub.woo.org/v1/hist/kline"
    timeframes = {tf: tf for tf in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                                    "6h", "12h", "1d")}
    candle_limit = 1000     # also: at most 7 days of 1m bars per request
    funding_url = "https://api.woo.org/v3/public/fundingRateHistory"
    funding_limit = 500
    pause = 0.15            # 10 requests per second per IP

    def __init__(self):
        super().__init__()
        self._known: set[str] = set()

    def symbol(self, base, quote):
        symbol = f"PERP_{base}_{quote}"
        if symbol not in self._known:
            # The candle endpoint answers an unknown symbol with no rows, so ask the
            # funding endpoint, which rejects it ("invalid symbol ...").
            self._request(self.funding_url, {"symbol": symbol, "size": 1})
            self._known.add(symbol)
        return symbol

    def candle_params(self, symbol, timeframe, start, end):
        # end_time is inclusive.
        return {"symbol": symbol, "type": self.timeframes[timeframe],
                "start_time": start, "end_time": end - 1, "size": self.candle_limit}

    def parse_candles(self, payload):
        return [[int(r["start_timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["volume"])] for r in payload["data"]["rows"]]

    def funding_params(self, symbol, start, end):
        # startTime is inclusive, endTime exclusive.
        return {"symbol": symbol, "startTime": start, "endTime": end,
                "size": self.funding_limit}

    def parse_funding(self, payload):
        return [[int(r["fundingRateTimestamp"]), float(r["fundingRate"])]
                for r in payload["data"]["rows"]]

    def error_message(self, payload):
        if isinstance(payload, dict) and payload.get("success") is False:
            return payload.get("message") or f"error {payload.get('code')}"
        return None

    def throttled(self, payload):
        return isinstance(payload, dict) and payload.get("code") == -1003


EXCHANGES: dict[str, type[Exchange]] = {cls.name: cls for cls in (
    Binance, Bybit, OKX, Aster, BingX, Bitfinex, Bitget, BloFin, CryptoCom, Deribit, DYDX,
    Gate, HTX, Hyperliquid, Kraken, KuCoin, MEXC, Phemex, Woo,
)}


def get_exchange(name: str) -> Exchange:
    """A connector by name, e.g. "binance"."""
    try:
        return EXCHANGES[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown exchange {name!r}. Available: {', '.join(EXCHANGES)}.") from None
