"""
Records what happened during a backtest, for analysis afterwards.

The broker reports three kinds of event to a Recorder:

    on_fill      an order was executed
    on_funding   a funding payment was settled
    on_bar_end   a bar finished

From these it builds:

    round trips       one row per completed trade, from opening a position to
                      being flat again (or flipping to the other side)
    account history   cash, equity, margin, exposure and leverage on every bar
    position history  size, entry price, margin, liquidation price and
                      unrealized PnL of each symbol on every bar

Recording never changes cash, positions or fills.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

ACCOUNT_FIELDS = ("cash", "equity", "margin", "gross_exposure", "net_exposure", "leverage")
POSITION_FIELDS = (
    "qty", "entry_price", "close", "exposure", "margin", "liquidation_price", "unrealized_pnl",
)
ROUND_TRIP_COLUMNS = [
    "symbol", "side", "entry_time", "exit_time", "bars_held", "duration", "qty",
    "entry_price", "exit_price", "pnl", "commission", "funding", "net_pnl", "return_pct",
    "mae", "mfe", "fills", "entry_label", "exit_label", "is_liquidation",
]


class Recorder:
    def __init__(self, data):
        self._index = pd.to_datetime(data.index)
        n = len(self._index)
        self._bars = 0

        self.round_trips: list[dict] = []
        self._open: dict[str, dict] = {}   # symbol -> the round trip in progress

        self._account = {f: np.full(n, np.nan) for f in ACCOUNT_FIELDS}
        # Filled in advance as if every symbol were flat on every bar; on_bar_end
        # then only writes the symbols that hold a position.
        self._positions = {}
        for symbol in data.symbols:
            fields = {f: np.zeros(n) for f in ("qty", "exposure", "margin", "unrealized_pnl")}
            fields["close"] = np.asarray(data[symbol].close, dtype=float).copy()
            fields["entry_price"] = np.full(n, np.nan)       # NaN while flat
            fields["liquidation_price"] = np.full(n, np.nan)
            self._positions[symbol] = {f: fields[f] for f in POSITION_FIELDS}
        self._rank = {symbol: k for k, symbol in enumerate(data.symbols)}
        self._held_source = None      # broker._held the order below was computed from
        self._held_order: list[str] = []

    # ------------------------------------------------------------ round trips ---

    def on_fill(self, i: int, order, qty_before: Decimal, pnl: float, commission: float,
                liquidation: bool) -> None:
        """
        Fold one executed order into the symbol's round trip.

        `qty_before` is the position size before the fill, `pnl` the realized
        PnL of the fill, `commission` what it cost.
        """
        symbol = order.symbol
        price = float(order.executed_price)
        size = abs(order.qty)

        if liquidation:
            trip = self._open.get(symbol)
            if trip is not None:
                trip["exit_notional"] += float(abs(qty_before)) * price
                trip["exit_qty"] += abs(qty_before)
                self._close(trip, i, price, pnl, commission, order, liquidation=True)
            return

        if qty_before == 0:
            self._open[symbol] = self._new_trip(i, order, size, price, commission)
            return

        trip = self._open[symbol]
        if qty_before * order.qty > 0:
            # Adding to the position.
            trip["entry_notional"] += float(size) * price
            trip["entry_qty"] += size
            trip["max_qty"] = max(trip["max_qty"], abs(qty_before + order.qty))
            trip["commission"] += commission
            trip["fills"] += 1
            self._stretch(trip, price)
            return

        # Reducing, closing or flipping.
        closed = min(abs(qty_before), size)
        share = float(closed / size)          # part of this fill that closes the old side
        trip["exit_notional"] += float(closed) * price
        trip["exit_qty"] += closed
        qty_after = qty_before + order.qty

        if qty_after != 0 and qty_after * qty_before > 0:
            # Partial close: the trip stays open.
            trip["pnl"] += pnl
            trip["commission"] += commission
            trip["fills"] += 1
            self._stretch(trip, price)
            return

        self._close(trip, i, price, pnl, commission * share, order, liquidation=False)
        if qty_after != 0:
            # Flipped: the rest of the fill opens a trip on the other side.
            self._open[symbol] = self._new_trip(
                i, order, abs(qty_after), price, commission * (1.0 - share))

    def on_funding(self, symbol: str, amount: float) -> None:
        trip = self._open.get(symbol)
        if trip is not None:
            trip["funding"] += amount

    def _new_trip(self, i: int, order, size: Decimal, price: float, commission: float) -> dict:
        return {
            "symbol": order.symbol,
            "side": 1 if order.qty > 0 else -1,
            "entry_i": i,
            "entry_notional": float(size) * price,
            "entry_qty": size,
            "exit_notional": 0.0,
            "exit_qty": Decimal("0"),
            "max_qty": size,
            "pnl": 0.0,
            "commission": commission,
            "funding": 0.0,
            "high": price,
            "low": price,
            "fills": 1,
            "entry_label": order.label,
        }

    @staticmethod
    def _stretch(trip: dict, *prices: float) -> None:
        """Widen the trip's best and worst price to include `prices`."""
        trip["high"] = max(trip["high"], *prices)
        trip["low"] = min(trip["low"], *prices)

    def _close(self, trip: dict, i: int, price: float, pnl: float, commission: float,
               order, liquidation: bool) -> None:
        self._stretch(trip, price)
        entry_price = trip["entry_notional"] / float(trip["entry_qty"])
        exit_price = trip["exit_notional"] / float(trip["exit_qty"])
        side = trip["side"]
        gross = trip["pnl"] + pnl
        fees = trip["commission"] + commission
        net = gross - fees + trip["funding"]

        # Furthest the price went for and against the position, relative to entry.
        best = trip["high"] if side > 0 else trip["low"]
        worst = trip["low"] if side > 0 else trip["high"]
        mfe = max(0.0, side * (best - entry_price) / entry_price)
        mae = min(0.0, side * (worst - entry_price) / entry_price)

        entry_time, exit_time = self._index[trip["entry_i"]], self._index[i]
        self.round_trips.append({
            "symbol": trip["symbol"],
            "side": "LONG" if side > 0 else "SHORT",
            "entry_time": entry_time,
            "exit_time": exit_time,
            "bars_held": i - trip["entry_i"],
            "duration": exit_time - entry_time,
            "qty": float(trip["max_qty"]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": gross,
            "commission": fees,
            "funding": trip["funding"],
            "net_pnl": net,
            "return_pct": net / trip["entry_notional"],
            "mae": mae,
            "mfe": mfe,
            "fills": trip["fills"] + 1,
            "entry_label": trip["entry_label"],
            "exit_label": order.label,
            "is_liquidation": liquidation,
        })
        del self._open[trip["symbol"]]

    # ------------------------------------------------------------ per-bar state ---

    def on_bar_end(self, i: int, broker) -> None:
        """Snapshot the account and every open position at the close of bar `i`."""
        if broker._held is not self._held_source:     # a position opened or closed
            self._held_source = broker._held
            self._held_order = sorted((s for s in broker._held if s in self._rank),
                                      key=self._rank.__getitem__)
        margin = gross = net = unrealized_total = 0.0
        positions, data, open_trips = broker.position, broker.data, self._open
        for symbol in self._held_order:
            fields = self._positions[symbol]
            position = positions[symbol]
            close = fields["close"][i]
            qty = float(position.qty)
            exposure = qty * close
            unrealized = (close - position.price) * qty   # Position.unrealized_pnl; never flat here
            fields["qty"][i] = qty
            fields["entry_price"][i] = position.price
            fields["exposure"][i] = exposure
            fields["margin"][i] = position.margin
            fields["liquidation_price"][i] = position.liquid_price
            fields["unrealized_pnl"][i] = unrealized

            margin += position.margin
            gross += abs(exposure)
            net += exposure
            unrealized_total += unrealized

            trip = open_trips.get(symbol)
            if trip is not None:
                d = data[symbol]
                high, low = float(d.high[i]), float(d.low[i])
                trip["high"] = max(trip["high"], high, low)
                trip["low"] = min(trip["low"], high, low)

        equity = broker.cash + unrealized_total
        account = self._account
        account["cash"][i] = broker.cash
        account["equity"][i] = equity
        account["margin"][i] = margin
        account["gross_exposure"][i] = gross
        account["net_exposure"][i] = net
        account["leverage"][i] = gross / equity if equity > 0 else np.nan
        self._bars = max(self._bars, i + 1)

    # ------------------------------------------------------------------ output ---

    def round_trips_frame(self) -> pd.DataFrame:
        """Completed round trips, one row each, in the order they closed."""
        return pd.DataFrame(self.round_trips, columns=ROUND_TRIP_COLUMNS)

    def open_symbols(self) -> list[str]:
        """Symbols with a round trip still in progress."""
        return list(self._open)

    def account_history(self) -> pd.DataFrame:
        """Account state at the close of every bar that has run."""
        n = self._bars
        return pd.DataFrame({f: values[:n] for f, values in self._account.items()},
                            index=self._index[:n])

    def position_history(self, symbol: str) -> pd.DataFrame:
        """One symbol's position at the close of every bar that has run."""
        if symbol not in self._positions:
            raise KeyError(f"No data for symbol {symbol!r}.")
        n = self._bars
        return pd.DataFrame({f: values[:n] for f, values in self._positions[symbol].items()},
                            index=self._index[:n])


class NullRecorder:
    """
    Records nothing. Used with `Strategy(record=False)` when only the returns
    matter, for example inside a parameter search. Every table comes back empty.
    """

    round_trips: list = []

    def __init__(self, data):
        self._symbols = set(data.symbols)

    def on_fill(self, *args, **kwargs) -> None:
        pass

    def on_funding(self, *args, **kwargs) -> None:
        pass

    def on_bar_end(self, *args, **kwargs) -> None:
        pass

    def round_trips_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=ROUND_TRIP_COLUMNS)

    def open_symbols(self) -> list[str]:
        return []

    def account_history(self) -> pd.DataFrame:
        return pd.DataFrame(columns=list(ACCOUNT_FIELDS))

    def position_history(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._symbols:
            raise KeyError(f"No data for symbol {symbol!r}.")
        return pd.DataFrame(columns=list(POSITION_FIELDS))
