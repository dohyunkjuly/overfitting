from __future__ import annotations

import warnings
from decimal import Decimal

import numpy as np
import pandas as pd

from overfitting.data import Data
from overfitting.errors import (
    EmptyOrderParameters,
    InsufficientMargin,
    InvalidOrderParameters,
    LiquidationError,
    StopOrderRejected,
)
from overfitting.order import Order, OrderType, Status
from overfitting.position import Position
from overfitting.recorder import NullRecorder, Recorder
from overfitting.slippage import Bar, resolve as resolve_slippage

# Column in the price data that holds the funding rate, if any.
FUNDING_COLUMN = "funding_rate"
# Relative slack on the margin check, so float rounding never rejects an all-in order.
MARGIN_TOLERANCE = 1e-9


class Broker:
    def __init__(self,
                 data: Data,
                 cash: float,
                 commission_rate: float,
                 maint_margin_rate: float,
                 maint_amount: float,
                 slippage=None,
                 record: bool = True,
                 funding: bool = False):

        self.data = data
        self.initial_capital = cash
        self.cash = self.initial_capital
        self.commission_rate = commission_rate
        self.maint_margin_rate = maint_margin_rate
        self.maint_amount = maint_amount
        # A function (order, bar) -> executed price, or None for no slippage.
        self.slippage = resolve_slippage(slippage)
        # True settles funding from the funding_rate column; off by default.
        self.funding = funding
        # open orders example:
        #   {
        #       btcusdt: {"order-id-01": <Order>, "order-id-02": <Order>},
        #       ethusdt: {"order-id-01": <Order>, "order-id-02": <Order>},
        #   }
        self.open_orders: dict[str, dict[str, Order]] = {}
        self.position: dict[str, Position] = {}
        # Symbols with an open position, in the order their positions were created.
        # Rebuilt only when a position opens or closes, so the per-bar loops skip
        # symbols that are flat.
        self._held: list[str] = []
        # Each symbol's funding_rate array (None without the column), looked up once.
        self._rates = {s: getattr(data[s], FUNDING_COLUMN, None) for s in data.symbols}
        self._funding_bars = None   # True where any symbol has a funding rate; built on first use

        self.trades = []
        self.funding_payments = []
        self.rejected_orders = []
        # Every order ever placed, whatever became of it (see Order.status).
        self.orders: list[Order] = []
        self.recorder = Recorder(data) if record else NullRecorder(data)
        self._i = 0

    @property
    def initial_captial(self):
        """Deprecated misspelling of `initial_capital`."""
        warnings.warn(
            "Broker.initial_captial is deprecated, use Broker.initial_capital",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.initial_capital

    def __repr__(self):
        return (f"Broker("
                f"initial_capital={self.initial_capital}, "
                f"cash={self.cash}, "
                f"commission_rate={self.commission_rate}, "
                f"maint_margin_rate={self.maint_margin_rate}, "
                f"maint_amount={self.maint_amount}, "
                f"open_orders={len(self.open_orders)}, "
                f"positions={list(self.position.keys())}, "
                f"trades={len(self.trades)})")

    def _d(self, symbol: str):
        return self.data[symbol]

    def _time(self, i: int) -> pd.Timestamp:
        # data.index holds tz-naive datetime64 values, so pd.Timestamp gives the
        # same result as pd.to_datetime at a fraction of the cost.
        return pd.Timestamp(self.data.index[i])

    def _bars(self, symbol: str, i: int) -> tuple:
        d = self.data[symbol]
        return d.open[i], d.high[i], d.low[i], d.close[i]

    def _open(self, symbol: str, i: int):
        return self._d(symbol).open[i]

    def _high(self, symbol: str, i: int):
        return self._d(symbol).high[i]

    def _low(self, symbol: str, i: int):
        return self._d(symbol).low[i]

    def _close(self, symbol: str, i: int):
        return self._d(symbol).close[i]

    def order(self,
              symbol: str,
              qty: float,
              price: float | None,
              *,
              type: str = "LIMIT",
              stop_price: float | None = None,
              label: str | None = None) -> Order:
        """
        Place an order and return it. `qty` is negative for a sell.

        `type` is "LIMIT" (the default), "MARKET" or "STOP". A LIMIT order needs
        `price`; a STOP order needs `stop_price`, and with a `price` as well it
        is a stop-limit. Raises if the order is rejected.
        """
        if not symbol or not isinstance(symbol, str):
            raise InvalidOrderParameters(f"symbol must be a non-empty string. - {symbol}")

        if not qty or not isinstance(qty, (int, float, Decimal)):
            raise InvalidOrderParameters(f"qty must be a non-empty number. - {qty}")

        self.get_position(symbol)
        self.open_orders.setdefault(symbol, {})

        if isinstance(type, OrderType):
            pass
        elif isinstance(type, str) and type.upper() in OrderType.__members__:
            type = OrderType[type.upper()]
        else:
            raise InvalidOrderParameters(f"Invalid Order Type - {type}")

        if type == OrderType.STOP and stop_price is None:
            raise EmptyOrderParameters("stop_price must be given for a STOP order.")

        if type == OrderType.LIMIT and price is None:
            raise EmptyOrderParameters("price must be given for a LIMIT order.")

        order = Order(self._time(self._i), symbol, qty, price, type, stop_price, label)
        self.orders.append(order)

        try:
            if type == OrderType.STOP:
                open = self._open(symbol, self._i)
                if ((order.qty > 0 and order.stop_price < open) or
                    (order.qty < 0 and order.stop_price > open)):
                    raise StopOrderRejected(
                        f"STOP order rejected: stop price {order.stop_price} would trigger immediately at current price {open}."
                    )

            # Reject orders the account cannot afford at the price we expect to pay.
            if type == OrderType.MARKET:
                expected_price = self._open(symbol, self._i)
            elif price is not None:
                expected_price = price
            else:
                expected_price = stop_price
            self._check_margin(symbol, order, expected_price)
        except (StopOrderRejected, InsufficientMargin) as e:
            order.status = Status.REJECTED
            order.reject_reason = str(e)
            raise

        # Put new order in the open_orders list if not rejected
        self.open_orders[symbol][order.id] = order
        return order

    def cancel_order(self, symbol: str, order_id: str) -> Order | None:
        order = self.open_orders.get(symbol, {}).pop(order_id, None)
        if order is not None:
            self._mark_cancelled(order)
        return order

    def _mark_cancelled(self, order: Order) -> None:
        order.status = Status.CANCELLED
        order.cancelled_at = self._time(min(self._i, len(self.data) - 1))

    def cancel_all_orders(self, symbol: str) -> None:
        if symbol not in self.open_orders:
            return

        orders = self.open_orders[symbol]
        for order_id in list(orders.keys()):
            self._mark_cancelled(orders[order_id])
            del orders[order_id]

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.maint_margin_rate, self.maint_amount)

        return self.position[symbol]

    def close_all_positions(self, symbol: str) -> None:
        position = self.get_position(symbol)
        if position.qty != 0:
            self.order(symbol, -position.qty, None, type="MARKET")

    def set_leverage(self, symbol: str, leverage: int) -> None:
        position = self.get_position(symbol)
        position.set_leverage(leverage)
        # Check if the position would be liquidated with the new leverage
        lp = position.liquid_price
        p = self._open(symbol, self._i)

        if (position.qty > 0 and p <= lp) or \
           (position.qty < 0 and p >= lp):
            raise LiquidationError(
                f"Cannot change leverage for {symbol}. Position would be liquidated at price {lp}.")

    def available_margin(self) -> float:
        """Cash that is not locked as margin in open positions."""
        return self.cash - sum(p.margin for p in self.position.values())

    def required_margin(self, symbol: str, qty, price: float) -> float:
        """
        Extra margin needed to trade `qty` of `symbol` at `price`.
        Orders that only reduce a position need none.
        """
        position = self.get_position(symbol)
        qty = Decimal(str(qty))
        new_qty = position.qty + qty

        if abs(new_qty) <= abs(position.qty):
            return 0.0  # reducing or closing

        if position.qty == 0 or position.qty * qty > 0:
            # Opening or adding: margin for the added size.
            return float(abs(qty)) * price / position.leverage

        # Flipping sides: the old margin is released, the new side needs its own.
        new_margin = float(abs(new_qty)) * price / position.leverage
        return max(0.0, new_margin - position.margin)

    def _check_margin(self, symbol: str, order: Order, price: float) -> None:
        required = self.required_margin(symbol, order.qty, price)
        if required <= 0:
            return  # reducing or closing is always allowed
        available = self.available_margin()
        if required > available + MARGIN_TOLERANCE * max(1.0, abs(available)):
            raise InsufficientMargin(
                f"Order for {float(order.qty)} {symbol} needs {required:.2f} margin "
                f"but only {available:.2f} is available."
            )

    def unrealized_pnl(self, i: int | None = None) -> float:
        """Unrealized PnL of all open positions, marked at the close of bar `i`."""
        if i is None:
            i = min(self._i, len(self.data) - 1)
        # Same sum as Position.unrealized_pnl, inlined: every held position has qty != 0.
        data, position = self.data, self.position
        return sum([(data[s].close[i] - position[s].price) * float(position[s].qty)
                    for s in self._held])

    def equity(self, i: int | None = None) -> float:
        """Cash plus unrealized PnL, marked at the close of bar `i`."""
        return self.cash + self.unrealized_pnl(i)

    def _find_funding_bars(self) -> np.ndarray:
        """Bars where any symbol has a non-zero funding rate; the rest can't settle anything."""
        bars = np.zeros(len(self.data), dtype=bool)
        for rates in self._rates.values():
            if rates is not None:
                r = pd.to_numeric(pd.Series(rates), errors="coerce").to_numpy(dtype=float)
                bars |= (r != 0) & ~np.isnan(r)
        return bars

    def _apply_funding(self) -> None:
        """
        Settle funding for positions held into this bar.

        A non-zero value in the `funding_rate` column means funding is paid at
        the open of that bar. Longs pay shorts when the rate is positive.
        """
        i = self._i
        for symbol in self._held:
            rates = self._rates[symbol]
            if rates is None:
                continue
            rate = rates[i]
            if rate is None or rate != rate or rate == 0:  # rate != rate catches NaN
                continue

            position = self.position[symbol]
            price = self.data[symbol].open[i]
            amount = -float(position.qty) * price * float(rate)
            self.cash += amount
            self.recorder.on_funding(symbol, amount)
            self.funding_payments.append({
                "time": self._time(self._i),
                "symbol": symbol,
                "qty": float(position.qty),
                "price": price,
                "funding_rate": float(rate),
                "amount": amount,
            })

    def _slippage(self, symbol: str, order: Order) -> float:
        if self.slippage:
            return self.slippage(order, Bar(*self._bars(symbol, self._i)))

        return order.theoretical_price

    def _execute_trade(self, symbol: str, order: Order, price: float | None = None,
                       liquidation: bool = False) -> None:
        order.theoretical_price = price if price else order.price
        order.executed_price = self._slippage(symbol, order)

        if not order.theoretical_price:
            raise EmptyOrderParameters("Cannot execute an order without a price.")

        position = self.position[symbol]

        if liquidation:
            # The exchange keeps the margin; nothing more is charged on top.
            commission = 0.0
        else:
            # The account may have changed since the order was placed.
            # Checked at the pre-slippage price, the same basis as at placement,
            # so slippage alone never drops an all-in order.
            try:
                self._check_margin(symbol, order, order.theoretical_price)
            except InsufficientMargin as e:
                self.open_orders.get(symbol, {}).pop(order.id, None)
                order.status = Status.REJECTED
                order.reject_reason = str(e)
                self.rejected_orders.append(order.to_dict())
                return
            notional = float(abs(order.qty)) * order.executed_price
            commission = notional * self.commission_rate
        qty_before = position.qty
        pnl = position.process_trade(order, liquidation)
        if (position.qty != 0) != (symbol in self._held):
            self._held = [s for s, p in self.position.items() if p.qty != 0]
        executed_time = self._time(self._i)

        order.pnl = pnl
        order.executed_at = executed_time
        order.commission = commission
        order.realized_pnl = pnl - commission
        order.is_liquidation_order = bool(liquidation)
        order.status = Status.FILLED
        side = 1 if order.qty > 0 else -1
        order.slippage = side * (order.executed_price - order.theoretical_price) * float(abs(order.qty))
        self.recorder.on_fill(self._i, order, qty_before, pnl, commission, bool(liquidation))

        self.trades.append(order.to_dict())
        self.cash += order.realized_pnl

        if symbol in self.open_orders and order.id in self.open_orders[symbol]:
            del self.open_orders[symbol][order.id]

    def next(self) -> None:
        data = self.data
        i = self._i
        if i != 0:  # liquidations, checked against the previous bar's range
            prev = i - 1
            for s in self._held:          # a liquidation rebuilds _held; this loop keeps the old list
                p = self.position[s]
                lp = p.liquid_price
                # Held positions are never flat, so qty > 0 means long.
                if (data[s].low[prev] <= lp) if p.qty > 0 else (data[s].high[prev] >= lp):
                    order = self.order(p.symbol, -p.qty, lp, type="MARKET")
                    self._execute_trade(p.symbol, order, lp, True)

        # Funding is settled at the open, before this bar's orders fill.
        if self.funding:
            if self._funding_bars is None:
                self._funding_bars = self._find_funding_bars()
            if self._funding_bars[i]:
                self._apply_funding()

        # Fill open orders. Iterate over copies: a fill removes the order from its book.
        for symbol, by_id in list(self.open_orders.items()):
            if not by_id:
                continue
            d = data[symbol]
            open, high, low = d.open[i], d.high[i], d.low[i]

            for _, order in list(by_id.items()):
                if order.type == OrderType.MARKET:
                    self._execute_trade(symbol, order, open)  # market orders fill at the open
                elif order.type == OrderType.LIMIT:
                    if ((order.qty > 0 and low < order.price) or
                        (order.qty < 0 and high > order.price)):
                        # Placed on this bar and the open is already better than the
                        # limit: it fills at the open. A buy limit at 110 with the
                        # open at 100 fills at 100.
                        if order.created_at == self._time(self._i):
                            if ((order.qty > 0 and order.price > open) or
                                (order.qty < 0 and order.price < open)):
                                self._execute_trade(symbol, order, open)
                                continue

                        self._execute_trade(symbol, order)
                else:
                    # STOP LIMIT, STOP MARKET Trigger Condition:
                    # LONG: Current Price >= Stop Price
                    # SHORT: Current Price <= Stop Price
                    if not order.is_triggered:
                        if ((order.qty > 0 and high >= order.stop_price) or
                            (order.qty < 0 and low <= order.stop_price)):
                            order.is_triggered = True
                            # The price when the stop triggers: the stop price, or the
                            # open if the bar opened already past it.
                            trigger = (max(order.stop_price, open) if order.qty > 0
                                       else min(order.stop_price, open))

                            if order.price is None: # STOP MARKET ORDER
                                self._execute_trade(symbol, order, trigger)
                                continue

                            # STOP LIMIT: a limit order placed at the trigger. A buy
                            # limit at or above the trigger (a sell limit at or below
                            # it) fills there at once, not at the limit price.
                            if ((order.qty > 0 and order.price >= trigger) or
                                (order.qty < 0 and order.price <= trigger)):
                                self._execute_trade(symbol, order, trigger)
                                continue

                if order.is_triggered:
                    if order.price is None: # STOP MARKET ORDER
                        self._execute_trade(symbol, order, order.stop_price)
                    else: # STOP LIMIT ORDER: resting limit, fills when the price trades through it
                        if ((order.qty > 0 and low < order.price) or
                            (order.qty < 0 and high > order.price)):
                            self._execute_trade(symbol, order)

        self.recorder.on_bar_end(self._i, self)
        self._i += 1
