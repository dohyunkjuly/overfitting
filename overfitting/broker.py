import pandas as pd
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from overfitting.data import Data
from overfitting.order import Order, OrderType, Status
from overfitting.position import Position
from overfitting.slippage import Bar, resolve as resolve_slippage
from overfitting.recorder import NullRecorder, Recorder
import warnings
from overfitting.errors import (
    EmptyOrderParameters,
    InsufficientMargin,
    InvalidOrderParameters,
    LiquidationError,
    StopOrderRejected,
)

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
                 maint_amount:float,
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
        self.open_orders: Dict[str, Dict[str, Order]] = {}
        self.position: Dict[str, Position] = {} 

        self.trades = []
        self.funding_payments = []
        self.rejected_orders = []
        # Every order ever placed, whatever became of it (see Order.status).
        self.orders: List[Order] = []
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

    def _bars(self, symbol: str, i: int) -> Tuple:
        d = self._d(symbol)
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
              price: float, 
              *, 
              type: str = "LIMIT",
              stop_price: float= None,
              label: str=None) -> Order:       
        """
        :param str symbol: symbol of the market (Mandatory)
        :param float qty: quantity of the trade (negative for short, Mandatory)
        :param float price: price of which to be executed (Mandatory for LIMIT Orders)
        :param str type: 'MARKET' or 'LIMIT' or 'STOP' (By default LIMIT)
        :param float stop_price: stop price for stop orders (Mandatory for STOP orders)
        """
        if not symbol or not isinstance(symbol, str):
            raise InvalidOrderParameters(f"symbol must be a non-empty string. - {symbol}")
        
        if not qty or not isinstance(qty, (int, float, Decimal)):
            raise InvalidOrderParameters(f"qty must be a non-empty number. - {qty}")

        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.maint_margin_rate, self.maint_amount)

        if symbol not in self.open_orders:
            self.open_orders[symbol] = {}

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

        timestamp = pd.to_datetime(self.data.index[self._i])
        order = Order(timestamp, symbol, qty, price, type, stop_price, label)
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
    
    def cancel_order(self, symbol, order_id: str) -> Optional[Order]:
        if symbol not in self.open_orders:
            return None
        
        if order_id not in self.open_orders[symbol]:
            return None

        order = self.open_orders[symbol][order_id]
        del self.open_orders[symbol][order_id]
        self._mark_cancelled(order)

        return order

    def _mark_cancelled(self, order: Order) -> None:
        order.status = Status.CANCELLED
        order.cancelled_at = pd.to_datetime(self.data.index[min(self._i, len(self.data) - 1)])

    def cancel_all_orders(self, symbol: str):
        if symbol not in self.open_orders:
            return None

        orders = self.open_orders[symbol]
        for order_id in list(orders.keys()):
            self._mark_cancelled(orders[order_id])
            del orders[order_id]

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.maint_margin_rate, self.maint_amount)
            
        return self.position[symbol]
    
    def close_all_positions(self, symbol: str):
        position = self.get_position(symbol)
        
        if position.qty != 0: # Open position
            self.order(symbol, -position.qty, None, type="MARKET")      

    def set_leverage(self, symbol: str, leverage: int):        
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

    def unrealized_pnl(self, i: Optional[int] = None) -> float:
        """Unrealized PnL of all open positions, marked at the close of bar `i`."""
        if i is None:
            i = min(self._i, len(self.data) - 1)
        return sum(
            p.unrealized_pnl(self._close(s, i))
            for s, p in self.position.items() if p.qty != 0
        )

    def equity(self, i: Optional[int] = None) -> float:
        """Cash plus unrealized PnL, marked at the close of bar `i`."""
        return self.cash + self.unrealized_pnl(i)

    def _apply_funding(self) -> None:
        """
        Settle funding for positions held into this bar.

        A non-zero value in the `funding_rate` column means funding is paid at
        the open of that bar. Longs pay shorts when the rate is positive.
        """
        for symbol, position in self.position.items():
            if position.qty == 0:
                continue
            rates = getattr(self._d(symbol), FUNDING_COLUMN, None)
            if rates is None:
                continue
            rate = rates[self._i]
            if rate is None or rate != rate or rate == 0:  # rate != rate catches NaN
                continue

            price = self._open(symbol, self._i)
            amount = -float(position.qty) * price * float(rate)
            self.cash += amount
            self.recorder.on_funding(symbol, amount)
            self.funding_payments.append({
                "time": pd.to_datetime(self.data.index[self._i]),
                "symbol": symbol,
                "qty": float(position.qty),
                "price": price,
                "funding_rate": float(rate),
                "amount": amount,
            })

    def _slippage(self, symbol: str, order: Order):
        if self.slippage:
            return self.slippage(order, Bar(*self._bars(symbol, self._i)))

        return order.theoretical_price
    
    def _execute_trade(self, symbol: str, order: Order,  price: float = None, liquidation = False):
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
        executed_time = pd.to_datetime(self.data.index[self._i])

        # Update Order fields as it has been filled
        order.pnl = pnl
        order.executed_at = executed_time
        order.commission = commission
        order.realized_pnl = pnl - commission
        order.is_liquidation_order = bool(liquidation)
        order.status = Status.FILLED
        side = 1 if order.qty > 0 else -1
        order.slippage = side * (order.executed_price - order.theoretical_price) * float(abs(order.qty))
        self.recorder.on_fill(self._i, order, qty_before, pnl, commission, bool(liquidation))

        # Update trades and balance
        self.trades.append(order.to_dict())
        self.cash += order.realized_pnl   

        if symbol in self.open_orders and order.id in self.open_orders[symbol]:
            del self.open_orders[symbol][order.id]

    def next(self):
        if self._i != 0: # Check Liquidation
            for s, p in list(self.position.items()):
                _, prev_high, prev_low, _ = self._bars(s, self._i - 1)
                lp = p.liquid_price
                # Check Liquidation Condition
                if ((p.qty > 0 and prev_low <= lp) or 
                    (p.qty < 0 and prev_high >= lp)):
                    # Create MARKET Order for liquidation & Execute
                    order = self.order(p.symbol,  -p.qty, lp, type="MARKET")
                    self._execute_trade(p.symbol, order, lp, True)

        # Funding is settled at the open, before this bar's orders fill.
        if self.funding:
            self._apply_funding()

        # Execute open orders (iterate snapshots to avoid mutation issues)
        for symbol, by_id in list(self.open_orders.items()):
            open, high, low, _ = self._bars(symbol, self._i)

            for _, order in list(by_id.items()):
                # Set market order price and update
                if order.type == OrderType.MARKET:
                    # Execute the trade with price being
                    # open price because its market order
                    self._execute_trade(symbol, order, open)
                elif order.type == OrderType.LIMIT:
                    if ((order.qty > 0 and low < order.price) or 
                        (order.qty < 0 and high > order.price)):
                        # If the order just has been created
                        if order.created_at == pd.to_datetime(self.data.index[self._i]):
                            # For example)
                            # If open price is 100 and places an limit buy order 110
                            # The order should be executed at 100 not 110
                            if ((order.qty > 0 and order.price > open) or
                                (order.qty < 0 and order.price < open)):
                                self._execute_trade(symbol, order, open)
                                continue
                        
                        self._execute_trade(symbol, order)
                else:
                    # STOP LIMIT, STOP MARKET Trigger Condition:
                    # LONG: Current Price >= Stop Price
                    # SHORT: Current Price <= Stop Price
                    if order.is_triggered == False:
                        # Check for the STOP order Trigger Condition
                        if ((order.qty > 0 and high >= order.stop_price) or 
                            (order.qty < 0 and low <= order.stop_price)):
                            order.is_triggered = True # Trigger the Order

                if order.is_triggered == True:
                    if order.price is None: # STOP MARKET ORDER
                        self._execute_trade(symbol, order, order.stop_price)
                    else: # STOP LIMIT ORDER
                        if ((order.qty > 0 and low < order.price) or 
                            (order.qty < 0 and high > order.price)):
                            # If the order just has been created
                            if order.created_at == pd.to_datetime(self.data.index[self._i]):
                                # For example)
                                # If open price is 100 and places an limit buy order 110
                                # The order should be executed at 100 not 110
                                if ((order.qty > 0 and order.price > open) or
                                    (order.qty < 0 and order.price < open)):
                                    self._execute_trade(symbol, order, open)
                                    continue
                            
                            self._execute_trade(symbol, order)

        self.recorder.on_bar_end(self._i, self)
        self._i += 1