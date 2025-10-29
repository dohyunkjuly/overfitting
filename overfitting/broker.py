import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from overfitting.data import Data, MultiCurrency
from overfitting.order import Order
from overfitting.position import Position
from overfitting.execution.slippage import SlippageModel
from overfitting.utils.entities import OrderType
from overfitting.utils.error import EmptyOrderParameters, InvalidOrderParameters, LiquidationError

class Broker:
    def __init__(self,
                 data: Union[Data, MultiCurrency], 
                 cash: float, 
                 commission_rate: float, 
                 maint_maring_rate: float, 
                 maint_amount:float,
                 slippage_model:SlippageModel | None):
        
        self.data = data
        self.initial_captial = cash
        self.cash = self.initial_captial
        self.commission_rate = commission_rate
        self.maint_maring_rate = maint_maring_rate
        self.maint_amount = maint_amount
        self.slippage_model= slippage_model

        self.open_orders: List[Order] = []
        self.position: Dict[str, Position] = {} 

        self.trades = []
        self._i = 0

    def __repr__(self):
        return (f"Broker("
                f"initial_capital={self.initial_captial}, "
                f"cash={self.cash}, "
                f"commission_rate={self.commission_rate}, "
                f"maint_margin_rate={self.maint_maring_rate}, "
                f"maint_amount={self.maint_amount}, "
                f"open_orders={len(self.open_orders)}, "
                f"positions={list(self.position.keys())}, "
                f"trades={len(self.trades)})")
    
    def _d(self, symbol: str) -> Data:
        return self.data[symbol] if isinstance(self.data, MultiCurrency) else self.data

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
              type: str= OrderType.LIMIT, 
              stop_price: float= None) -> Order:       
        """
        :param str symbol: symbol of the market (Mandatory)
        :param float qty: quantity of the trade (negative for short, Mandatory)
        :param float price: price of which to be executed (Mandatory for LIMIT Orders)
        :param str type: 'MARKET' or 'LIMIT' or 'STOP' (By default LIMIT)
        :param float stop_price: stop price for stop orders (Mandatory for STOP orders)
        """
        if not symbol or not isinstance(symbol, str):
            raise InvalidOrderParameters(f"symbol must be a non-empty string. - {symbol}")
        
        if not qty or not isinstance(qty, (int, float)):
            raise InvalidOrderParameters(f"qty must be a non-empty float. - {qty}")

        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.maint_maring_rate, self.maint_amount)

        if type.upper() == "LIMIT":
            type = OrderType.LIMIT
        elif type.upper() == "MARKET":
            type = OrderType.MARKET
        elif type.upper() == 'STOP':
            type = OrderType.STOP
        else:
            raise InvalidOrderParameters(f"Invalid Order Type - {type}")
        
        if type == OrderType.STOP and stop_price is None:
            raise EmptyOrderParameters("stop_price must be specficed for STOP order")
        
        if type == OrderType.LIMIT and price is None:
            raise EmptyOrderParameters("price must be specifed for LIMIT order")

        timestamp = pd.to_datetime(self.data.index[self._i])
        order = Order(timestamp, symbol, qty, price, type, stop_price)

        if type == OrderType.STOP:
            open = self._open(symbol, self._i)
            if ((order.qty > 0 and order.stop_price < open) or
                (order.qty < 0 and order.stop_price > open)):
                # Check if order would be triggered immedately. If True, reject.
                order.reject("STOP order would Immedately Trigger")
                self.trades.append(order.to_dict())
                return order
    
        # Put new order in the open_orders list if not rejected
        self.open_orders.append(order)
        return order
    
    def cancel_order(self, order_id: str, reason: str = None) -> Optional[Order]:
        for i, o in enumerate(self.open_orders):
            if o.id == order_id:
                o.cancel(reason)
                return self.open_orders.pop(i)
        return None

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)
            
        return self.position[symbol]
    
    def set_leverage(self, symbol: str, leverage: int):
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)
        
        position = self.position[symbol]
        position.set_leverage(leverage)
        # Check if the position would be liquidated with the new leverage
        lp = position.liquid_price
        p = self._open(symbol, self._i)

        if (position.qty > 0 and p <= lp) or \
           (position.qty < 0 and p >= lp):
            raise LiquidationError(
                f"Cannot change leverage for {symbol}. Position would be liquidated at price {lp}.")

    def _slippage(self, symbol: str, order: Order):
        if self.slippage_model:
            bars = self._bars(symbol, self._i)
            self.slippage_model.set_context(order, bars)
            return self.slippage_model.compute()
        
        return order.price 
    
    def _execute_trade(self, symbol: str, order: Order,  price: float = None, liquidation = False):
        if price: # For Market Orders or Liquidation Orders
            order.price = price
        # Lastly Check for slippage    
        order.price = self._slippage(symbol, order)

        if not order.price:
            raise EmptyOrderParameters("Cannot Exeucte Orders without Price.")
        
        notional = abs(order.qty) * order.price
        commission = notional * self.commission_rate
        position = self.position[symbol]
        pnl = position.update(order, liquidation)

        reason = 'liquidation' if liquidation else None
        order.fill(commission, pnl, order.price ,reason)

        # Update trades and balance
        self.trades.append(order.to_dict())
        self.cash += order.realized_pnl   
        self.open_orders.remove(order)

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
                        
        # Iterate over a shallow copy of the list
        for order in self.open_orders[:]:
            symbol = order.symbol
            open, high, low, _ = self._bars(symbol, self._i)
            # Set market order price and update
            if order.type == OrderType.MARKET:
                # Execute the trade with price being
                # open price because its market order
                self._execute_trade(symbol, order, open)
            elif order.type == OrderType.LIMIT:
                if ((order.qty > 0 and low < order.price) or 
                    (order.qty < 0 and high > order.price)):
                    self._execute_trade(symbol, order)
            else:
                # STOP LIMIT, STOP MARKET Trigger Condition:
                # LONG: Current Price >= Stop Price
                # SHORT: Current Price <= Stop Price
                if order.is_triggered == False:
                    # Check for the STOP order Trigger Condition
                    if ((order.qty > 0 and high >= order.stop_price) or 
                        (order.qty < 0 and low <= order.stop_price)):
                        order.trigger() # Trigger the Order

            if order.is_triggered == True:
                if order.price is None:
                    # STOP MARKET ORDER
                    self._execute_trade(symbol, order, open)
                else: # STOP LIMIT ORDER
                    if ((order.qty > 0 and high > order.price) or 
                        (order.qty < 0 and low < order.price)):
                        self._execute_trade(symbol, order)
        
        self._i += 1