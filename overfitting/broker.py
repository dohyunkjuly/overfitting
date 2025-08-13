import abc
import pandas as pd
from typing import List, Dict
from overfitting.order import Order
from overfitting.position import Position
from overfitting.functions.type import OrderType, Status
from overfitting.functions.data import Data
from overfitting.error import EmptyOrderParameters, InvalidOrderParameters

class Broker:
    def __init__(self,
                 data: Data, 
                 cash: float, 
                 commission_rate: float, 
                 maint_maring_rate: float, 
                 maint_amount:float):
        self.data = data
        self.initial_captial = cash
        self.cash = self.initial_captial
        self.commission_rate = commission_rate
        self.maint_maring_rate = maint_maring_rate
        self.maint_amount = maint_amount

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

    def order(self, symbol: str, qty: float, price: float, *, type: str, stop_price: float= None):       
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
            # By Default LIMIT ORDER
            type = OrderType.LIMIT
        
        if type == OrderType.STOP and stop_price is None:
            raise EmptyOrderParameters("stop_price must be specficed for STOP order")
        
        if type == OrderType.LIMIT and price is None:
            raise EmptyOrderParameters("price must be specifed for LIMIT order")

        timestamp = pd.to_datetime(self.data['timestamp'][self._i])
        order = Order(timestamp, symbol, qty, price, type, stop_price)

        # Put new order in the open_orders list
        self.open_orders.append(order)
        return order
    
    def get_position(self, symbol):
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)
            
        return self.position[symbol]
    
    def set_leverage(self, symbol, leverage):
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)
        
        position = self.position[symbol]
        position.set_leverage(leverage)
        # Check if the position would be liquidated with the new leverage
        lp = position.liquid_price
        p = self.data.open[self._i]

        if (position.qty > 0 and p <= lp) or \
           (position.qty < 0 and p >= lp):
            raise Exception(f"Cannot change leverage for {symbol}. Position would be liquidated at price {lp}.")

    def _execute_trade(self, symbol: str, order: Order,  price: float = None, liquidation = False):
        if price: # For Market Orders or Liquidation Orders
            order.price = price
        
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

    def _reject_order(self, order: Order, reason: str):
        """
        Reject order when stop order would be immedately triggered
        """
        order.reject(reason)
        self.trades.append(order.to_dict())
        self.open_orders.remove(order)

    def next(self):
        data = self.data
        open, high, low = data.open[self._i], data.high[self._i], data.low[self._i]

        # Check Liquidation
        if self._i != 0:
            prev_high = data.high[self._i - 1]
            prev_low  = data.low[self._i - 1]
            for _, s in enumerate(self.position):
                p = self.position[s] 
                lp = p.liquid_price
                # Check Liquidation Condition
                if ((p.qty > 0 and prev_low <= lp) or 
                    (p.qty < 0 and prev_high >= lp)):

                    # Create Order for liquidation & Execute
                    type = 'MARKET' # Market Order for Liquidation Order
                    order = self.order(p.symbol,  -p.qty, lp, type=type)
                    self._execute_trade(p.symbol, order, lp, True)
                        
        # Iterate over a shallow copy of the list
        for order in self.open_orders[:]:
            symbol = order.symbol
            # Set market order price and update
            if order.type == OrderType.MARKET:
                # Execute the trade with price being
                # open price because its market order
                self._execute_trade(symbol, order, open)
            elif order.type == OrderType.LIMIT:
                if ((order.qty > 0 and high > order.price) or 
                    (order.qty < 0 and low < order.price)):
                    self._execute_trade(symbol, order)
            else:
                # STOP LIMIT, STOP MARKET Trigger Condition:
                # LONG: Current Price >= Stop Price
                # SHORT: Current Price <= Stop Price
                if pd.to_datetime(data.timestamp[self._i]) == order.created_at:
                    # Check if the STOP order has been created recently
                    if ((order.qty > 0 and order.stop_price < open) or
                        (order.qty < 0 and order.stop_price > open)):
                        # this would trigger the stop order immediately
                        self._reject_order(order, "STOP order would Immedately Trigger")
        
                if order.is_triggered == False:
                    # Check for the STOP order Trigger Condition
                    if ((order.qty > 0 and high > order.stop_price) or 
                        (order.qty < 0 and low < order.stop_price)):
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
                
    
    
    

        
        