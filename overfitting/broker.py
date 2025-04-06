import abc
import pandas as pd
from typing import List, Dict, Optional
from overfitting.order import Order
from overfitting.position import Position
from overfitting.functions.type import TYPE

class Broker:
    def __init__(self, data, cash, commission_rate, maint_maring_rate, maint_amount):
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

    def order(self, symbol: str, qty: float, price: float, *, type: Optional[str]):             
        # Initialize Position Dict if necessary
        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.maint_maring_rate, self.maint_amount)

        timestamp = pd.to_datetime(self.data['timestamp'][self._i])
        order = Order(timestamp, symbol, qty, price, type)

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
            raise Exception("Cannot Exeucte Orders without Price.")
        
        notional = abs(order.qty) * order.price
        commission = notional * self.commission_rate
        position = self.position[symbol]
        pnl = position.update(order, liquidation)

        status = 'liquidation' if liquidation else None
        order.fill(commission, pnl, status)

        # Update trades and balance
        self.trades.append(order.to_dict())
        self.cash += order.realized_pnl   
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
                    type = 'market' # Market Order for Liquidation Order
                    order = self.order(p.symbol,  -p.qty, lp, type=type)
                    self._execute_trade(p.symbol, order, lp, True)
                    
        # Iterate over a shallow copy of the list
        for order in self.open_orders[:]:
            symbol = order.symbol
            # Set market order price and update
            if order.type == TYPE.market:
                # Execute the trade with price being
                # open price because its market order
                self._execute_trade(symbol, order, open)
            else:
                # Check conditions for limit orders
                if order.qty > 0 and high > order.price:  # Long condition
                    self._execute_trade(symbol, order)
                elif order.qty < 0 and low < order.price:  # Short condition
                    self._execute_trade(symbol, order)

        self._i += 1
                
    
    
    

        
        