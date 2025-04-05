import abc
import pandas as pd
from typing import List, Dict, Optional
from overfitting.order import Order
from overfitting.position import Position
from overfitting.functions.type import TYPE

class Broker:
    def __init__(self, data, cash, commission_rate, slippage_rate):
        self.data = data
        self.initial_captial = cash
        self.cash = self.initial_captial
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.open_orders: List[Order] = []
        self.position: Dict[str, Position] = {} 

        self.trades = []
        self._i = 0

    def __repr__(self):
        return """
{class_name}(
    cash={cash},
    initial_captial={initial_captial},
    commission_rate={commissino_rate},
    slippage_rate={slippage_rate},
    open_orders={open_orders},
    order_history={order_history},
    position={position})
""".strip().format(class_name=self.__class__.__name__,
                   cash=self.cash,
                   initial_captial=self.initial_captial,
                   commission_rate=self.commission_rate,
                   slippage_rate=self.slippage_rate,
                   open_orders=self.open_orders,
                   order_history=self.order_history,
                   position=self.position)
    
    def order(self, symbol: str, qty: float, price: float, *, type: Optional[str]):             
        # Initialize Position Dict if necessary
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)

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
        p = self.data.Open[self._i]

        if (position.qty > 0 and p <= lp) or \
           (position.qty < 0 and p >= lp):
            raise Exception(f"Cannot change leverage for {symbol}. Position would be liquidated at price {lp}.")

    def _execute_trade(self, symbol: str, order: Order,  price: float = None):
        if price:
            order.price = price
        
        pnl = self.position[symbol].update(order)
        notional = abs(order.qty) * order.price
        commission = notional * self.commission_rate

        order.fill(commission, pnl)
        
        # Update trades and balance
        self.trades.append(order.to_dict())
        self.cash += order.realized_pnl        
        # Remove the filled Order
        self.open_orders.remove(order)

    def next(self):
        data = self.data
        open, high, low = data.open[self._i], data.high[self._i], data.low[self._i]

        if self._i != 0:
            prev_high = data.high[self._i - 1]
            prev_low  = data.low[self._i - 1]

            # Check for the liquidation price
            for _, s in enumerate(self.position):
                p = self.position[s] # position
                lp = p.liquid_price # lp price
                # Check conditions for liquidation
                if ((p.qty > 0 and prev_low <= lp) or 
                    (p.qty < 0 and prev_high >= lp)):
                    margin = p.liquidate()
                    self.cash -= margin
                    
                    
                    # notional = abs(order.qty) * lp
                    # commission = notional * self.commission_rate
                    # order.fill(commission, )
                    
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

        # Lastly update index
        self._i += 1
                
    
    
    

        
        