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

    def _calculate_commission(self, order: Order) -> float:
        return abs(order.qty) * order.price * self.commission_rate

    def next(self):
        data = self.data
        open, high, low = data.open[self._i], data.high[self._i], data.low[self._i]


        if self._i != 0:
            prev_high = data.high[self._i - 1]
            prev_low  = data.low[self._i - 1]

            # Check for the liquidation price
            for _, s in enumerate(self.position):
                p = self.position[s]
                lp = p.liquid_price
                # Check conditions for liquidation

                ### TO DO ###
                # 1. Debug the liquidation logic...

                if ((p.qty > 0 and prev_low <= lp) or 
                    (p.qty < 0 and prev_high >= lp)):
                    print('liquidation')
                    self.cash += p.liquidate()
                    
        # Iterate over a shallow copy of the list
        for order in self.open_orders[:]:
            symbol = order.symbol
            pnl = None 

            # Set market order price and update
            if order.type == TYPE.market:
                order.price = open  # Set Entry Price for Market Order
                pnl = self.position[symbol].update(order)
                comission = self._calculate_commission(order)

                order.fill(comission, pnl)
            else:
                # Check conditions for limit orders
                if order.qty > 0 and high > order.price:  # Long condition
                    pnl = self.position[symbol].update(order)
                    comission = self._calculate_commission(order)

                    order.fill(comission, pnl)

                elif order.qty < 0 and low < order.price:  # Short condition
                    pnl = self.position[symbol].update(order)
                    comission = self._calculate_commission(order)

                    order.fill(comission, pnl)

            if pnl is not None:
                # If pnl is not None, the order was executed
                self.trades.append(order.to_dict())
                self.cash += order.realized_pnl
                
                # Remove the filled Order
                self.open_orders.remove(order)
        

        # Lastly update index
        self._i += 1
                
    
    
    

        
        