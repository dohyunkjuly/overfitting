import abc
import pandas as pd
from typing import List, Dict, Optional
from overfitting.order import Order
from overfitting.position import Position

class Broker:
    def __init__(self, data, cash, commission_rate, slippage_rate):
        self.data = data
        self.initial_captial = cash
        self.cash = self.initial_captial
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.open_orders: List[Order] = []
        self.position: Dict[Position] = {}
        
        self.trades = []
        self.trade_number = 0
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
    
    def order(self, 
              symbol: str, 
              qty: float, 
              price: float, 
              *, 
              type: Optional[str], 
              stop: Optional[float], 
              trailing: Optional[float]):             
        
        # Initialize Position Dict
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)

        timestamp = pd.to_datetime(self.data['timestamp'][self._i])
        order = Order(timestamp, symbol, qty, price, type=type)
        # Put new order in the open_orders List
        self.open_orders.append(order)
        return order
    
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
            raise Exception(f"Cannot change leverage for {symbol}. 
                            Position would be liquidated at price {lp}.")

    def next(self):
        data = self.data
        open, high, low = data.Open[self._i], data.High[self._i], data.Low[self._i]

        if self._i != 0:
            prev_high = data.High[self._i - 1]
            prev_low  = data.Low[self._i - 1]

            #Check for the liquidation price
            for _, s in enumerate(self.position):
                p = self.position[s]
                lp = p.liquid_price
                # Check conditions for liquidation
                if ((p.qty > 0 and prev_low <= lp) or 
                    (p.qty < 0 and prev_high >= lp)):

                    self.cash += p.liquidate()
                    
        # Check for orders 
        for order in self.open_orders:
            symbol = order.symbol
            price = order.price
            # Check for market order
            if order.type == 'market':
                # Determin the Entry Price
                self.position[symbol].price = open
                pnl = self.position[symbol].update(order)
                order.fill()
            else:
                # Check for the condition
                if order.qty > 0 and high > price:
                    pnl = self.position[symbol].update(order)
                    order.fill()
                elif order.qty < 0 and low < price:
                    pnl = self.position[symbol].update(order)
                    order.fill()

            if pnl:
                commission = self.commission_rate * 2 * order.qty
                self.cash += pnl - commission
                
        # Lastly update index
        self._i += 1
                
    
        
    

        
        