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
        
        # Log for the trades
        self.trades= []
        self.trade_number = 0
    
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
              timestamp: pd.Timestamp, 
              symbol: str, 
              qty: float, 
              price: float, 
              direction: str, 
              *, 
              type: Optional[str], 
              stop: Optional[float], 
              trailing: Optional[float]):             
        
        # Initialize Position Dict
        if symbol not in self.position:
            self.position[symbol] = Position(symbol)

        order = Order(timestamp, symbol, qty, price, direction, type=type)
        # Put new order in the open_orders List
        self.open_orders.append(order)
        return order
    
    def next(self):
        
        
        
    

        
        