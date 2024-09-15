import abc
from typing import List, Dict
from overfitting.order import Order
from overfitting.position import Position

class Broker:
    def __init__(self,
                 data,
                 fx_slippage=0,
                 linear_slippage=0,
                 fx_commission=0,
                 linear_commission=0):
        
        self.data = data
        self.fx_slippage = fx_slippage
        self.linear_slippage = linear_slippage
        self.fx_commission = fx_commission
        self.linear_commission = linear_commission

        self.open_orders: List[Order] = []
        self.new_orders: List[Order] = []
        self.orders: Dict[Order] = {}
        self.position: Dict[Position] = {}

    def __repr__(self):
        return """
{class_name}(
    data={data},
    fx_slippage={fx_slippage},
    linear_slippage={linear_slippage},
    fx_commission={fx_commission},
    linear_commission={linear_commission},
    open_orders={open_orders},
    new_orders={new_orders},
    orders={orders},
    position={position})
""".strip().format(class_name=self.__class__.__name__,
                   data=self.data,
                   fx_slippage=self.fx_slippage,
                   linear_slippage=self.linear_slippage,
                   fx_commission=self.fx_commission,
                   linear_commission=self.linear_commission,
                   open_orders=self.open_orders,
                   new_orders=self.new_orders,
                   orders=self.orders,
                   position=self.position)