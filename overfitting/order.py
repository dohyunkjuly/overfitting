import uuid
import pandas as pd
from overfitting.types import OrderType

class Order:
    def __init__(self, 
                 time: pd.Timestamp, 
                 symbol: str, 
                 qty: float, 
                 price:float, 
                 type: OrderType, 
                 stop_price: float = None,
                 label: str= None):
        
        self.id = uuid.uuid4().hex[:16]
        self.created_at = time
        self.executed_at = None
        self.symbol = symbol
        self.side = "LONG" if qty > 0 else "SHORT"
        self.qty = qty
        self.price = price
        self.type = type
        self.stop_price = stop_price
        self.is_triggered = False
        self.is_liquidation_order = False
        self.theoretical_price = 0
        self.executed_price = 0
        self.commission = 0
        self.pnl = 0
        self.realized_pnl = 0
        self.label = label

    def __repr__(self):
        return (f"Order(id={self.id}"
                f"created_at={self.created_at}, "
                f"exeucted_at={self.executed_at}, "
                f"symbol={self.symbol}, "
                f"side={self.side}, "
                f"qty={self.qty}, "
                f"price={self.price}, "
                f"type={self.type}, "
                f"stop_price={self.stop_price}, "
                f"is_triggered={self.is_triggered}, "
                f"theoretical_price={self.theoretical_price}, "
                f"executed_price={self.executed_price}, "
                f"commission={self.commission}, "
                f"pnl={self.pnl}, "
                f"realized_pnl={self.realized_pnl}, "
                f"label={self.label})")

    def to_dict(self):
        return {
            k: (v.name if isinstance(v, OrderType) else v)
            for k, v in self.__dict__.items()
        }

        