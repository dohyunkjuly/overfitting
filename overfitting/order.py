import uuid
import pandas as pd
from overfitting.entities.order import OrderType, Status

class Order:
    def __init__(self, 
                 time: pd.Timestamp, 
                 symbol: str, 
                 qty: float, 
                 price:float, 
                 type: OrderType, 
                 stop_price: float = None,
                 label: str= None):
        
        self.id = self.make_id()
        self.created_at = time
        self.executed_at = None
        self.symbol = symbol
        self.side = "LONG" if qty > 0 else "SHORT"
        self.qty = qty
        self.price = price
        self.type = type
        self._status = Status.OPEN
        self.stop_price = stop_price
        self.is_triggered = False
        self.reason = None
        self.theoretical_price = 0
        self.executed_price = 0
        self.commission = 0
        self.pnl = 0
        self.realized_pnl = 0
        self.label = label

    def __repr__(self):
        return (f"Order(id={self.id}, "
                f"created_at={self.created_at}, "
                f"exeucted_at={self.executed_at}, "
                f"symbol={self.symbol}, "
                f"side={self.side}, "
                f"qty={self.qty}, "
                f"price={self.price}, "
                f"type={self.type}, "
                f"status={self._status}, "
                f"stop_price={self.stop_price}, "
                f"is_triggered={self.is_triggered}, "
                f"reason={self.reason}, "
                f"theoretical_price={self.theoretical_price}, "
                f"executed_price={self.executed_price}, "
                f"commission={self.commission}, "
                f"pnl={self.pnl}, "
                f"realized_pnl={self.realized_pnl}, "
                f"label={self.label})")

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if k == "_status":
                out["status"] = v.name
            elif k == "type":
                out["type"] = v.name
            else:
                out[k] = v
        return out
    
    @staticmethod
    def make_id():
        return uuid.uuid4().hex[:16]

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def trigger(self):
        self.is_triggered = True

    def cancel(self, reason=None):
        self.status = Status.CANCELLED
        self.reason = reason
    
    def reject(self, reason=None):
        self.status = Status.REJECTED
        self.reason = reason

    def fill(self, commission, pnl, executed_price, time, reason=None):
        self.status = Status.FILLED
        self.executed_at = time
        self.commission = commission
        self.pnl = pnl
        realized = pnl - commission
        self.realized_pnl = realized
        self.executed_price = executed_price
        self.reason = reason
        