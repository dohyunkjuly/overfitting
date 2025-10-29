import uuid
import pandas as pd
from overfitting.utils.entities import OrderType, Status

class Order:
    def __init__(self, 
                 time: pd.Timestamp, 
                 symbol: str, 
                 qty: float, 
                 price:float, 
                 type: OrderType, 
                 stop_price: float =None):
        
        self.id = self.make_id()
        self.created_at = time
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.type = type
        self._status = Status.OPEN
        self.stop_price = stop_price
        self.is_triggered = False
        self.reason = None
        self.executed_price = 0
        self.commission = 0
        self.pnl = 0
        self.realized_pnl = 0
        
    def __repr__(self):
        return (f"Order(id={self.id}, created_at={self.created_at}, "
                f"symbol='{self.symbol}', qty={self.qty}, price={self.price}, "
                f"type={self.type}, status={self._status}, "
                f"stop_price={self.stop_price}, is_triggered={self.is_triggered}"
                f"reason='{self.reason}', executed_price={self.executed_price}"
                f"commission={self.commission}, pnl={self.pnl}")

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

    def fill(self, commission, pnl, executed_price, reason=None):
        self.status = Status.FILLED
        self.commission = commission
        self.pnl = pnl
        realized = pnl - commission
        self.realized_pnl = realized
        self.executed_price = executed_price
        self.reason = reason
        