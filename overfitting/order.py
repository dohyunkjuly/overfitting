import math
import uuid
import pandas as pd
from overfitting.functions.type import OrderType, Status
from overfitting.error import InvalidOrderType

class Order:
    __slots__ = ['id', 'created_at', 'symbol', 'qty', 'price', 'type', '_status', 
                 'stop_price', 'trailing_delta', 'is_triggered','reason',
                 'commission', 'pnl', 'realized_pnl']

    def __init__(self, 
                 time: pd.Timestamp, 
                 symbol: str, 
                 qty: float, 
                 price:float,
                 type: OrderType,
                 stop_price: float =None, 
                 trailing_delta:float =None):
        
        self.id = self.make_id()
        self.created_at = time
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.type = type
        self._status = Status.OPEN
        self.stop_price = stop_price
        self.trailing_delta = trailing_delta
        self.is_triggered = False
        self.reason = None
        self.commission = 0
        self.pnl = 0
        self.realized_pnl = 0
        
    def __repr__(self):
        return (f"Order(id={self.id}, created_at={self.created_at}, "
                f"symbol='{self.symbol}', qty={self.qty}, price={self.price}, "
                f"type={self.type}, status={self._status}, "
                f"stop_price={self.stop_price}, "
                f"trailing_delta={self.trailing_delta}, "
                f"is_triggered={self.is_triggered}, reason='{self.reason}')"
                f"commission={self.commission}, pnl={self.pnl}")

    def to_dict(self):
        result = {}
        for slot in self.__slots__:
            value = getattr(self, slot)

            if slot == 'type' and hasattr(value, 'name'):
                result['type'] = value.name.upper()
            elif slot == '_status' and hasattr(value, 'name'):
                result['status'] = value.name.upper()
            elif slot != '_status':  # Skip _status itself since we renamed it
                result[slot] = value

        return result
    
    @staticmethod
    def make_id():
        return uuid.uuid4().hex[:16]

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def cancel(self):
        self.status = Status.CANCELLED

    def fill(self, commission, pnl, reason=None):
        self.status = Status.FILLED
        self.commission = commission
        self.pnl = pnl
        realized = pnl - commission
        self.realized_pnl = realized
        self.reason = reason

    def rejected(self, reason=''):
        self.status = Status.REJECTED
        self.reason = reason

    # def check_trigger_status(self, updated_price):
    #     if self.qty > 0: # qty > 0 == Long
    #         if self.type == 'tp' and updated_price > self.price:
    #             self.is_triggered = True
    #         elif self.type =='sl' and updated_price < self.price:
    #             self.is_triggered = True
    #     else:
    #         if self.type == 'tp' and updated_price < self.price:
    #             self.is_triggered = True
    #         elif self.type =='sl' and updated_price > self.price:
    #             self.is_triggered = True

    #     return self.is_triggered

    # def _check_trigger_conditions(self):
    #     """
    #     Checks whether the order conditions (stop and limit prices) are valid.
    #     Raises an InvalidOrder exception if any condition is invalid.
    #     """
    #     if self.stop_price is not None or self.trailing_delta is not None:
    #         return
        
    #     flag = 0
    #     if self.qty > 0:
    #         if self.type == 'tp':
    #             if self.stop_price < self.price:
    #                 flag = 1
    #         else: 
    #             # when type is stop loss
    #             if self.stop_price > self.price:
    #                 flag = 1
    #     else:
    #         # when direction is sell
    #         if self.type == 'tp':
    #             if self.stop_price > self.price:
    #                 flag = 1
    #         else:
    #             if self.stop_price < self.price:
    #                 flag = 1 
    #     if flag == 1:
    #         raise InvalidOrderType('Invalid Conditional Order')
        