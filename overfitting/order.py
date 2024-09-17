import math
import uuid
from overfitting.functions.type import enum
from overfitting.error import InvalidOrder

TYPE = enum('tp', 'sl', 'limit', 'market')
STATUS = enum(
    'OPEN', 
    'CANCELLED', 
    'FILLED', 
    'REJECTED'
)

class Order:
    __slots__ = ['id', 'created_at', 'symbol', 'qty', 'price', 'type', '_status', 
                 'stop_price', 'trailing_delta', 'is_triggered','reason']

    def __init__(self, time, symbol, qty, price, type,
                 stop_price=None, trailing_delta=None):
        
        """@time: Pandas Date Time"""
        type = type.lower()

        self.created_at = time
        self.type = self._get_enum_value(TYPE, type, 'type')

        self.id = self.make_id()
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self._status = STATUS.OPEN
        self.stop_price = stop_price
        self.trailing_delta = trailing_delta
        self.is_triggered = False
        self.reason = None
    
        # Check Conditions
        # TO do List
        # 1. Debug _check_trigger_conditions()

    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @staticmethod
    def _get_enum_value(enum_type, value, name):
        try:
            return enum_type[value]
        except KeyError:
            raise ValueError(f'Invalid {name} value: {value}')

    @staticmethod
    def make_id():
        return uuid.uuid4().hex

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def cancel(self):
        self.status = STATUS.CANCELLED

    def fill(self):
        self.status = STATUS.FILLED

    def rejected(self, reason=''):
        self.status = STATUS.REJECTED
        self.reason = reason

    def check_trigger_status(self, updated_price):
        if self.qty > 0: # qty > 0 == Long
            if self.type == 'tp' and updated_price > self.price:
                self.is_triggered = True
            elif self.type =='sl' and updated_price < self.price:
                self.is_triggered = True
        else:
            if self.type == 'tp' and updated_price < self.price:
                self.is_triggered = True
            elif self.type =='sl' and updated_price > self.price:
                self.is_triggered = True

        return self.is_triggered

    def _check_trigger_conditions(self):
        """
        Checks whether the order conditions (stop and limit prices) are valid.
        Raises an InvalidOrder exception if any condition is invalid.
        """
        if self.stop_price is not None or self.trailing_delta is not None:
            return
        
        flag = 0
        if self.qty > 0:
            if self.type == 'tp':
                if self.stop_price < self.price:
                    flag = 1
            else: 
                # when type is stop loss
                if self.stop_price > self.price:
                    flag = 1
        else:
            # when direction is sell
            if self.type == 'tp':
                if self.stop_price > self.price:
                    flag = 1
            else:
                if self.stop_price < self.price:
                    flag = 1 
        if flag == 1:
            raise InvalidOrder('Invalid Conditional Order')
        