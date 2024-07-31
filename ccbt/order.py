import math
import uuid

from ccbt.functions.type import enum

TYPE = enum('tp', 'sl', 'limit', 'market')
DIRECTION = enum('buy', 'sell')
STATUS = enum(
    'OPEN', 
    'CANCELLED', 
    'FILLED', 
    'REJECTED'
)

class Order:
    __slots__ = ['id', 'symbol', 'qty', 'price', 'type', 'direction'
                 'status', 'stop_price', 'trailing_delta']

    def __init__(self, symbol, qty, price, type, direction, 
                 stop_price=None, trailing_delta=None):
        
        # Convert to lowercase to ensure case-insensitivity
        type = type.lower()
        direction = direction.lower()

        try:
            self.type = TYPE[type]
        except KeyError:
            raise ValueError('type must be an instance of Type Enum')
        
        try:
            self.direction = DIRECTION[direction]
        except KeyError:
            raise ValueError('direction must be an instance of Direction Enum')
        
        self.id = self.make_id()
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.type = type
        self.direction = direction
        self.status = STATUS.OPEN
        self.stop_price = stop_price
        self.trailing_delta = trailing_delta
        self.reason = None
    
    @staticmethod
    def make_id():
        return uuid.uuid4().hex

    def cancel(self):
        self.status = STATUS.CANCELLED
    
    def rejected(self, reason=''):
        self.status = STATUS.REJECTED
        self.reason = reason
    
