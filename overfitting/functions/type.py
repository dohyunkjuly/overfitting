from enum import Enum

class OrderType(Enum):
    TP = 0
    SL = 1
    LIMIT = 2
    MARKET = 3

class Status(Enum):
    OPEN = 0
    CANCELLED = 1
    FILLED = 2
    REJECTED = 3