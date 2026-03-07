from enum import Enum

class OrderType(Enum):
    LIMIT = 0
    MARKET = 1
    STOP = 2

class Status(Enum):
    OPEN = 0
    CANCELLED = 1
    FILLED = 2
    REJECTED = 3