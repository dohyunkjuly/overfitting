import uuid
from decimal import Decimal
from enum import Enum

import pandas as pd


class OrderType(Enum):
    LIMIT = 0
    MARKET = 1
    STOP = 2


class Status(Enum):
    OPEN = 0
    FILLED = 1
    REJECTED = 2
    CANCELLED = 3


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
        self.cancelled_at = None
        self.status = Status.OPEN
        self.reject_reason = None
        self.symbol = symbol
        # Decimal qty so Position bookkeeping is exact
        self.qty = Decimal(str(qty))
        self.side = "LONG" if self.qty > 0 else "SHORT"
        self.price = price
        self.type = type
        self.stop_price = stop_price
        self.is_triggered = False
        self.is_liquidation_order = False
        self.theoretical_price = 0
        self.executed_price = 0
        # What slippage cost on this fill, in currency (negative if it helped).
        self.slippage = 0
        self.commission = 0
        self.pnl = 0
        self.realized_pnl = 0
        self.label = label

    def __repr__(self):
        return (f"Order(id={self.id}, "
                f"created_at={self.created_at}, "
                f"executed_at={self.executed_at}, "
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
            k: (
                v.name if isinstance(v, Enum)
                else float(v) if isinstance(v, Decimal)
                else v
            )
            for k, v in self.__dict__.items()
        }

        