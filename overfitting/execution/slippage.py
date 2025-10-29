from abc import ABC, abstractmethod
from typing import Tuple
from overfitting.order import Order

class SlippageModel(ABC):
    def set_context(self, order:Order, bar: Tuple) -> None:
        self.order = order
        self.o, self.h, self.l, self.c = bar

    @abstractmethod
    def compute(self) -> float:
        pass

class Zero(SlippageModel):
    def compute(self) -> float:
        return self.order.price

class Skewed(SlippageModel):
    def __init__(self, f: float = 0.15):
        if not (0.0 <= f <= 1.0):
            raise ValueError("f must be between 0 to 1.")
        self.f = f

    def compute(self) -> float:
        if self.order.qty > 0:  # long
            return self.o + self.f * max(0.0, self.h - self.o)
        else:                   # short
            return self.o - self.f * max(0.0, self.o - self.l)
            
class FixedPercent(SlippageModel):
    def __init__(self, f: float = 0.01):
        if f < 0:
            raise ValueError("f must be >= 0.")
        self.f = f
    
    def compute(self) -> float:
        bump = (1 + self.f) if self.order.qty > 0 else (1 - self.f)
        return bump * self.order.price
    

class Slippage:
    @staticmethod
    def Zero() -> SlippageModel:
        return Zero()

    @staticmethod
    def Skewed(f: float = 0.15) -> SlippageModel:
        return Skewed(f=f)

    @staticmethod
    def FixedPercent(f: float = 0.01) -> SlippageModel:
        return FixedPercent(f=f)