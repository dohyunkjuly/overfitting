"""
Slippage: how far the executed price of a fill is from the price the order asked for.

A slippage model is a function

    model(order, bar) -> executed price

where `order.theoretical_price` is the price before slippage, `order.qty` is
positive for a buy and negative for a sell, and `bar` has the `open`, `high`,
`low` and `close` of the bar the order fills on.

Pass one to a strategy as `slippage=`:

    MyStrategy(data, slippage=0.001)                    # 0.1% worse on every fill
    MyStrategy(data, slippage=skewed(0.15))             # a built-in model
    MyStrategy(data, slippage=lambda order, bar: ...)   # your own
"""
from __future__ import annotations

import numbers
from typing import Callable, NamedTuple, Optional

__all__ = ["Bar", "fixed_percent", "skewed", "resolve"]


class Bar(NamedTuple):
    open: float
    high: float
    low: float
    close: float


SlippageFunction = Callable[[object, Bar], float]


def fixed_percent(rate: float = 0.001) -> SlippageFunction:
    """Every fill is `rate` worse than asked: buys pay more, sells receive less."""
    if rate < 0:
        raise ValueError("rate must be >= 0.")

    def model(order, bar: Bar) -> float:
        return order.theoretical_price * (1 + rate if order.qty > 0 else 1 - rate)

    return model


def skewed(fraction: float = 0.15) -> SlippageFunction:
    """
    Fills slip from the bar's open toward its extreme: buys by `fraction` of the
    way to the high, sells by `fraction` of the way to the low.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1.")

    def model(order, bar: Bar) -> float:
        if order.qty > 0:
            return bar.open + fraction * max(0.0, bar.high - bar.open)
        return bar.open - fraction * max(0.0, bar.open - bar.low)

    return model


def resolve(slippage) -> Optional[SlippageFunction]:
    """Turn whatever the user passed as `slippage=` into a model function, or None."""
    if slippage is None:
        return None
    if isinstance(slippage, numbers.Real) and not isinstance(slippage, bool):
        return fixed_percent(float(slippage)) if slippage else None
    if callable(slippage):
        return slippage
    raise TypeError(
        "slippage must be a number, a function (order, bar) -> price, or None; "
        f"got {type(slippage).__name__}."
    )
