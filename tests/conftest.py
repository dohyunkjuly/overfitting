"""Shared helpers for building tiny, hand-checkable markets."""
import matplotlib

matplotlib.use("Agg")  # draw charts without opening windows

import pandas as pd
import pytest

from overfitting.broker import Broker
from overfitting.data import Data
from overfitting.order import Order
from overfitting.position import Position
from overfitting.types import OrderType

SYMBOL = "BTC"
START = "2024-01-01"


def make_frame(bars, start=START, freq="h"):
    """Build an OHLC DataFrame from a list of (open, high, low, close) tuples."""
    index = pd.date_range(start, periods=len(bars), freq=freq)
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=index, dtype=float)


def make_broker(
    bars,
    *,
    cash=10_000.0,
    commission_rate=0.0,
    maint_margin_rate=0.005,
    maint_amount=0.0,
    slippage=None,
    symbol=SYMBOL,
):
    """A Broker over a single symbol. Commission defaults to zero so prices are easy to check."""
    data = Data({symbol: make_frame(bars)})
    return Broker(
        data=data,
        cash=cash,
        commission_rate=commission_rate,
        maint_margin_rate=maint_margin_rate,
        maint_amount=maint_amount,
        slippage=slippage,
    )


def fill(position: Position, qty, price, *, liquidation=False):
    """Apply a filled order of `qty` at `price` to a Position. Returns realized PnL."""
    order = Order(pd.Timestamp(START), position.symbol, qty, price, OrderType.MARKET)
    order.executed_price = price
    return position.process_trade(order, liquidation)


@pytest.fixture
def position():
    return Position(SYMBOL, maint_margin_rate=0.005, maint_amount=0.0)
