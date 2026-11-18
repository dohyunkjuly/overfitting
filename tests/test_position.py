"""Position bookkeeping: averaging in, partial close, full close, flipping sides."""
from decimal import Decimal

import pandas as pd
import pytest

from conftest import SYMBOL, fill
from overfitting.order import Order
from overfitting.position import Position
from overfitting.types import OrderType


def test_new_position_is_flat(position):
    assert position.qty == 0
    assert position.price == 0
    assert position.margin == 0
    assert position.liquid_price == 0
    assert position.leverage == 1


def test_open_long(position):
    pnl = fill(position, 2, 100)

    assert pnl == 0
    assert position.qty == Decimal("2")
    assert position.price == 100


def test_open_short(position):
    pnl = fill(position, -2, 100)

    assert pnl == 0
    assert position.qty == Decimal("-2")
    assert position.price == 100


# ---------------------------------------------------------- averaging in ---

def test_averaging_into_long_uses_weighted_entry(position):
    fill(position, 1, 100)
    pnl = fill(position, 1, 110)

    assert pnl == 0
    assert position.qty == Decimal("2")
    assert position.price == pytest.approx(105)


def test_averaging_into_short_uses_weighted_entry(position):
    fill(position, -1, 100)
    fill(position, -3, 120)

    assert position.qty == Decimal("-4")
    assert position.price == pytest.approx(115)


def test_averaging_in_moves_liquidation_price(position):
    position.set_leverage(10)
    fill(position, 1, 100)
    first = position.liquid_price
    fill(position, 1, 120)

    assert position.liquid_price > first
    assert position.liquid_price == pytest.approx(110 * (1 - 1 / 10 + 0.005))


# --------------------------------------------------------- partial close ---

def test_partial_close_long_realizes_pnl_and_keeps_entry(position):
    fill(position, 2, 100)
    pnl = fill(position, -1, 110)

    assert pnl == pytest.approx(10)
    assert position.qty == Decimal("1")
    assert position.price == 100


def test_partial_close_short_realizes_pnl_and_keeps_entry(position):
    fill(position, -4, 100)
    pnl = fill(position, 1, 90)

    assert pnl == pytest.approx(10)
    assert position.qty == Decimal("-3")
    assert position.price == 100


def test_partial_close_shrinks_margin(position):
    position.set_leverage(10)
    fill(position, 2, 100)
    before = position.margin
    fill(position, -1, 110)

    assert position.margin == pytest.approx(before / 2)


# ------------------------------------------------------------ full close ---

@pytest.mark.parametrize("qty, entry, exit_price, expected", [
    (2, 100, 110, 20),    # long, profit
    (2, 100, 95, -10),    # long, loss
    (-2, 100, 90, 20),    # short, profit
    (-2, 100, 105, -10),  # short, loss
])
def test_full_close_pnl(position, qty, entry, exit_price, expected):
    fill(position, qty, entry)
    pnl = fill(position, -qty, exit_price)

    assert pnl == pytest.approx(expected)


def test_full_close_resets_position(position):
    position.set_leverage(5)
    fill(position, 2, 100)
    fill(position, -2, 110)

    assert position.qty == 0
    assert position.price == 0
    assert position.margin == 0
    assert position.liquid_price == 0
    assert position.leverage == 5  # leverage setting survives


def test_decimal_quantities_close_exactly(position):
    # 0.1 + 0.2 != 0.3 in floats; the position must still end exactly flat.
    fill(position, 0.1, 100)
    fill(position, 0.2, 100)
    fill(position, -0.3, 100)

    assert position.qty == 0
    assert position.price == 0


# -------------------------------------------------------------- flipping ---

def test_flip_long_to_short(position):
    position.set_leverage(10)
    fill(position, 1, 100)
    pnl = fill(position, -3, 110)

    assert pnl == pytest.approx(10)          # only the 1 closed unit realizes
    assert position.qty == Decimal("-2")
    assert position.price == 110             # new entry is the flip price
    assert position.liquid_price > 110       # short liquidates above entry


def test_flip_short_to_long(position):
    position.set_leverage(10)
    fill(position, -2, 100)
    pnl = fill(position, 5, 90)

    assert pnl == pytest.approx(20)
    assert position.qty == Decimal("3")
    assert position.price == 90
    assert position.liquid_price < 90        # long liquidates below entry


# ------------------------------------------------------------ validation ---

def test_rejects_order_for_another_symbol():
    order = Order(pd.Timestamp("2024-01-01"), SYMBOL, 1, 100, OrderType.MARKET)
    order.executed_price = 100
    with pytest.raises(ValueError):
        Position("ETH").process_trade(order)


def test_rejects_zero_quantity_order(position):
    with pytest.raises(ValueError):
        fill(position, 0, 100)


@pytest.mark.parametrize("leverage", [0, -1, 101])
def test_leverage_out_of_range_is_rejected(position, leverage):
    with pytest.raises(Exception):
        position.set_leverage(leverage)


@pytest.mark.parametrize("leverage", [1, 100])
def test_leverage_bounds_are_inclusive(position, leverage):
    position.set_leverage(leverage)
    assert position.leverage == leverage
