"""Order fills: market, limit, stop-market and stop-limit.

Each test follows the engine's loop: place orders for bar i, then call
broker.next() to process bar i.
"""
from decimal import Decimal

import pandas as pd
import pytest

from conftest import SYMBOL, make_broker
from overfitting.errors import (
    EmptyOrderParameters,
    InvalidOrderParameters,
    StopOrderRejected,
)
from overfitting.types import OrderType

# (open, high, low, close)
QUIET = (100, 110, 95, 105)


def last_trade(broker):
    return broker.trades[-1]


# --------------------------------------------------------------- market ---

def test_market_buy_fills_at_open_of_current_bar():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, 1, None, type="MARKET")
    broker.next()

    assert len(broker.trades) == 1
    assert last_trade(broker)["executed_price"] == 100
    assert broker.get_position(SYMBOL).qty == Decimal("1")
    assert broker.get_position(SYMBOL).price == 100
    assert broker.open_orders[SYMBOL] == {}


def test_market_sell_opens_short_at_open():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, -2, None, type="MARKET")
    broker.next()

    assert last_trade(broker)["executed_price"] == 100
    assert last_trade(broker)["side"] == "SHORT"
    assert broker.get_position(SYMBOL).qty == Decimal("-2")


def test_market_order_records_execution_time_of_its_bar():
    broker = make_broker([QUIET, QUIET])
    broker.next()
    broker.order(SYMBOL, 1, None, type="MARKET")
    broker.next()

    assert last_trade(broker)["executed_at"] == pd.Timestamp("2024-01-01 01:00")


# ---------------------------------------------------------------- limit ---

def test_limit_buy_below_range_stays_open():
    broker = make_broker([QUIET])
    order = broker.order(SYMBOL, 1, 90, type="LIMIT")
    broker.next()

    assert broker.trades == []
    assert order.id in broker.open_orders[SYMBOL]
    assert broker.get_position(SYMBOL).qty == 0


def test_limit_buy_fills_at_limit_price_on_a_later_bar():
    broker = make_broker([QUIET, (100, 101, 85, 90)])
    broker.order(SYMBOL, 1, 90, type="LIMIT")
    broker.next()
    assert broker.trades == []

    broker.next()
    assert last_trade(broker)["executed_price"] == 90
    assert last_trade(broker)["executed_at"] == pd.Timestamp("2024-01-01 01:00")
    assert broker.open_orders[SYMBOL] == {}


def test_limit_sell_fills_at_limit_price_when_high_trades_through():
    broker = make_broker([QUIET, (105, 120, 104, 118)])
    broker.order(SYMBOL, -1, 115, type="LIMIT")
    broker.next()
    assert broker.trades == []

    broker.next()
    assert last_trade(broker)["executed_price"] == 115
    assert broker.get_position(SYMBOL).qty == Decimal("-1")


@pytest.mark.parametrize("qty, price, bar", [
    (1, 95, (100, 110, 95, 105)),    # low touches the buy price exactly
    (-1, 110, (100, 110, 95, 105)),  # high touches the sell price exactly
])
def test_limit_needs_price_to_trade_through_not_just_touch(qty, price, bar):
    broker = make_broker([bar])
    broker.order(SYMBOL, qty, price, type="LIMIT")
    broker.next()

    assert broker.trades == []


def test_marketable_limit_buy_fills_at_open_not_limit():
    # Open is 100; a buy limit at 108 should get the better price.
    broker = make_broker([QUIET])
    broker.order(SYMBOL, 1, 108, type="LIMIT")
    broker.next()

    assert last_trade(broker)["executed_price"] == 100


def test_marketable_limit_sell_fills_at_open_not_limit():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, -1, 97, type="LIMIT")
    broker.next()

    assert last_trade(broker)["executed_price"] == 100


# ---------------------------------------------------------- stop-market ---

def test_stop_market_buy_triggers_and_fills_at_stop_price():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, 1, None, type="STOP", stop_price=105)
    broker.next()

    assert last_trade(broker)["executed_price"] == 105
    assert last_trade(broker)["type"] == "STOP"
    assert broker.get_position(SYMBOL).qty == Decimal("1")


def test_stop_market_sell_triggers_and_fills_at_stop_price():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, -1, None, type="STOP", stop_price=96)
    broker.next()

    assert last_trade(broker)["executed_price"] == 96
    assert broker.get_position(SYMBOL).qty == Decimal("-1")


def test_stop_market_triggers_when_price_only_touches_stop():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, 1, None, type="STOP", stop_price=110)  # high == stop
    broker.next()

    assert len(broker.trades) == 1


def test_stop_market_stays_open_until_triggered():
    broker = make_broker([QUIET, (105, 125, 104, 120)])
    order = broker.order(SYMBOL, 1, None, type="STOP", stop_price=120)
    broker.next()
    assert broker.trades == []
    assert order.is_triggered is False

    broker.next()
    assert last_trade(broker)["executed_price"] == 120


@pytest.mark.parametrize("qty, stop_price", [
    (1, 99),    # buy stop below the open would trigger immediately
    (-1, 101),  # sell stop above the open would trigger immediately
])
def test_stop_on_wrong_side_of_market_is_rejected(qty, stop_price):
    broker = make_broker([QUIET])
    with pytest.raises(StopOrderRejected):
        broker.order(SYMBOL, qty, None, type="STOP", stop_price=stop_price)

    assert broker.open_orders[SYMBOL] == {}


# ----------------------------------------------------------- stop-limit ---

def test_stop_limit_buy_triggers_then_fills_at_limit_price():
    # Stop at 105, limit at 104: once triggered, wait for a pullback below 104.
    broker = make_broker([(100, 104, 99, 103), (103, 106, 103.5, 105)])
    order = broker.order(SYMBOL, 1, 104, type="STOP", stop_price=105)
    broker.next()
    assert order.is_triggered is False
    assert broker.trades == []

    broker.next()
    assert order.is_triggered is True
    assert last_trade(broker)["executed_price"] == 104


def test_stop_limit_triggered_but_not_filled_waits_for_limit():
    broker = make_broker([(100, 106, 104.5, 105), (105, 105, 103, 104)])
    order = broker.order(SYMBOL, 1, 104, type="STOP", stop_price=105)
    broker.next()
    assert order.is_triggered is True
    assert broker.trades == []
    assert order.id in broker.open_orders[SYMBOL]

    broker.next()
    assert last_trade(broker)["executed_price"] == 104


def test_stop_limit_sell_triggers_then_fills_at_limit_price():
    # Stop at 95, limit at 96: once triggered, wait for a bounce above 96.
    broker = make_broker([(100, 101, 97, 98), (98, 97, 94, 95), (95, 97, 94, 96)])
    order = broker.order(SYMBOL, -1, 96, type="STOP", stop_price=95)
    broker.next()
    assert order.is_triggered is False

    broker.next()
    assert order.is_triggered is True
    assert last_trade(broker)["executed_price"] == 96
    assert broker.get_position(SYMBOL).qty == Decimal("-1")


# ------------------------------------------------------------ validation ---

def test_order_requires_symbol():
    broker = make_broker([QUIET])
    with pytest.raises(InvalidOrderParameters):
        broker.order("", 1, 100, type="LIMIT")


def test_order_rejects_zero_quantity():
    broker = make_broker([QUIET])
    with pytest.raises(InvalidOrderParameters):
        broker.order(SYMBOL, 0, 100, type="LIMIT")


def test_order_rejects_unknown_type():
    broker = make_broker([QUIET])
    with pytest.raises(InvalidOrderParameters):
        broker.order(SYMBOL, 1, 100, type="ICEBERG")


def test_limit_order_requires_price():
    broker = make_broker([QUIET])
    with pytest.raises(EmptyOrderParameters):
        broker.order(SYMBOL, 1, None, type="LIMIT")


def test_stop_order_requires_stop_price():
    broker = make_broker([QUIET])
    with pytest.raises(EmptyOrderParameters):
        broker.order(SYMBOL, 1, None, type="STOP")


def test_order_type_is_case_insensitive():
    broker = make_broker([QUIET])
    order = broker.order(SYMBOL, 1, 90, type="limit")
    assert order.type is OrderType.LIMIT


def test_order_defaults_to_limit_when_type_omitted():
    broker = make_broker([QUIET])
    order = broker.order(SYMBOL, 1, 90)
    assert order.type is OrderType.LIMIT


def test_order_accepts_the_enum_as_type():
    broker = make_broker([QUIET])
    order = broker.order(SYMBOL, 1, None, type=OrderType.MARKET)
    assert order.type is OrderType.MARKET


# ---------------------------------------------------- cancel and flatten ---

def test_cancel_order_removes_it_and_returns_it():
    broker = make_broker([QUIET])
    order = broker.order(SYMBOL, 1, 90, type="LIMIT")

    assert broker.cancel_order(SYMBOL, order.id) is order
    assert broker.open_orders[SYMBOL] == {}
    broker.next()
    assert broker.trades == []


def test_cancel_unknown_order_returns_none():
    broker = make_broker([QUIET])
    assert broker.cancel_order(SYMBOL, "nope") is None
    assert broker.cancel_order("ETH", "nope") is None


def test_cancel_all_orders_only_touches_that_symbol():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, 1, 90, type="LIMIT")
    broker.order(SYMBOL, 1, 91, type="LIMIT")
    broker.cancel_all_orders(SYMBOL)

    assert broker.open_orders[SYMBOL] == {}


def test_close_all_positions_flattens_with_a_market_order():
    broker = make_broker([QUIET, (105, 106, 104, 105)])
    broker.order(SYMBOL, 3, None, type="MARKET")
    broker.next()

    broker.close_all_positions(SYMBOL)
    broker.next()

    assert broker.get_position(SYMBOL).qty == 0
    assert last_trade(broker)["executed_price"] == 105
    assert last_trade(broker)["qty"] == -3.0


def test_close_all_positions_when_flat_places_nothing():
    broker = make_broker([QUIET])
    broker.close_all_positions(SYMBOL)
    assert broker.open_orders.get(SYMBOL, {}) == {}


# ---------------------------------------------------------- order object ---

def test_order_side_and_decimal_quantity():
    broker = make_broker([QUIET])
    buy = broker.order(SYMBOL, 0.1, 90, type="LIMIT")
    sell = broker.order(SYMBOL, -0.1, 120, type="LIMIT")

    assert buy.side == "LONG" and sell.side == "SHORT"
    assert buy.qty == Decimal("0.1")
    assert buy.created_at == pd.Timestamp("2024-01-01 00:00")


def test_trade_log_entry_is_plain_python():
    broker = make_broker([QUIET])
    broker.order(SYMBOL, 1.5, None, type="MARKET", label="entry")
    broker.next()
    trade = last_trade(broker)

    assert trade["type"] == "MARKET"
    assert isinstance(trade["qty"], float) and trade["qty"] == 1.5
    assert trade["label"] == "entry"
    assert trade["symbol"] == SYMBOL
