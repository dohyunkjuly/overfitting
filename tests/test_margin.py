"""Margin check: orders the account cannot afford are rejected."""
from decimal import Decimal

import pytest

from conftest import SYMBOL, make_broker
from overfitting.errors import InsufficientMargin

FLAT = (100, 101, 99, 100)


def market(broker, qty):
    broker.order(SYMBOL, qty, None, type="MARKET")
    broker.next()


# ------------------------------------------------------------- placement ---

def test_order_within_cash_is_accepted():
    broker = make_broker([FLAT], cash=1_000)
    market(broker, 10)  # notional 1,000 at 1x

    assert broker.get_position(SYMBOL).qty == Decimal("10")


def test_order_beyond_cash_is_rejected():
    broker = make_broker([FLAT], cash=1_000)
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, 10.1, None, type="MARKET")

    assert broker.open_orders[SYMBOL] == {}


def test_short_beyond_cash_is_rejected():
    broker = make_broker([FLAT], cash=1_000)
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, -11, None, type="MARKET")


def test_leverage_raises_buying_power():
    broker = make_broker([FLAT, FLAT], cash=1_000)
    broker.set_leverage(SYMBOL, 10)
    market(broker, 100)  # notional 10,000, margin 1,000

    assert broker.get_position(SYMBOL).qty == Decimal("100")
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, 1, None, type="MARKET")


@pytest.mark.parametrize("kwargs", [
    dict(price=100, type="LIMIT"),                    # 11 * 100 > 1,000
    dict(price=None, type="STOP", stop_price=100),    # stop-market, priced at the stop
    dict(price=101, type="STOP", stop_price=100),     # stop-limit, priced at the limit
])
def test_every_order_type_is_checked(kwargs):
    broker = make_broker([FLAT], cash=1_000)
    price = kwargs.pop("price")
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, 11, price, **kwargs)


def test_limit_order_is_priced_at_its_limit():
    # 20 units at a limit of 50 costs 1,000 even though the market is at 100.
    broker = make_broker([FLAT], cash=1_000)
    order = broker.order(SYMBOL, 20, 50, type="LIMIT")

    assert order.id in broker.open_orders[SYMBOL]


# ------------------------------------------------------ available margin ---

def test_open_position_locks_its_margin():
    broker = make_broker([FLAT, FLAT], cash=1_000)
    broker.set_leverage(SYMBOL, 5)
    market(broker, 20)  # notional 2,000, margin 400

    assert broker.get_position(SYMBOL).margin == pytest.approx(400)
    assert broker.available_margin() == pytest.approx(600)
    assert broker.cash == 1_000


def test_adding_to_a_position_is_limited_by_what_is_left():
    broker = make_broker([FLAT, FLAT], cash=1_000)
    market(broker, 6)   # locks 600
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, 5, None, type="MARKET")  # needs 500, only 400 left

    market(broker, 4)   # exactly what is left
    assert broker.get_position(SYMBOL).qty == Decimal("10")
    assert broker.available_margin() == pytest.approx(0)


def test_margin_is_shared_across_symbols():
    from conftest import make_frame
    from overfitting.broker import Broker
    from overfitting.data import Data

    data = Data({"BTC": make_frame([FLAT, FLAT]), "ETH": make_frame([FLAT, FLAT])})
    broker = Broker(data, 1_000, 0.0, 0.005, 0, None)
    broker.order("BTC", 7, None, type="MARKET")
    broker.next()

    with pytest.raises(InsufficientMargin):
        broker.order("ETH", 4, None, type="MARKET")  # needs 400, only 300 left


def test_closing_releases_margin():
    broker = make_broker([FLAT, FLAT, FLAT], cash=1_000)
    market(broker, 10)
    market(broker, -10)

    assert broker.available_margin() == pytest.approx(1_000)
    market(broker, 10)
    assert broker.get_position(SYMBOL).qty == Decimal("10")


# ------------------------------------------------- reduce, close and flip ---

def test_reducing_orders_never_need_margin():
    # After a loss the account has less cash than the position's margin,
    # but it must still be able to get out.
    broker = make_broker([FLAT, (40, 41, 39, 40), (40, 41, 39, 40)], cash=1_000)
    market(broker, 10)          # long 10 from 100
    market(broker, -4)          # sells 4 at 40, realizing -240
    assert broker.cash == pytest.approx(760)

    market(broker, -6)
    assert broker.get_position(SYMBOL).qty == 0


def test_flip_needs_margin_only_for_the_new_side():
    broker = make_broker([FLAT, FLAT], cash=1_000)
    market(broker, 10)          # long 10, all margin used
    market(broker, -20)         # flip to short 10: old 1,000 released, new 1,000 needed

    assert broker.get_position(SYMBOL).qty == Decimal("-10")


def test_flip_that_is_too_large_is_rejected():
    broker = make_broker([FLAT, FLAT], cash=1_000)
    market(broker, 10)
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, -21, None, type="MARKET")  # would be short 11


def test_required_margin_helper():
    broker = make_broker([FLAT, FLAT], cash=1_000)
    broker.set_leverage(SYMBOL, 4)
    assert broker.required_margin(SYMBOL, 8, 100) == pytest.approx(200)

    market(broker, 8)
    assert broker.required_margin(SYMBOL, -3, 100) == 0          # reduce
    assert broker.required_margin(SYMBOL, -8, 100) == 0          # close
    assert broker.required_margin(SYMBOL, 2, 100) == pytest.approx(50)   # add
    assert broker.required_margin(SYMBOL, -12, 100) == pytest.approx(0)  # flip to 4 short: 100 needed, 200 released


# ------------------------------------------------------------- fill time ---

def test_resting_order_is_dropped_if_unaffordable_when_it_fills():
    # Two limit orders are each affordable when placed, but not together.
    broker = make_broker([FLAT, (100, 100, 80, 85)], cash=1_000)
    first = broker.order(SYMBOL, 6, 90, type="LIMIT")    # 540
    second = broker.order(SYMBOL, 6, 89, type="LIMIT")   # 534
    broker.next()
    broker.next()

    assert broker.get_position(SYMBOL).qty == Decimal("6")
    assert len(broker.trades) == 1
    assert broker.trades[0]["id"] == first.id
    assert broker.open_orders[SYMBOL] == {}
    assert len(broker.rejected_orders) == 1
    assert broker.rejected_orders[0]["id"] == second.id
    assert "margin" in broker.rejected_orders[0]["reject_reason"]


def test_all_in_order_is_not_rejected_by_rounding():
    broker = make_broker([(0.1, 0.1, 0.1, 0.1)], cash=1_000)
    broker.set_leverage(SYMBOL, 3)
    market(broker, 1_000 * 3 / 0.1)

    assert broker.get_position(SYMBOL).qty != 0
    assert broker.rejected_orders == []


def test_slippage_alone_does_not_reject_an_all_in_order():
    from overfitting import Slippage

    broker = make_broker([FLAT], cash=1_000, slippage_model=Slippage.FixedPercent(f=0.01))
    market(broker, 10)  # affordable at 100, fills at 101

    assert broker.get_position(SYMBOL).qty == Decimal("10")
    assert broker.rejected_orders == []


def test_closing_is_allowed_even_when_available_margin_is_negative():
    from overfitting import Slippage

    # All-in at 100 fills at 101, so the locked margin (1,010) exceeds cash (1,000).
    broker = make_broker([FLAT, FLAT], cash=1_000, slippage_model=Slippage.FixedPercent(f=0.01))
    market(broker, 10)
    assert broker.available_margin() < 0

    market(broker, -10)
    assert broker.get_position(SYMBOL).qty == 0
