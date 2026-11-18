"""Liquidation price maths and the broker's forced-close behaviour (isolated margin)."""
from decimal import Decimal

import pytest

from conftest import SYMBOL, fill, make_broker
from overfitting.errors import LiquidationError
from overfitting.position import Position

MMR = 0.005
LEVERAGES = [1, 2, 5, 10, 25, 50, 100]


# --------------------------------------------------- liquidation price ---

@pytest.mark.parametrize("leverage", LEVERAGES)
def test_long_liquidation_price(position, leverage):
    position.set_leverage(leverage)
    fill(position, 1, 100)

    # entry * (1 - 1/leverage + maintenance rate)
    assert position.liquid_price == pytest.approx(100 * (1 - 1 / leverage + MMR))
    assert position.liquid_price < 100


@pytest.mark.parametrize("leverage", LEVERAGES)
def test_short_liquidation_price(position, leverage):
    position.set_leverage(leverage)
    fill(position, -1, 100)

    # entry * (1 + 1/leverage - maintenance rate)
    assert position.liquid_price == pytest.approx(100 * (1 + 1 / leverage - MMR))
    assert position.liquid_price > 100


@pytest.mark.parametrize("leverage, long_liq, short_liq", [
    (1, 0.5, 199.5),
    (2, 50.5, 149.5),
    (10, 90.5, 109.5),
    (50, 98.5, 101.5),
    (100, 99.5, 100.5),
])
def test_liquidation_price_worked_examples(leverage, long_liq, short_liq):
    long = Position(SYMBOL, MMR, 0)
    long.set_leverage(leverage)
    fill(long, 1, 100)

    short = Position(SYMBOL, MMR, 0)
    short.set_leverage(leverage)
    fill(short, -1, 100)

    assert long.liquid_price == pytest.approx(long_liq)
    assert short.liquid_price == pytest.approx(short_liq)


def test_liquidation_price_does_not_depend_on_size_without_maint_amount():
    small, large = Position(SYMBOL, MMR, 0), Position(SYMBOL, MMR, 0)
    for p in (small, large):
        p.set_leverage(10)
    fill(small, 0.01, 100)
    fill(large, 250, 100)

    assert small.liquid_price == pytest.approx(large.liquid_price)


def test_maint_amount_lowers_maintenance_margin():
    # notional 20,000; initial margin 2,000; maintenance 20,000 * 0.005 - 50 = 50
    position = Position(SYMBOL, MMR, 50)
    position.set_leverage(10)
    fill(position, 1, 20_000)

    assert position.margin == pytest.approx(2_050)
    assert position.liquid_price == pytest.approx(20_000 - 1_950)


def test_maintenance_margin_never_goes_negative():
    # notional 100 * 0.005 - 50 would be negative; it is floored at zero.
    position = Position(SYMBOL, MMR, 50)
    position.set_leverage(10)
    fill(position, 1, 100)

    assert position.margin == pytest.approx(10)
    assert position.liquid_price == pytest.approx(90)


def test_margin_is_initial_plus_maintenance(position):
    position.set_leverage(10)
    fill(position, 2, 100)

    assert position.margin == pytest.approx(200 / 10 + 200 * MMR)


def test_changing_leverage_recalculates_liquidation_price(position):
    fill(position, 1, 100)
    assert position.liquid_price == pytest.approx(0.5)

    position.set_leverage(10)
    assert position.liquid_price == pytest.approx(90.5)


# ------------------------------------------------- broker forced close ---

def open_position(broker, qty, leverage):
    broker.set_leverage(SYMBOL, leverage)
    broker.order(SYMBOL, qty, None, type="MARKET")
    broker.next()


def test_long_is_liquidated_after_low_crosses_liquidation_price():
    # 10x long from 100 liquidates at 90.5.
    broker = make_broker([(100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92)])
    open_position(broker, 1, 10)

    broker.next()  # bar 1 trades through 90.5, detected on the next bar
    assert broker.get_position(SYMBOL).qty == Decimal("1")

    broker.next()
    assert broker.get_position(SYMBOL).qty == 0
    assert len(broker.trades) == 2
    assert broker.trades[-1]["executed_price"] == pytest.approx(90.5)
    assert broker.trades[-1]["qty"] == -1.0


def test_short_is_liquidated_after_high_crosses_liquidation_price():
    # 10x short from 100 liquidates at 109.5.
    broker = make_broker([(100, 101, 99, 100), (100, 110, 100, 108), (108, 109, 107, 108)])
    open_position(broker, -1, 10)
    broker.next()
    broker.next()

    assert broker.get_position(SYMBOL).qty == 0
    assert broker.trades[-1]["executed_price"] == pytest.approx(109.5)
    assert broker.trades[-1]["qty"] == 1.0


def test_liquidation_triggers_when_price_only_touches_level():
    broker = make_broker([(100, 101, 99, 100), (100, 100, 90.5, 92), (92, 93, 91, 92)])
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    assert broker.get_position(SYMBOL).qty == 0


def test_no_liquidation_when_price_stays_inside():
    broker = make_broker([(100, 101, 99, 100), (100, 100, 91, 92), (92, 93, 91, 92)])
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    assert broker.get_position(SYMBOL).qty == Decimal("1")
    assert len(broker.trades) == 1


def test_liquidation_loses_the_whole_margin():
    # Current rule: the loss is initial margin + maintenance margin (10 + 0.5).
    # ROADMAP Phase 2 reviews this against exchange rules.
    broker = make_broker([(100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92)])
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    assert broker.trades[-1]["pnl"] == pytest.approx(-10.5)
    assert broker.cash == pytest.approx(10_000 - 10.5)


def test_liquidation_also_charges_commission():
    broker = make_broker(
        [(100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92)],
        commission_rate=0.001,
    )
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    entry_fee = 100 * 0.001
    liquidation_fee = 90.5 * 0.001
    assert broker.trades[-1]["commission"] == pytest.approx(liquidation_fee)
    assert broker.cash == pytest.approx(10_000 - 10.5 - entry_fee - liquidation_fee)


def test_liquidation_leaves_no_open_orders_behind():
    broker = make_broker([(100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92)])
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    assert broker.open_orders[SYMBOL] == {}


def test_position_can_be_reopened_after_liquidation():
    broker = make_broker([
        (100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92), (92, 93, 91, 92),
    ])
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    broker.order(SYMBOL, 2, None, type="MARKET")
    broker.next()
    position = broker.get_position(SYMBOL)

    assert position.qty == Decimal("2")
    assert position.price == 92
    assert position.liquid_price == pytest.approx(92 * (1 - 1 / 10 + MMR))


@pytest.mark.xfail(strict=True, reason="ROADMAP Phase 2: liquidation flag is set on the broker, not the order")
def test_liquidation_is_flagged_in_the_trade_log():
    broker = make_broker([(100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 91, 92)])
    open_position(broker, 1, 10)
    broker.next()
    broker.next()

    assert broker.trades[0]["is_liquidation_order"] is False
    assert broker.trades[-1]["is_liquidation_order"] is True


# ------------------------------------------------- leverage change guard ---

def test_raising_leverage_into_liquidation_is_refused():
    # 1x long from 100, market now at 95. At 50x the liquidation price would be 98.5.
    broker = make_broker([(100, 101, 99, 100), (95, 96, 94, 95)])
    open_position(broker, 1, 1)

    with pytest.raises(LiquidationError):
        broker.set_leverage(SYMBOL, 50)


def test_safe_leverage_change_is_allowed():
    broker = make_broker([(100, 101, 99, 100), (95, 96, 94, 95)])
    open_position(broker, 1, 1)
    broker.set_leverage(SYMBOL, 5)

    assert broker.get_position(SYMBOL).leverage == 5
    assert broker.get_position(SYMBOL).liquid_price == pytest.approx(100 * (1 - 1 / 5 + MMR))
