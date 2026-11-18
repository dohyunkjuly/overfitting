"""Commission, realized PnL, cash updates and slippage models."""
import pytest

from conftest import SYMBOL, make_broker
from overfitting.slippage import Bar, fixed_percent, resolve, skewed

RATE = 0.001


def market(broker, qty):
    broker.order(SYMBOL, qty, None, type="MARKET")
    broker.next()
    return broker.trades[-1]


# ------------------------------------------------------------ commission ---

def test_commission_is_rate_times_notional():
    broker = make_broker([(100, 101, 99, 100)], commission_rate=RATE)
    trade = market(broker, 2)

    assert trade["commission"] == pytest.approx(2 * 100 * RATE)


def test_commission_is_charged_on_shorts_too():
    broker = make_broker([(100, 101, 99, 100)], commission_rate=RATE)
    trade = market(broker, -2)

    assert trade["commission"] == pytest.approx(2 * 100 * RATE)


def test_opening_a_position_only_costs_commission():
    broker = make_broker([(100, 101, 99, 100)], commission_rate=RATE)
    trade = market(broker, 2)

    assert trade["pnl"] == 0
    assert trade["realized_pnl"] == pytest.approx(-0.2)
    assert broker.cash == pytest.approx(10_000 - 0.2)


def test_zero_commission_rate_charges_nothing():
    broker = make_broker([(100, 101, 99, 100)], commission_rate=0.0)
    trade = market(broker, 2)

    assert trade["commission"] == 0
    assert broker.cash == 10_000


# ---------------------------------------------------------- pnl and cash ---

def test_long_round_trip_updates_cash_by_pnl_minus_fees():
    broker = make_broker([(100, 101, 99, 100), (110, 111, 109, 110)], commission_rate=RATE)
    market(broker, 2)
    close = market(broker, -2)

    fees = 2 * 100 * RATE + 2 * 110 * RATE
    assert close["pnl"] == pytest.approx(20)
    assert close["realized_pnl"] == pytest.approx(20 - 2 * 110 * RATE)
    assert broker.cash == pytest.approx(10_000 + 20 - fees)


def test_short_round_trip_loss():
    broker = make_broker([(100, 101, 99, 100), (110, 111, 109, 110)], commission_rate=RATE)
    market(broker, -2)
    close = market(broker, 2)

    fees = 2 * 100 * RATE + 2 * 110 * RATE
    assert close["pnl"] == pytest.approx(-20)
    assert broker.cash == pytest.approx(10_000 - 20 - fees)


def test_partial_close_realizes_only_the_closed_part():
    broker = make_broker([(100, 101, 99, 100), (110, 111, 109, 110)])
    market(broker, 4)
    close = market(broker, -1)

    assert close["pnl"] == pytest.approx(10)
    assert broker.cash == pytest.approx(10_010)


def test_cash_ignores_unrealized_pnl():
    # Cash is realized-only: holding through a rally does not change it.
    broker = make_broker([(100, 101, 99, 100), (150, 151, 149, 150)])
    market(broker, 1)
    broker.next()

    assert broker.cash == 10_000


def test_leverage_does_not_change_pnl():
    results = []
    for leverage in (1, 20):
        broker = make_broker([(100, 101, 99, 100), (103, 104, 102, 103)])
        broker.set_leverage(SYMBOL, leverage)
        market(broker, 2)
        market(broker, -2)
        results.append(broker.cash)

    assert results[0] == pytest.approx(results[1])


def test_trades_are_logged_in_execution_order():
    broker = make_broker([(100, 101, 99, 100), (110, 111, 109, 110)])
    market(broker, 1)
    market(broker, -1)

    assert [t["qty"] for t in broker.trades] == [1.0, -1.0]
    assert [t["executed_price"] for t in broker.trades] == [100, 110]


# -------------------------------------------------------------- slippage ---

def test_no_slippage_model_executes_at_theoretical_price():
    broker = make_broker([(100, 110, 90, 105)])
    trade = market(broker, 1)

    assert trade["theoretical_price"] == 100
    assert trade["executed_price"] == 100


def test_zero_slippage_model_executes_at_theoretical_price():
    broker = make_broker([(100, 110, 90, 105)], slippage=0)
    trade = market(broker, 1)

    assert trade["executed_price"] == 100


@pytest.mark.parametrize("qty, expected", [(1, 101.0), (-1, 99.0)])
def test_fixed_percent_slippage_moves_price_against_the_trader(qty, expected):
    broker = make_broker([(100, 110, 90, 105)], slippage=0.01)
    trade = market(broker, qty)

    assert trade["theoretical_price"] == 100
    assert trade["executed_price"] == pytest.approx(expected)


@pytest.mark.parametrize("qty, expected", [
    (1, 100 + 0.5 * (110 - 100)),   # buys slip toward the high
    (-1, 100 - 0.5 * (100 - 90)),   # sells slip toward the low
])
def test_skewed_slippage_moves_toward_bar_extreme(qty, expected):
    broker = make_broker([(100, 110, 90, 105)], slippage=skewed(0.5))
    trade = market(broker, qty)

    assert trade["executed_price"] == pytest.approx(expected)


def test_commission_uses_the_slipped_price():
    broker = make_broker(
        [(100, 110, 90, 105)],
        commission_rate=RATE,
        slippage=0.01,
    )
    trade = market(broker, 2)

    assert trade["commission"] == pytest.approx(2 * 101 * RATE)


def test_slippage_costs_show_up_in_round_trip_pnl():
    broker = make_broker(
        [(100, 101, 99, 100), (100, 101, 99, 100)],
        slippage=0.01,
    )
    market(broker, 1)   # buys at 101
    market(broker, -1)  # sells at 99

    assert broker.cash == pytest.approx(10_000 - 2)


@pytest.mark.parametrize("factory, bad", [
    (skewed, -0.1),
    (skewed, 1.1),
    (fixed_percent, -0.01),
])
def test_slippage_models_validate_their_parameter(factory, bad):
    with pytest.raises(ValueError):
        factory(bad)


def test_a_number_means_fixed_percent_slippage():
    assert resolve(None) is None
    assert resolve(0) is None
    order = type("O", (), {"qty": 1, "theoretical_price": 200.0})()
    assert resolve(0.01)(order, Bar(200, 210, 190, 205)) == pytest.approx(202)


def test_slippage_can_be_any_function_of_order_and_bar():
    # One tick (0.5) worse than asked, whatever the price.
    def one_tick(order, bar):
        return order.theoretical_price + (0.5 if order.qty > 0 else -0.5)

    broker = make_broker([(100, 110, 90, 105), (100, 110, 90, 105)], slippage=one_tick)
    assert market(broker, 1)["executed_price"] == 100.5
    assert market(broker, -1)["executed_price"] == 99.5


def test_slippage_function_receives_the_bar_by_name():
    seen = []
    broker = make_broker([(100, 110, 90, 105)],
                         slippage=lambda order, bar: seen.append(bar) or bar.high)
    trade = market(broker, 1)

    assert seen[0] == Bar(open=100, high=110, low=90, close=105)
    assert trade["executed_price"] == 110


def test_invalid_slippage_is_rejected():
    with pytest.raises(TypeError):
        make_broker([(100, 110, 90, 105)], slippage="a lot")


# ----------------------------------------------------- broker bookkeeping ---

def test_unrealized_pnl_and_equity_follow_the_close():
    broker = make_broker([(100, 101, 99, 100), (100, 131, 99, 130)])
    market(broker, 2)
    broker.next()

    assert broker.unrealized_pnl(0) == 0
    assert broker.unrealized_pnl(1) == pytest.approx(60)
    assert broker.equity(1) == pytest.approx(10_060)
    assert broker.cash == 10_000


def test_initial_capital_and_its_old_misspelling():
    broker = make_broker([(100, 101, 99, 100)], cash=5_000)

    assert broker.initial_capital == 5_000
    with pytest.warns(DeprecationWarning):
        assert broker.initial_captial == 5_000
