"""Funding payments from an optional `funding_rate` column."""
import numpy as np
import pandas as pd
import pytest

from conftest import SYMBOL, make_frame
from overfitting import Strategy
from overfitting.broker import Broker
from overfitting.data import Data

FLAT = (100, 101, 99, 100)


def make_funded_broker(bars, rates, cash=10_000.0):
    frame = make_frame(bars)
    frame["funding_rate"] = rates
    return Broker(Data({SYMBOL: frame}), cash, 0.0, 0.005, 0, None)


def hold(broker, qty, bars_to_run):
    broker.order(SYMBOL, qty, None, type="MARKET")
    for _ in range(bars_to_run):
        broker.next()


def test_long_pays_positive_funding():
    broker = make_funded_broker([FLAT, FLAT], [0.0, 0.001])
    hold(broker, 2, 2)

    # 2 units * 100 * 0.1% = 0.2 paid
    assert broker.cash == pytest.approx(10_000 - 0.2)


def test_short_receives_positive_funding():
    broker = make_funded_broker([FLAT, FLAT], [0.0, 0.001])
    hold(broker, -2, 2)

    assert broker.cash == pytest.approx(10_000 + 0.2)


def test_negative_rate_reverses_the_direction():
    long = make_funded_broker([FLAT, FLAT], [0.0, -0.001])
    short = make_funded_broker([FLAT, FLAT], [0.0, -0.001])
    hold(long, 2, 2)
    hold(short, -2, 2)

    assert long.cash == pytest.approx(10_000 + 0.2)
    assert short.cash == pytest.approx(10_000 - 0.2)


def test_funding_uses_the_open_of_the_funding_bar():
    broker = make_funded_broker([FLAT, (120, 121, 119, 120)], [0.0, 0.001])
    hold(broker, 1, 2)

    assert broker.cash == pytest.approx(10_000 - 120 * 0.001)


def test_funding_is_charged_at_every_funding_bar():
    broker = make_funded_broker([FLAT] * 5, [0.0, 0.001, 0.0, 0.001, 0.002])
    hold(broker, 1, 5)

    assert broker.cash == pytest.approx(10_000 - 100 * (0.001 + 0.001 + 0.002))
    assert len(broker.funding_payments) == 3


@pytest.mark.parametrize("rates", [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
def test_zero_or_missing_rate_means_no_funding(rates):
    broker = make_funded_broker([FLAT] * 3, rates)
    hold(broker, 1, 3)

    assert broker.cash == 10_000
    assert broker.funding_payments == []


def test_no_funding_column_means_no_funding():
    broker = Broker(Data({SYMBOL: make_frame([FLAT] * 3)}), 10_000, 0.0, 0.005, 0, None)
    hold(broker, 1, 3)

    assert broker.cash == 10_000
    assert broker.funding_payments == []


def test_flat_account_pays_nothing():
    broker = make_funded_broker([FLAT] * 3, [0.001] * 3)
    for _ in range(3):
        broker.next()

    assert broker.cash == 10_000


def test_position_opened_on_the_funding_bar_is_not_charged():
    # Funding settles at the open; the order fills after it.
    broker = make_funded_broker([FLAT, FLAT], [0.001, 0.0])
    hold(broker, 1, 2)

    assert broker.cash == 10_000


def test_position_closed_on_the_funding_bar_is_still_charged():
    # It was held up to the funding time.
    broker = make_funded_broker([FLAT, FLAT], [0.0, 0.001])
    hold(broker, 1, 1)
    broker.order(SYMBOL, -1, None, type="MARKET")
    broker.next()

    assert broker.get_position(SYMBOL).qty == 0
    assert broker.cash == pytest.approx(10_000 - 0.1)


def test_funding_payment_log():
    broker = make_funded_broker([FLAT, FLAT], [0.0, 0.001])
    hold(broker, 2, 2)
    payment = broker.funding_payments[0]

    assert payment["time"] == pd.Timestamp("2024-01-01 01:00")
    assert payment["symbol"] == SYMBOL
    assert payment["qty"] == 2.0
    assert payment["price"] == 100
    assert payment["funding_rate"] == 0.001
    assert payment["amount"] == pytest.approx(-0.2)


def test_funding_is_not_logged_as_a_trade():
    broker = make_funded_broker([FLAT, FLAT], [0.0, 0.001])
    hold(broker, 2, 2)

    assert len(broker.trades) == 1


def test_funding_flows_into_strategy_balances():
    class Hold(Strategy):
        def init(self):
            pass

        def next(self, i):
            if i == 0:
                self.market_order(SYMBOL, 1)

    frame = make_frame([FLAT] * 4)
    frame["funding_rate"] = [0.0, 0.01, 0.0, 0.01]
    strategy = Hold(data={SYMBOL: frame}, initial_capital=10_000, commission_rate=0.0)
    strategy.run()

    assert strategy.realized_balances == pytest.approx([10_000, 9_999, 9_999, 9_998])
    assert strategy.equity_curve == pytest.approx([10_000, 9_999, 9_999, 9_998])
    assert list(strategy.fetch_funding()["amount"]) == pytest.approx([-1, -1])
