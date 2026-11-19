"""Round trips, per-bar account and position history, and order status."""
import numpy as np
import pandas as pd
import pytest

from conftest import SYMBOL, make_broker, make_frame
from overfitting import Strategy
from overfitting.broker import Broker
from overfitting.data import Data
from overfitting.errors import InsufficientMargin, StopOrderRejected

approx = pytest.approx
FLAT = (100, 101, 99, 100)


def market(broker, qty, label=None):
    broker.order(SYMBOL, qty, None, type="MARKET", label=label)
    broker.next()


def trips(broker):
    return broker.recorder.round_trips_frame()


# ------------------------------------------------------------ round trips ---

def test_no_round_trip_until_the_position_is_closed():
    broker = make_broker([FLAT, FLAT])
    market(broker, 1)

    assert trips(broker).empty
    assert broker.recorder.open_symbols() == [SYMBOL]


def test_simple_long_round_trip():
    # Buy at 100, hold a bar that ranges 95 to 120, sell at 110.
    broker = make_broker([FLAT, (100, 120, 95, 110), (110, 111, 109, 110)], commission_rate=0.001)
    market(broker, 2, label="entry")
    broker.next()
    market(broker, -2, label="exit")
    trip = trips(broker).iloc[0]

    assert trip["symbol"] == SYMBOL and trip["side"] == "LONG"
    assert trip["entry_time"] == pd.Timestamp("2024-01-01 00:00")
    assert trip["exit_time"] == pd.Timestamp("2024-01-01 02:00")
    assert trip["bars_held"] == 2
    assert trip["duration"] == pd.Timedelta(hours=2)
    assert trip["qty"] == 2
    assert trip["entry_price"] == 100 and trip["exit_price"] == 110
    assert trip["pnl"] == approx(20)
    assert trip["commission"] == approx(0.2 + 0.22)
    assert trip["net_pnl"] == approx(20 - 0.42)
    assert trip["return_pct"] == approx((20 - 0.42) / 200)
    assert trip["fills"] == 2
    assert trip["entry_label"] == "entry" and trip["exit_label"] == "exit"
    assert not trip["is_liquidation"]
    assert broker.recorder.open_symbols() == []


def test_short_round_trip_profit_and_side():
    broker = make_broker([FLAT, (90, 91, 89, 90)])
    market(broker, -3)
    market(broker, 3)
    trip = trips(broker).iloc[0]

    assert trip["side"] == "SHORT"
    assert trip["pnl"] == approx(30)
    assert trip["return_pct"] == approx(30 / 300)


def test_excursions_for_a_long():
    # Entry 100. Worst low 90 (-10%), best high 125 (+25%).
    broker = make_broker([FLAT, (100, 125, 90, 110), (110, 111, 109, 110)])
    market(broker, 1)
    broker.next()
    market(broker, -1)
    trip = trips(broker).iloc[0]

    assert trip["mae"] == approx(-0.10)
    assert trip["mfe"] == approx(0.25)


def test_excursions_for_a_short_are_mirrored():
    # Entry 100. A short is hurt by the high of 125 and helped by the low of 90.
    broker = make_broker([FLAT, (100, 125, 90, 110), (110, 111, 109, 110)])
    market(broker, -1)
    broker.next()
    market(broker, 1)
    trip = trips(broker).iloc[0]

    assert trip["mae"] == approx(-0.25)
    assert trip["mfe"] == approx(0.10)


def test_excursions_have_the_right_sign_even_for_a_one_bar_trade():
    broker = make_broker([(100, 100, 100, 100), (104, 104, 104, 104)])
    market(broker, 1)
    market(broker, -1)
    trip = trips(broker).iloc[0]

    assert trip["mae"] == 0
    assert trip["mfe"] == approx(0.04)


def test_scaling_in_and_out_is_one_round_trip():
    bars = [FLAT, (110, 111, 109, 110), (120, 121, 119, 120), (130, 131, 129, 130)]
    broker = make_broker(bars)
    market(broker, 1)      # 1 @ 100
    market(broker, 1)      # 1 @ 110, average entry 105
    market(broker, -1)     # 1 @ 120, realizes 15
    market(broker, -1)     # 1 @ 130, realizes 25
    result = trips(broker)

    assert len(result) == 1
    trip = result.iloc[0]
    assert trip["qty"] == 2                      # largest size held
    assert trip["entry_price"] == approx(105)
    assert trip["exit_price"] == approx(125)
    assert trip["pnl"] == approx(40)
    assert trip["fills"] == 4
    assert trip["return_pct"] == approx(40 / 210)


def test_flip_closes_one_trip_and_opens_another():
    broker = make_broker([FLAT, (110, 111, 109, 110), (100, 101, 99, 100)], commission_rate=0.001)
    market(broker, 1)      # long 1 @ 100
    market(broker, -3)     # sells 3 @ 110: closes the long, opens short 2
    market(broker, 2)      # covers 2 @ 100
    result = trips(broker)

    assert list(result["side"]) == ["LONG", "SHORT"]
    assert list(result["qty"]) == [1, 2]
    assert result["pnl"].tolist() == approx([10, 20])
    # The flip fill's commission (0.33) is split 1/3 to the long, 2/3 to the short.
    assert result["commission"].tolist() == approx([0.1 + 0.11, 0.22 + 0.2])


def test_liquidation_closes_the_round_trip():
    broker = make_broker([FLAT, (100, 100, 90, 92), (92, 93, 91, 92)])
    broker.set_leverage(SYMBOL, 10)
    market(broker, 1)
    broker.next()
    broker.next()
    trip = trips(broker).iloc[0]

    assert trip["is_liquidation"]
    assert trip["exit_price"] == approx(90.5)
    assert trip["pnl"] == approx(-10)            # the whole margin
    assert trip["commission"] == 0
    assert trip["mae"] == approx(-0.10)          # the bar's low of 90
    assert broker.recorder.open_symbols() == []


def test_funding_is_attributed_to_the_open_round_trip():
    frame = make_frame([FLAT, FLAT, FLAT])
    frame["funding_rate"] = [0.0, 0.01, 0.0]
    broker = Broker(Data({SYMBOL: frame}), 10_000, 0.0, 0.005, 0, None, funding=True)
    market(broker, 1)
    broker.next()
    market(broker, -1)
    trip = trips(broker).iloc[0]

    assert trip["funding"] == approx(-1)
    assert trip["net_pnl"] == approx(-1)


def test_slippage_shows_in_entry_and_exit_prices():
    broker = make_broker([FLAT, FLAT], slippage=0.01)
    market(broker, 1)
    market(broker, -1)
    trip = trips(broker).iloc[0]

    assert trip["entry_price"] == approx(101) and trip["exit_price"] == approx(99)
    assert trip["pnl"] == approx(-2)


def test_round_trips_add_up_to_the_change_in_cash():
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    bars = [(c, c * 1.01, c * 0.99, c) for c in closes]
    broker = make_broker(bars, commission_rate=0.0005, slippage=0.0005)
    for i in range(len(bars)):
        position = broker.get_position(SYMBOL).qty
        target = int(rng.integers(-3, 4))
        if target != position and i < len(bars) - 1:
            broker.order(SYMBOL, target - float(position), None, type="MARKET")
        elif i == len(bars) - 1 and position != 0:
            broker.order(SYMBOL, -float(position), None, type="MARKET")
        broker.next()

    result = trips(broker)
    assert broker.get_position(SYMBOL).qty == 0
    assert len(result) > 20
    assert result["net_pnl"].sum() == approx(broker.cash - 10_000)
    assert result["commission"].sum() == approx(sum(t["commission"] for t in broker.trades))
    assert (result["mae"] <= 0).all() and (result["mfe"] >= 0).all()


def test_empty_round_trip_table_keeps_its_columns():
    broker = make_broker([FLAT])
    assert list(trips(broker).columns)[:4] == ["symbol", "side", "entry_time", "exit_time"]


# --------------------------------------------------------- account history ---

def test_account_history_has_one_row_per_bar_run():
    broker = make_broker([FLAT, FLAT, FLAT])
    broker.next()
    broker.next()
    history = broker.recorder.account_history()

    assert len(history) == 2
    assert history.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert list(history.columns) == [
        "cash", "equity", "margin", "gross_exposure", "net_exposure", "leverage"]


def test_account_history_while_flat():
    broker = make_broker([FLAT])
    broker.next()
    row = broker.recorder.account_history().iloc[0]

    assert row["cash"] == 10_000 and row["equity"] == 10_000
    assert row["margin"] == 0 and row["gross_exposure"] == 0 and row["leverage"] == 0


def test_account_history_tracks_a_leveraged_long():
    # Long 20 from 100 at 5x on 1,000 of cash, then the price closes at 110.
    broker = make_broker([FLAT, (100, 111, 99, 110)], cash=1_000)
    broker.set_leverage(SYMBOL, 5)
    market(broker, 20)
    broker.next()
    first, second = broker.recorder.account_history().itertuples(index=False)

    assert first.margin == approx(400)
    assert first.gross_exposure == approx(2_000) and first.leverage == approx(2.0)
    assert second.equity == approx(1_200)
    assert second.gross_exposure == approx(2_200)
    assert second.leverage == approx(2_200 / 1_200)
    assert second.cash == 1_000


def test_exposure_is_signed_for_shorts():
    broker = make_broker([FLAT])
    market(broker, -3)
    row = broker.recorder.account_history().iloc[0]

    assert row["net_exposure"] == approx(-300)
    assert row["gross_exposure"] == approx(300)


def test_long_and_short_in_different_symbols_net_out():
    data = Data({"BTC": make_frame([FLAT]), "ETH": make_frame([FLAT])})
    broker = Broker(data, 10_000, 0.0, 0.005, 0, None)
    broker.order("BTC", 2, None, type="MARKET")
    broker.order("ETH", -2, None, type="MARKET")
    broker.next()
    row = broker.recorder.account_history().iloc[0]

    assert row["gross_exposure"] == approx(400)
    assert row["net_exposure"] == approx(0)
    assert row["margin"] == approx(400)


def test_account_equity_matches_the_strategy_equity_curve():
    class Hold(Strategy):
        def init(self):
            pass

        def next(self, i):
            if i == 1:
                self.market_order(SYMBOL, 3)
            if i == 6:
                self.market_order(SYMBOL, -3)

    bars = [(100 + i, 103 + i, 98 + i, 101 + i) for i in range(10)]
    strategy = Hold(data={SYMBOL: make_frame(bars)}, initial_capital=10_000)
    strategy.run()
    history = strategy.fetch_account_history()

    assert history["equity"].tolist() == approx(strategy.equity_curve)
    assert history["cash"].tolist() == approx(strategy.realized_balances)
    assert len(strategy.fetch_position_history(SYMBOL)) == 10


# -------------------------------------------------------- position history ---

def test_position_history_while_holding():
    broker = make_broker([FLAT, (100, 111, 99, 110)])
    broker.set_leverage(SYMBOL, 10)
    market(broker, 2)
    broker.next()
    history = broker.recorder.position_history(SYMBOL)
    last = history.iloc[-1]

    assert last["qty"] == 2 and last["entry_price"] == 100 and last["close"] == 110
    assert last["exposure"] == approx(220)
    assert last["margin"] == approx(20)
    assert last["liquidation_price"] == approx(90.5)
    assert last["unrealized_pnl"] == approx(20)


def test_position_history_while_flat_has_no_entry_or_liquidation_price():
    broker = make_broker([FLAT, FLAT])
    market(broker, 1)
    market(broker, -1)
    last = broker.recorder.position_history(SYMBOL).iloc[-1]

    assert last["qty"] == 0 and last["exposure"] == 0 and last["unrealized_pnl"] == 0
    assert np.isnan(last["entry_price"]) and np.isnan(last["liquidation_price"])


def test_untraded_symbol_still_has_history():
    data = Data({"BTC": make_frame([FLAT]), "ETH": make_frame([FLAT])})
    broker = Broker(data, 10_000, 0.0, 0.005, 0, None)
    broker.next()

    assert broker.recorder.position_history("ETH")["qty"].tolist() == [0.0]
    with pytest.raises(KeyError):
        broker.recorder.position_history("DOGE")


# ------------------------------------------------------------ order status ---

def test_order_status_through_its_life():
    broker = make_broker([FLAT, FLAT])
    filled = broker.order(SYMBOL, 1, None, type="MARKET")
    resting = broker.order(SYMBOL, 1, 50, type="LIMIT")
    cancelled = broker.order(SYMBOL, 1, 51, type="LIMIT")
    assert filled.status.name == "OPEN"

    broker.cancel_order(SYMBOL, cancelled.id)
    broker.next()

    assert filled.status.name == "FILLED"
    assert resting.status.name == "OPEN"
    assert cancelled.status.name == "CANCELLED"
    assert cancelled.cancelled_at == pd.Timestamp("2024-01-01 00:00")
    assert [o.id for o in broker.orders] == [filled.id, resting.id, cancelled.id]


def test_cancel_all_marks_every_order_cancelled():
    broker = make_broker([FLAT])
    orders = [broker.order(SYMBOL, 1, 50 + i, type="LIMIT") for i in range(3)]
    broker.cancel_all_orders(SYMBOL)

    assert {o.status.name for o in orders} == {"CANCELLED"}


def test_orders_rejected_at_placement_are_logged_with_a_reason():
    broker = make_broker([FLAT], cash=100)
    with pytest.raises(InsufficientMargin):
        broker.order(SYMBOL, 50, None, type="MARKET")
    with pytest.raises(StopOrderRejected):
        broker.order(SYMBOL, 0.1, None, type="STOP", stop_price=90)

    assert [o.status.name for o in broker.orders] == ["REJECTED", "REJECTED"]
    assert "margin" in broker.orders[0].reject_reason
    assert "STOP order rejected" in broker.orders[1].reject_reason
    assert broker.open_orders[SYMBOL] == {}


def test_order_rejected_at_fill_is_marked():
    broker = make_broker([FLAT, (100, 100, 80, 85)], cash=1_000)
    broker.order(SYMBOL, 6, 90, type="LIMIT")
    second = broker.order(SYMBOL, 6, 89, type="LIMIT")
    broker.next()
    broker.next()

    assert second.status.name == "REJECTED"
    assert broker.rejected_orders[0]["status"] == "REJECTED"
    assert "margin" in broker.rejected_orders[0]["reject_reason"]


def test_slippage_cost_is_recorded_per_fill():
    broker = make_broker([FLAT, FLAT], slippage=0.01)
    market(broker, 2)     # pays 101 for 100
    market(broker, -2)    # receives 99 for 100

    assert [t["slippage"] for t in broker.trades] == approx([2.0, 2.0])


def test_no_slippage_cost_without_a_slippage_model():
    broker = make_broker([FLAT])
    market(broker, 2)
    assert broker.trades[0]["slippage"] == 0


def test_trade_log_carries_status_as_text():
    broker = make_broker([FLAT])
    market(broker, 1)
    assert broker.trades[0]["status"] == "FILLED"


# ---------------------------------------------------------------- strategy ---

def test_strategy_exposes_the_recorded_tables():
    class Trader(Strategy):
        def init(self):
            pass

        def next(self, i):
            if i == 0:
                self.market_order(SYMBOL, 1, label="in")
                self.limit_order(SYMBOL, 1, 10)
            if i == 3:
                self.cancel_all_orders(SYMBOL)
                self.market_order(SYMBOL, -1, label="out")

    bars = [(100 + 5 * i, 104 + 5 * i, 98 + 5 * i, 102 + 5 * i) for i in range(6)]
    strategy = Trader(data={SYMBOL: make_frame(bars)}, initial_capital=10_000, commission_rate=0.0)
    strategy.run()

    round_trips = strategy.fetch_round_trips()
    assert len(round_trips) == 1
    assert round_trips.loc[0, "pnl"] == approx(15)
    assert round_trips.loc[0, "entry_label"] == "in"

    orders = strategy.fetch_orders()
    assert sorted(orders["status"]) == ["CANCELLED", "FILLED", "FILLED"]
    assert len(strategy.fetch_account_history()) == 6
