"""The Strategy loop: order helpers, data accessors, run() output and the trade log."""
from decimal import Decimal

import pandas as pd
import pytest

from conftest import make_frame
from overfitting import Strategy
from overfitting.indicators import SMA

# Opens: 100, 110, 120, 130, 140
RALLY = [(100 + 10 * i, 105 + 10 * i, 95 + 10 * i, 102 + 10 * i) for i in range(5)]


class Scripted(Strategy):
    """Runs `actions[i](self)` on bar i, so each test can script its own trades."""
    actions = {}

    def init(self):
        pass

    def next(self, i):
        action = self.actions.get(i)
        if action:
            action(self)


def run(actions, bars=RALLY, symbols=("BTC",), **kwargs):
    cls = type("Case", (Scripted,), {"actions": actions})
    kwargs.setdefault("commission_rate", 0.0)
    kwargs.setdefault("initial_capital", 10_000)
    strategy = cls(data={s: make_frame(bars) for s in symbols}, **kwargs)
    returns = strategy.run()
    return strategy, returns


# ----------------------------------------------------------------- run() ---

def test_run_returns_one_value_per_bar_indexed_by_time():
    _, returns = run({})

    assert isinstance(returns, pd.Series)
    assert len(returns) == len(RALLY)
    assert returns.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert returns.index[-1] == pd.Timestamp("2024-01-01 04:00")


def test_idle_strategy_has_flat_balance_and_zero_returns():
    strategy, returns = run({})

    assert strategy.balances == [10_000] * 5
    assert (returns == 0).all()
    assert strategy.fetch_trades().empty


def test_buy_then_sell_books_profit_on_the_closing_bar():
    strategy, returns = run({
        0: lambda s: s.market_order("BTC", 1),   # buys at 100
        3: lambda s: s.market_order("BTC", -1),  # sells at 130
    })

    assert strategy.balances == [10_000, 10_000, 10_000, 10_030, 10_030]
    assert returns.iloc[3] == pytest.approx(30 / 10_000)
    assert returns.iloc[4] == 0
    assert strategy.get_balance() == 10_030


def test_commission_reduces_balance_on_the_entry_bar():
    strategy, _ = run({0: lambda s: s.market_order("BTC", 1)}, commission_rate=0.001)

    assert strategy.balances[0] == pytest.approx(10_000 - 0.1)


def test_default_arguments_match_the_readme():
    strategy = Scripted(data={"BTC": make_frame(RALLY)})

    assert strategy.broker.cash == 100_000
    assert strategy.broker.commission_rate == 0.0002
    assert strategy.broker.maint_margin_rate == 0.005
    assert strategy.broker.maint_amount == 0


@pytest.mark.xfail(strict=True, reason="ROADMAP Phase 2: equity curve is realized-only, not mark-to-market")
def test_equity_curve_includes_unrealized_pnl():
    strategy, _ = run({0: lambda s: s.market_order("BTC", 1)})  # never closed

    # Long 1 from 100; bar 2 closes at 122, so equity should be 10,022.
    assert strategy.balances[2] == pytest.approx(10_022)


# ---------------------------------------------------------- order helpers ---

def test_limit_order_helper():
    strategy, _ = run({0: lambda s: s.limit_order("BTC", 1, 96)})  # bar 0 low is 95
    trades = strategy.fetch_trades()

    assert len(trades) == 1
    assert trades.loc[0, "type"] == "LIMIT"
    assert trades.loc[0, "executed_price"] == 96


def test_stop_market_order_helper():
    strategy, _ = run({0: lambda s: s.stop_market_order("BTC", 1, 104)})  # bar 0 high is 105
    trades = strategy.fetch_trades()

    assert trades.loc[0, "type"] == "STOP"
    assert trades.loc[0, "executed_price"] == 104


def test_stop_limit_order_helper():
    # Stop 104 triggers on bar 0 (high 105); limit 96 fills there too (low 95).
    strategy, _ = run({0: lambda s: s.stop_limit_order("BTC", 1, 96, 104)})
    trades = strategy.fetch_trades()

    assert trades.loc[0, "executed_price"] == 96


def test_order_label_reaches_the_trade_log():
    strategy, _ = run({0: lambda s: s.market_order("BTC", 1, label="entry")})

    assert strategy.fetch_trades().loc[0, "label"] == "entry"


def test_cancel_order_and_open_orders():
    seen = {}

    def place(s):
        order = s.limit_order("BTC", 1, 50)
        seen["before"] = s.get_open_orders("BTC")
        s.cancel_order("BTC", order.id)
        seen["after"] = s.get_open_orders("BTC")

    strategy, _ = run({0: place})

    assert len(seen["before"]) == 1
    assert seen["after"] == {}
    assert strategy.fetch_trades().empty


def test_cancel_all_orders():
    def place(s):
        s.limit_order("BTC", 1, 50)
        s.limit_order("BTC", 1, 51)
        s.cancel_all_orders("BTC")

    strategy, _ = run({0: place})

    assert strategy.get_open_orders("BTC") == {}


def test_close_all_positions():
    strategy, _ = run({
        0: lambda s: s.market_order("BTC", 2),
        2: lambda s: s.close_all_positions("BTC"),  # sells at 120
    })

    assert strategy.get_position("BTC").qty == 0
    assert strategy.get_balance() == 10_040


def test_set_leverage_sets_liquidation_price():
    strategy, _ = run({0: lambda s: (s.set_leverage("BTC", 10), s.market_order("BTC", 1))})
    position = strategy.get_position("BTC")

    assert position.leverage == 10
    assert position.liquid_price == pytest.approx(90.5)


# --------------------------------------------------------- data accessors ---

def test_price_accessors_return_scalar_or_series():
    strategy, _ = run({})

    assert strategy.open("BTC", 1) == 110
    assert strategy.high("BTC", 1) == 115
    assert strategy.low("BTC", 1) == 105
    assert strategy.close("BTC", 1) == 112
    assert strategy.bars("BTC", 1) == (110, 115, 105, 112)
    assert list(strategy.close("BTC")) == [102, 112, 122, 132, 142]


def test_val_reads_any_column_and_rejects_unknown_ones():
    strategy, _ = run({})

    assert strategy.val("BTC", 2, "close") == 122
    with pytest.raises(AttributeError):
        strategy.val("BTC", 2, "funding")


def test_indicator_binds_the_symbol_series():
    strategy, _ = run({})
    sma = strategy.indicator(SMA, "BTC", window=2)

    # SMA is shifted one bar, so the value at bar i only uses closes before i.
    assert pd.isna(sma[0]) and pd.isna(sma[1])
    assert sma[2] == pytest.approx((102 + 112) / 2)
    assert sma.ready(2) and not sma.ready(1)


def test_indicator_source_and_bad_arguments():
    strategy, _ = run({})
    sma_high = strategy.indicator(SMA, "BTC", source="high", window=2)

    assert sma_high[2] == pytest.approx((105 + 115) / 2)
    with pytest.raises(ValueError):
        strategy.indicator(SMA, "BTC", source="vwap")
    with pytest.raises(TypeError):
        strategy.indicator(SMA, "BTC", windw=2)


# ------------------------------------------------------- multiple symbols ---

def test_symbols_are_traded_independently():
    strategy, _ = run(
        {
            0: lambda s: (s.market_order("BTC", 1), s.market_order("ETH", -2)),
            2: lambda s: s.market_order("BTC", -1),
        },
        symbols=("BTC", "ETH"),
    )

    assert strategy.get_position("BTC").qty == 0
    assert strategy.get_position("ETH").qty == Decimal("-2")
    assert strategy.get_balance() == 10_020  # only the BTC leg is realized
    assert list(strategy.fetch_trades()["symbol"]) == ["BTC", "ETH", "BTC"]


# --------------------------------------------------------------- trade log ---

def test_fetch_trades_columns():
    strategy, _ = run({0: lambda s: s.market_order("BTC", 1)})
    trades = strategy.fetch_trades()

    for column in ("id", "created_at", "executed_at", "symbol", "qty", "side", "price",
                   "type", "stop_price", "executed_price", "commission", "pnl",
                   "realized_pnl", "label"):
        assert column in trades.columns


def test_save_trades_to_csv(tmp_path):
    strategy, _ = run({
        0: lambda s: s.market_order("BTC", 1),
        1: lambda s: s.market_order("BTC", -1),
    })
    strategy.save_trades_to_csv(path=str(tmp_path), filename="trades")
    saved = pd.read_csv(tmp_path / "trades.csv")

    assert len(saved) == 2
    assert list(saved["executed_price"]) == [100, 110]
