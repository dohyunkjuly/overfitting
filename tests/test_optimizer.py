"""Strategy parameters, the no-recording mode, and the parameter optimizer."""
import multiprocessing

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from overfitting import Strategy, optimize
from overfitting.analysis import metrics
from overfitting.optimizer import METRICS, Params, _daily_groups, _measure

approx = pytest.approx

# Falls from 100 to 60 over 40 bars, then rises to 140: the best bar to buy is bar 40.
V_SHAPE = [100 - i for i in range(40)] + [60 + i for i in range(81)]
V_BARS = [(c, c + 0.5, c - 0.5, c) for c in V_SHAPE]


class BuyAt(Strategy):
    """Buys `size` units on bar `entry` and holds."""
    entry = 0
    size = 1

    def init(self):
        pass

    def next(self, i):
        if i == self.entry:
            self.market_order("BTC", self.size)


class Swing(Strategy):
    """Buys every `period` bars and sells `hold` bars later."""
    period = 10
    hold = 5

    def init(self):
        pass

    def next(self, i):
        qty = float(self.get_position("BTC").qty)
        if i % self.period == 0 and qty == 0:
            self.market_order("BTC", 1)
        elif i % self.period == self.hold and qty > 0:
            self.market_order("BTC", -qty)


def frame(bars=V_BARS):
    return {"BTC": make_frame(bars)}


def noisy(n=24 * 80, seed=0):
    closes = 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0.0003, 0.01, n))
    return {"BTC": make_frame([(c, c * 1.003, c * 0.997, c) for c in closes])}


# ------------------------------------------------------- strategy parameters ---

def test_parameters_are_class_attributes_overridden_per_run():
    default = BuyAt(data=frame())
    custom = BuyAt(data=frame(), params={"entry": 7, "size": 3})

    assert (default.entry, default.size) == (0, 1)
    assert (custom.entry, custom.size) == (7, 3)
    assert custom.params == {"entry": 7, "size": 3}
    assert BuyAt.entry == 0                               # the class itself is untouched


def test_parameters_are_set_before_init_runs():
    seen = []

    class Reads(Strategy):
        window = 5

        def init(self):
            seen.append(self.window)

        def next(self):
            pass

    Reads(data=frame(), params={"window": 9})
    assert seen == [9]


def test_unknown_parameter_is_rejected_with_a_hint():
    with pytest.raises(AttributeError, match="has no parameter 'entyr'"):
        BuyAt(data=frame(), params={"entyr": 3})


def test_data_can_be_prepared_once_and_shared():
    from overfitting.data import Data

    shared = Data(frame())
    a, b = BuyAt(data=shared, params={"entry": 1}), BuyAt(data=shared, params={"entry": 2})
    assert a.data is shared and b.data is shared
    assert a.run().sum() != b.run().sum()


# ----------------------------------------------------------- record=False ---

def test_without_recording_returns_are_identical_and_tables_empty():
    recorded = Swing(data=noisy(), initial_capital=10_000)
    bare = Swing(data=noisy(), initial_capital=10_000, record=False)
    r1, r2 = recorded.run(), bare.run()

    assert (r1.to_numpy() == r2.to_numpy()).all()
    assert bare.get_balance() == recorded.get_balance()
    assert len(bare.fetch_trades()) == len(recorded.fetch_trades()) > 0      # fills are still logged
    assert len(recorded.fetch_round_trips()) > 0 and bare.fetch_round_trips().empty
    assert bare.fetch_account_history().empty and bare.fetch_position_history("BTC").empty
    assert list(bare.fetch_round_trips().columns) == list(recorded.fetch_round_trips().columns)
    with pytest.raises(KeyError):
        bare.fetch_position_history("DOGE")


# ----------------------------------------------------------------- measuring ---

def test_measure_matches_the_metrics_module_on_daily_returns():
    strategy = Swing(data=noisy(), initial_capital=10_000)
    returns = strategy.run()
    daily = metrics.aggregate_returns(returns, "D")
    measured = _measure(returns.to_numpy(), _daily_groups(returns.index), 7)

    assert set(measured) == set(METRICS)
    assert measured["sharpe_ratio"] == approx(metrics.sharpe_ratio(daily))
    assert measured["total_return"] == approx(metrics.total_return(returns))
    assert measured["max_drawdown"] == approx(metrics.max_drawdown(daily))
    assert measured["calmar_ratio"] == approx(metrics.calmar_ratio(daily))
    assert measured["fills"] == 7


def test_params_read_by_key_or_attribute():
    p = Params(fast=5, slow=20)
    assert p.fast == 5 and p["slow"] == 20
    with pytest.raises(AttributeError):
        p.missing


# ------------------------------------------------------------------ optimize ---

def test_grid_search_finds_the_best_entry():
    result = optimize(BuyAt, frame(), {"entry": range(0, 120, 5)}, maximize="total_return",
                      test_size=0, workers=1, initial_capital=1_000)

    assert result.best_params == {"entry": 40}            # the bottom of the V
    assert result.trials == 24 and len(result.table) == 24
    assert result.table["entry"].iloc[0] == 40
    assert result.table["objective"].is_monotonic_decreasing
    assert result.test is None and result.split_time is None


def test_table_has_parameters_metrics_and_objective():
    result = optimize(Swing, noisy(), {"period": [8, 12], "hold": [3, 5]}, workers=1)

    assert list(result.table.columns) == ["period", "hold", *METRICS, "objective"]
    assert (result.table["objective"] == result.table["sharpe_ratio"]).all()
    assert isinstance(result.best_params["period"], int)  # plain Python values, not numpy


def test_constraint_skips_combinations():
    result = optimize(Swing, noisy(), {"period": [6, 10, 14], "hold": [4, 8, 12]},
                      constraint=lambda p: p.hold < p.period, workers=1)

    assert result.trials == 6
    assert (result.table["hold"] < result.table["period"]).all()
    with pytest.raises(ValueError, match="No parameter combination"):
        optimize(Swing, noisy(), {"period": [6]}, constraint=lambda p: False, workers=1)


def test_maximize_another_metric_or_a_function():
    data = noisy()
    grid = {"period": [6, 10, 14, 20], "hold": [3, 5]}
    by_drawdown = optimize(Swing, data, grid, maximize="max_drawdown", workers=1)
    custom = optimize(Swing, data, grid, maximize=lambda m: -m["fills"], workers=1)

    assert by_drawdown.table["max_drawdown"].iloc[0] == by_drawdown.table["max_drawdown"].max()
    assert custom.table["fills"].iloc[0] == custom.table["fills"].min()   # fewest fills wins
    assert custom.maximize == "<lambda>"


def test_random_search_tries_n_combinations_reproducibly():
    data = noisy()
    grid = {"period": range(6, 30), "hold": range(2, 6)}
    first = optimize(Swing, data, grid, method="random", n=10, seed=3, workers=1)
    again = optimize(Swing, data, grid, method="random", n=10, seed=3, workers=1)
    other = optimize(Swing, data, grid, method="random", n=10, seed=4, workers=1)

    assert first.trials == 10 and first.method == "random"
    assert first.table.equals(again.table)
    assert not first.table[["period", "hold"]].equals(other.table[["period", "hold"]])
    everything = optimize(Swing, data, {"period": [6, 8], "hold": [2]}, method="random", n=50, workers=1)
    assert everything.trials == 2                         # n above the grid size tries the whole grid


def test_strategy_arguments_are_passed_through():
    data = noisy()
    free = optimize(Swing, data, {"period": [10]}, workers=1, commission_rate=0.0)
    costly = optimize(Swing, data, {"period": [10]}, workers=1, commission_rate=0.01)

    assert costly.train["total_return"] < free.train["total_return"]


# --------------------------------------------------------- train and test ---

def test_search_only_sees_the_train_period():
    # Over all the data the best entry is bar 40. With the last 70% held out, the
    # search sees bars 0 to 35 only, where the price just falls: buying latest loses least.
    result = optimize(BuyAt, frame(), {"entry": range(0, 36, 5)}, maximize="total_return",
                      test_size=0.7, workers=1, initial_capital=1_000)

    assert result.best_params == {"entry": 35}
    assert result.split_time == pd.Timestamp("2024-01-01") + pd.Timedelta(hours=36)
    assert result.train["total_return"] < 0 < result.test["total_return"]


def test_not_trading_beats_losing():
    # Entries after bar 35 never happen inside the train period, so they return 0,
    # which is better than every entry that bought into the fall.
    result = optimize(BuyAt, frame(), {"entry": [10, 20, 30, 60]}, maximize="total_return",
                      test_size=0.7, workers=1, initial_capital=1_000)

    assert result.best_params == {"entry": 60}
    assert result.train["total_return"] == 0 and result.train["fills"] == 0


def test_winner_is_rerun_on_all_data_with_recording():
    data = noisy()
    result = optimize(Swing, data, {"period": [8, 12, 16], "hold": [3, 5]}, workers=1)

    assert len(result.returns) == len(data["BTC"])
    assert result.strategy.params == result.best_params
    assert len(result.strategy.fetch_round_trips()) > 0   # full tables for the report
    assert result.train["fills"] + result.test["fills"] == len(result.strategy.fetch_trades())


def test_train_metrics_equal_the_winning_row_of_the_table():
    # The full run's first 70% is the same backtest the search ran, bar for bar.
    result = optimize(Swing, noisy(), {"period": [8, 12, 16], "hold": [3, 5]}, workers=1)
    row = result.table.iloc[0]

    for name in ("total_return", "sharpe_ratio", "max_drawdown", "fills", "objective"):
        assert result.train[name] == approx(row[name])


def test_test_metrics_come_from_the_held_out_returns():
    data = noisy()
    result = optimize(Swing, data, {"period": [10], "hold": [5]}, test_size=0.25, workers=1)
    split = len(data["BTC"]) - round(len(data["BTC"]) * 0.25)
    held_out = metrics.aggregate_returns(result.returns.iloc[split:], "D")

    assert result.test["total_return"] == approx(metrics.total_return(held_out))
    assert result.test["sharpe_ratio"] == approx(metrics.sharpe_ratio(held_out))


# ---------------------------------------------------------------- edge cases ---

def test_undefined_objectives_never_win():
    class Sometimes(Strategy):
        trade = 0

        def init(self):
            pass

        def next(self, i):
            if self.trade and i % 20 == 0:
                self.market_order("BTC", 1 if (i // 20) % 2 == 0 else -1)

    result = optimize(Sometimes, noisy(), {"trade": [0, 1]}, workers=1)
    assert result.best_params == {"trade": 1}
    assert np.isnan(result.table["objective"].iloc[-1])   # never traded: no Sharpe ratio

    with pytest.raises(ValueError, match="undefined for every combination"):
        optimize(Sometimes, noisy(), {"trade": [0]}, workers=1)


def test_bad_arguments_are_rejected():
    data = noisy()
    with pytest.raises(AttributeError, match="has no parameter 'peroid'"):
        optimize(Swing, data, {"peroid": [5]})
    with pytest.raises(ValueError, match="maximize must be one of"):
        optimize(Swing, data, {"period": [5]}, maximize="profit")
    with pytest.raises(ValueError, match="test_size"):
        optimize(Swing, data, {"period": [5]}, test_size=1.0)
    with pytest.raises(ValueError, match="needs n"):
        optimize(Swing, data, {"period": [5]}, method="random")
    with pytest.raises(ValueError, match="'grid' or 'random'"):
        optimize(Swing, data, {"period": [5]}, method="annealing")
    with pytest.raises(ValueError, match="at least one parameter"):
        optimize(Swing, data, {})
    with pytest.raises(ValueError, match="no values"):
        optimize(Swing, data, {"period": []})


def test_a_single_value_can_be_given_without_a_list():
    result = optimize(Swing, noisy(), {"period": 10, "hold": [3, 5]}, workers=1)
    assert result.trials == 2 and set(result.table["period"]) == {10}


# ------------------------------------------------------------------ parallel ---

@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="parallel search needs the fork start method")
def test_parallel_search_gives_the_same_table_as_one_core():
    data = noisy()
    grid = {"period": range(6, 26, 2), "hold": [2, 3, 4]}
    serial = optimize(Swing, data, grid, workers=1)
    parallel = optimize(Swing, data, grid, workers=3)

    assert parallel.trials == 30
    assert parallel.best_params == serial.best_params
    key = ["period", "hold"]
    pd.testing.assert_frame_equal(serial.table.sort_values(key).reset_index(drop=True),
                                  parallel.table.sort_values(key).reset_index(drop=True))


def test_errors_inside_the_strategy_are_not_swallowed():
    class Broken(Strategy):
        x = 1

        def init(self):
            pass

        def next(self, i):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        optimize(Broken, noisy(240), {"x": range(40)}, workers=2)


# ------------------------------------------------------------------- output ---

def test_heatmap_is_a_grid_of_two_parameters():
    result = optimize(Swing, noisy(), {"period": [8, 12, 16], "hold": [3, 5]}, workers=1)
    grid = result.heatmap("period", "hold")

    assert list(grid.columns) == [8, 12, 16] and list(grid.index) == [3, 5]
    best = result.best_params
    assert grid.loc[best["hold"], best["period"]] == approx(result.table["objective"].iloc[0])
    assert result.heatmap("period", "hold", metric="fills").loc[3, 8] > 0
    with pytest.raises(KeyError):
        result.heatmap("period", "speed")


def test_printed_result_shows_trials_parameters_and_both_periods():
    text = str(optimize(Swing, noisy(), {"period": [8, 12], "hold": [3, 5]}, workers=1))

    assert "Optimization of Swing" in text
    assert "4 (grid)" in text and "sharpe_ratio" in text
    assert "Train (70%)" in text and "Test (30%)" in text
    assert "best of 4 tries" in text and "played no part" in text


def test_printed_result_warns_when_nothing_was_held_out():
    text = str(optimize(Swing, noisy(), {"period": [8, 12]}, test_size=0, workers=1))

    assert "Train (100%)" in text and "Test" not in text
    assert "No data was held out" in text
