"""Metrics are checked against values worked out by hand from their definitions."""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import SYMBOL, make_frame
from overfitting import Strategy
from overfitting.analysis import metrics as m
from overfitting.analysis.report import Report

approx = pytest.approx


def series(values, start="2024-01-01", freq="D"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


# +10%, -10%, +20%, -5%: equity 1.1, 0.99, 1.188, 1.1286
MIXED = series([0.10, -0.10, 0.20, -0.05])
EMPTY = pd.Series([], dtype=float)


# ------------------------------------------------------------------ returns ---

def test_equity_curve_and_total_return():
    assert list(m.equity_curve(MIXED)) == approx([1.1, 0.99, 1.188, 1.1286])
    assert m.equity_curve(MIXED, start=1000).iloc[-1] == approx(1128.6)
    assert m.total_return(MIXED) == approx(0.1286)


def test_cagr_compounds_to_one_year():
    # 1% a period for 2 periods, 4 periods a year: growth 1.01^2 over half a year.
    assert m.cagr(series([0.01, 0.01]), periods_per_year=4) == approx(1.01 ** 4 - 1)
    # A full year of data: CAGR equals the total return.
    assert m.cagr(MIXED, periods_per_year=4) == approx(0.1286)


def test_cagr_is_undefined_after_a_wipe_out():
    assert math.isnan(m.cagr(series([0.5, -1.0, 0.2])))


def test_annualized_mean():
    assert m.annualized_mean(MIXED, periods_per_year=4) == approx(0.15)


def test_best_worst_and_period_shares():
    assert m.best(MIXED) == 0.20
    assert m.worst(MIXED) == -0.10
    assert m.positive_share(MIXED) == 0.5
    assert m.average_positive(MIXED) == approx(0.15)
    assert m.average_negative(MIXED) == approx(-0.075)


def test_longest_streak():
    r = series([0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.0])
    assert m.longest_streak(r, positive=True) == 3
    assert m.longest_streak(r, positive=False) == 2
    assert m.longest_streak(EMPTY) == 0


def test_aggregate_returns_compounds_within_each_period():
    hourly = series([0.01] * 48, freq="h")
    daily = m.aggregate_returns(hourly, "D")

    assert len(daily) == 2
    assert daily.iloc[0] == approx(1.01 ** 24 - 1)
    assert m.total_return(daily) == approx(m.total_return(hourly))


def test_aggregate_returns_skips_empty_periods_and_needs_dates():
    gappy = pd.Series([0.1, 0.1], index=pd.to_datetime(["2024-01-01", "2024-01-05"]))
    assert len(m.aggregate_returns(gappy, "D")) == 2
    with pytest.raises(TypeError):
        m.aggregate_returns(pd.Series([0.1, 0.2]), "D")


def test_calendar_returns_table():
    daily = series([0.01] * 60, start="2024-01-01")  # January (31 days) and February (29)
    table = m.calendar_returns(daily)

    assert list(table.columns) == list(range(1, 13))
    assert table.loc[2024, 1] == approx(1.01 ** 31 - 1)
    assert table.loc[2024, 2] == approx(1.01 ** 29 - 1)
    assert math.isnan(table.loc[2024, 3])


def test_trailing_return():
    daily = series([0.5] + [0.01] * 10)
    assert m.trailing_return(daily, "10D") == approx(1.01 ** 10 - 1)


# --------------------------------------------------------------------- risk ---

def test_volatility():
    sd = np.std([0.10, -0.10, 0.20, -0.05], ddof=1)
    assert m.volatility(MIXED, periods_per_year=1) == approx(sd)
    assert m.volatility(MIXED, periods_per_year=4) == approx(sd * 2)


def test_downside_deviation_counts_gains_as_zero():
    # shortfalls: 0, -0.10, 0, -0.05 -> sqrt((0.01 + 0.0025) / 4)
    assert m.downside_deviation(MIXED, periods_per_year=1) == approx(math.sqrt(0.0125 / 4))


def test_sharpe_ratio():
    mean, sd = 0.0375, np.std([0.10, -0.10, 0.20, -0.05], ddof=1)
    assert m.sharpe_ratio(MIXED, periods_per_year=1) == approx(mean / sd)
    assert m.sharpe_ratio(MIXED, periods_per_year=4) == approx(mean / sd * 2)
    assert m.sharpe_ratio(MIXED, risk_free=0.0375, periods_per_year=1) == approx(0)


def test_sharpe_ratio_is_undefined_without_variation():
    assert math.isnan(m.sharpe_ratio(series([0.01, 0.01, 0.01])))
    assert math.isnan(m.sharpe_ratio(series([0.01])))


def test_sortino_ratio():
    assert m.sortino_ratio(MIXED, periods_per_year=1) == approx(0.0375 / math.sqrt(0.0125 / 4))
    assert math.isnan(m.sortino_ratio(series([0.01, 0.02, 0.03])))  # no downside


def test_omega_and_gain_to_pain():
    # gains 0.30, losses 0.15
    assert m.omega_ratio(MIXED) == approx(2.0)
    assert m.gain_to_pain_ratio(MIXED) == approx(0.15 / 0.15)
    assert math.isnan(m.omega_ratio(series([0.01, 0.02])))


def test_tail_and_common_sense_ratio():
    r = series(np.linspace(-0.10, 0.20, 101))  # 5th pct -0.085, 95th pct 0.185
    assert m.tail_ratio(r) == approx(0.185 / 0.085)

    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    assert m.common_sense_ratio(r) == approx(0.185 / 0.085 * gains / losses)


def test_value_at_risk_and_expected_shortfall_historical():
    r = series(np.arange(-50, 50) / 1000)  # -0.050 ... 0.049, 100 values
    var = m.value_at_risk(r, 0.95)

    assert var == approx(np.quantile(r, 0.05))
    assert m.expected_shortfall(r, 0.95) == approx(r[r <= var].mean())
    assert m.expected_shortfall(r, 0.95) < var < 0


def test_value_at_risk_and_expected_shortfall_parametric():
    r = series(np.arange(-50, 50) / 1000)
    mean, sd = r.mean(), r.std(ddof=1)

    assert m.value_at_risk(r, 0.95, "parametric") == approx(mean - 1.6448536 * sd)
    # normal density at -1.645 is 0.103136
    assert m.expected_shortfall(r, 0.95, "parametric") == approx(mean - sd * 0.1031356 / 0.05)
    with pytest.raises(ValueError):
        m.value_at_risk(r, 0.95, "magic")


def test_skewness_and_kurtosis_of_known_shapes():
    rng = np.random.default_rng(0)
    normal = series(rng.normal(0, 0.01, 50_000))
    assert m.skewness(normal) == approx(0, abs=0.05)
    assert m.kurtosis(normal) == approx(0, abs=0.1)

    left_tailed = series([0.01] * 99 + [-0.5])
    assert m.skewness(left_tailed) < -5
    assert m.kurtosis(left_tailed) > 50
    assert m.skewness(series([0.1, -0.1, 0.0])) == approx(0)


def test_jarque_bera_separates_normal_from_fat_tailed():
    rng = np.random.default_rng(1)
    _, p_normal = m.jarque_bera(series(rng.normal(0, 0.01, 2_000)))
    _, p_fat = m.jarque_bera(series(rng.standard_t(2, 2_000) * 0.01))

    assert p_normal > 0.01
    assert p_fat < 0.001


def test_autocorrelation():
    assert m.autocorrelation(series([0.1, -0.1] * 20), 1) == approx(-1)
    assert m.autocorrelation(series([0.1, -0.1] * 20), 2) == approx(1)


def test_kelly_fraction():
    assert m.kelly_fraction(MIXED) == approx(0.0375 / np.var([0.10, -0.10, 0.20, -0.05], ddof=1))


# ----------------------------------------------------------------- drawdown ---

def test_drawdown_series():
    # equity 1.1, 0.99, 1.188, 1.1286 against peaks 1.1, 1.1, 1.188, 1.188
    assert list(m.drawdown_series(MIXED)) == approx([0, -0.1, 0, -0.05])
    assert m.max_drawdown(MIXED) == approx(-0.1)
    assert m.current_drawdown(MIXED) == approx(-0.05)


def test_starting_value_counts_as_a_peak():
    # Losing on the first period is a drawdown from the starting capital.
    r = series([-0.2, 0.1])
    assert list(m.drawdown_series(r)) == approx([-0.2, -0.12])
    assert m.max_drawdown(r) == approx(-0.2)


def test_no_drawdown_when_always_rising():
    r = series([0.01, 0.02, 0.0, 0.03])
    assert m.max_drawdown(r) == 0
    assert m.drawdown_table(r).empty
    assert m.longest_drawdown(r) == 0
    assert m.time_underwater(r) == 0


# equity: 1.10, 0.88, 0.968, 1.1616, 1.1616, 0.92928, 0.836352
EPISODES = series([0.10, -0.20, 0.10, 0.20, 0.0, -0.20, -0.10])


def test_drawdown_table():
    table = m.drawdown_table(EPISODES)
    assert len(table) == 2

    deepest = table.iloc[0]   # the unrecovered one at the end: 0.836352 / 1.1616 - 1
    assert deepest["depth"] == approx(-0.28)
    assert deepest["peak"] == pd.Timestamp("2024-01-05")
    assert deepest["valley"] == pd.Timestamp("2024-01-07")
    assert pd.isna(deepest["recovery"])
    assert deepest["duration"] == 2
    assert pd.isna(deepest["to_recovery"])

    first = table.iloc[1]     # 0.88 / 1.10 - 1, recovered on the 4th
    assert first["depth"] == approx(-0.20)
    assert first["peak"] == pd.Timestamp("2024-01-01")
    assert first["valley"] == pd.Timestamp("2024-01-02")
    assert first["recovery"] == pd.Timestamp("2024-01-04")
    assert first["duration"] == 2
    assert first["to_valley"] == 1
    assert first["to_recovery"] == 2

    assert len(m.drawdown_table(EPISODES, top=1)) == 1


def test_drawdown_statistics():
    assert m.average_drawdown(EPISODES) == approx(-0.24)
    assert m.longest_drawdown(EPISODES) == 2
    assert m.average_drawdown_duration(EPISODES) == approx(2)
    assert m.average_recovery_time(EPISODES) == approx(2)
    assert m.time_underwater(EPISODES) == approx(4 / 7)
    assert m.drawdown_count(EPISODES) == 2
    assert m.drawdown_count(EPISODES, 0.25) == 1
    assert m.drawdown_count(EPISODES, 0.30) == 0


def test_ulcer_and_pain_index():
    dd = np.array([0, -0.1, 0, -0.05])
    assert m.ulcer_index(MIXED) == approx(math.sqrt((dd ** 2).mean()))
    assert m.pain_index(MIXED) == approx(0.0375)


def test_drawdown_based_ratios():
    assert m.calmar_ratio(MIXED, periods_per_year=4) == approx(0.1286 / 0.1)
    assert m.recovery_factor(MIXED) == approx(0.1286 / 0.1)
    assert m.martin_ratio(MIXED, periods_per_year=4) == approx(0.1286 / m.ulcer_index(MIXED))
    assert math.isnan(m.calmar_ratio(series([0.01, 0.02])))  # no drawdown to divide by


# ---------------------------------------------------------------- benchmark ---

BENCH = series([0.02, -0.01, 0.03, -0.02, 0.01, 0.00])


def test_beta_alpha_of_a_levered_benchmark():
    strategy = 2 * BENCH + 0.001  # twice the benchmark plus 0.1% a period
    assert m.beta(strategy, BENCH) == approx(2)
    assert m.alpha(strategy, BENCH, periods_per_year=100) == approx(0.1)
    assert m.correlation(strategy, BENCH) == approx(1)
    assert m.r_squared(strategy, BENCH) == approx(1)


def test_benchmark_against_itself():
    assert m.beta(BENCH, BENCH) == approx(1)
    assert m.alpha(BENCH, BENCH) == approx(0, abs=1e-12)
    assert m.tracking_error(BENCH, BENCH) == approx(0, abs=1e-12)
    assert m.excess_return(BENCH, BENCH) == approx(0)
    assert m.outperformance_share(BENCH, BENCH) == 0
    assert m.up_capture(BENCH, BENCH) == approx(1)
    assert m.down_capture(BENCH, BENCH) == approx(1)


def test_inverse_strategy_has_negative_beta():
    assert m.beta(-BENCH, BENCH) == approx(-1)
    assert m.correlation(-BENCH, BENCH) == approx(-1)


def test_tracking_error_and_information_ratio():
    strategy = BENCH + series([0.01, -0.01, 0.02, 0.00, 0.01, -0.01])
    active = np.array([0.01, -0.01, 0.02, 0.00, 0.01, -0.01])

    assert m.tracking_error(strategy, BENCH, periods_per_year=1) == approx(active.std(ddof=1))
    assert m.information_ratio(strategy, BENCH, periods_per_year=1) == approx(
        active.mean() / active.std(ddof=1))
    assert m.outperformance_share(strategy, BENCH) == approx(3 / 6)


def test_capture_ratios():
    # Gains half of each up period, loses a quarter of each down period.
    strategy = BENCH.where(BENCH <= 0, BENCH / 2).where(BENCH >= 0, BENCH / 4)
    # Ratio of geometric average returns over the 3 up periods and the 2 down periods.
    up = ((1.01 * 1.015 * 1.005) ** (1 / 3) - 1) / ((1.02 * 1.03 * 1.01) ** (1 / 3) - 1)
    down = ((0.9975 * 0.995) ** (1 / 2) - 1) / ((0.99 * 0.98) ** (1 / 2) - 1)

    assert m.up_capture(strategy, BENCH) == approx(up)
    assert m.down_capture(strategy, BENCH) == approx(down)
    assert 0.45 < m.up_capture(strategy, BENCH) < 0.55
    assert 0.2 < m.down_capture(strategy, BENCH) < 0.3


def test_capture_ratio_does_not_shrink_with_sample_length():
    rng = np.random.default_rng(4)
    bench = series(rng.normal(0.001, 0.03, 2_000))
    half = bench / 2

    assert m.up_capture(half, bench) == approx(0.5, abs=0.02)
    assert m.down_capture(half, bench) == approx(0.5, abs=0.02)


def test_benchmark_metrics_use_only_shared_periods():
    longer = series([0.5] + list(BENCH), start="2023-12-31")
    assert m.beta(BENCH, longer) == approx(1)


# ------------------------------------------------------------------ rolling ---

def test_rolling_metrics_match_the_plain_ones_on_each_window():
    rng = np.random.default_rng(2)
    r = series(rng.normal(0.001, 0.02, 60))
    b = series(rng.normal(0.0005, 0.02, 60))
    last = r.iloc[-20:]

    assert m.rolling_volatility(r, 20).iloc[-1] == approx(m.volatility(last))
    assert m.rolling_sharpe(r, 20).iloc[-1] == approx(m.sharpe_ratio(last))
    assert m.rolling_sortino(r, 20).iloc[-1] == approx(m.sortino_ratio(last))
    assert m.rolling_beta(r, b, 20).iloc[-1] == approx(m.beta(last, b.iloc[-20:]))
    assert m.rolling_correlation(r, b, 20).iloc[-1] == approx(m.correlation(last, b.iloc[-20:]))
    assert m.rolling_max_drawdown(r, 20).iloc[-1] == approx(m.max_drawdown(last))
    assert m.rolling_sharpe(r, 20).iloc[:19].isna().all()


# ----------------------------------------- significance of the Sharpe ratio ---

def normal_returns(n, mean=0.001, sd=0.01, seed=3):
    return series(np.random.default_rng(seed).normal(mean, sd, n))


def test_sharpe_standard_error_for_normal_returns():
    # With no skew and normal tails the formula reduces to sqrt((1 + SR^2 / 2) / (n - 1)).
    r = normal_returns(5_000)
    sr = r.mean() / r.std(ddof=1)
    expected = math.sqrt((1 + sr ** 2 / 2) / (len(r) - 1))

    assert m.sharpe_standard_error(r, periods_per_year=1) == approx(expected, rel=0.02)


def test_sharpe_standard_error_shrinks_with_more_data():
    assert m.sharpe_standard_error(normal_returns(4_000)) < m.sharpe_standard_error(normal_returns(250))


def test_sharpe_confidence_interval_brackets_the_estimate():
    r = normal_returns(1_000)
    low, high = m.sharpe_confidence_interval(r)
    centre = m.sharpe_ratio(r)

    assert low < centre < high
    assert (high - centre) == approx(1.959964 * m.sharpe_standard_error(r))
    narrow = m.sharpe_confidence_interval(r, confidence=0.5)
    assert narrow[1] - narrow[0] < high - low


def test_probabilistic_sharpe_ratio():
    strong = normal_returns(2_000, mean=0.002)
    assert m.probabilistic_sharpe_ratio(strong) > 0.99

    # Against its own Sharpe ratio as the threshold, the probability is one half.
    own = m.sharpe_ratio(strong)
    assert m.probabilistic_sharpe_ratio(strong, threshold=own) == approx(0.5)

    losing = normal_returns(2_000, mean=-0.002)
    assert m.probabilistic_sharpe_ratio(losing) < 0.01


def test_same_sharpe_is_less_certain_on_a_short_record():
    long_record = normal_returns(2_000)
    short_record = long_record.iloc[:100]
    assert m.sharpe_standard_error(short_record) > m.sharpe_standard_error(long_record)


def test_minimum_track_record_length_is_consistent_with_the_probability():
    r = normal_returns(3_000, mean=0.0008)
    needed = m.minimum_track_record_length(r, confidence=0.95)
    assert needed > 1

    # Rebuild the probability by hand at exactly that many observations.
    sr, spread, _ = m._sharpe_inputs(r)
    from scipy import stats
    assert stats.norm.cdf(sr * math.sqrt(needed - 1) / math.sqrt(spread)) == approx(0.95)


def test_minimum_track_record_length_is_undefined_below_the_threshold():
    assert math.isnan(m.minimum_track_record_length(normal_returns(500, mean=-0.001)))


# -------------------------------------------------------- summary and edges ---

def test_summary_contains_the_headline_metrics():
    result = m.summary(MIXED, periods_per_year=4)

    assert result["total_return"] == approx(0.1286)
    assert result["max_drawdown"] == approx(-0.1)
    assert result["periods"] == 4
    assert "alpha" not in result
    for key in ("sharpe_ratio", "sortino_ratio", "value_at_risk_95", "ulcer_index",
                "sharpe_ci_low", "sharpe_ci_high", "probabilistic_sharpe_ratio"):
        assert key in result


def test_summary_adds_benchmark_metrics_when_given_one():
    result = m.summary(2 * BENCH, benchmark=BENCH)

    assert result["beta"] == approx(2)
    for key in ("alpha", "correlation", "tracking_error", "information_ratio",
                "up_capture", "down_capture"):
        assert key in result


@pytest.mark.parametrize("returns", [EMPTY, series([0.01]), series([0.0, 0.0, 0.0])])
def test_degenerate_input_gives_nan_not_errors(returns):
    result = m.summary(returns, benchmark=returns)

    assert result["periods"] == len(returns)
    assert not any(isinstance(v, float) and math.isinf(v) for v in result)


def test_missing_values_are_ignored():
    with_gaps = series([0.10, np.nan, -0.10, 0.20, np.nan, -0.05])
    assert m.total_return(with_gaps) == approx(0.1286)
    assert m.sharpe_ratio(with_gaps) == approx(m.sharpe_ratio(MIXED))


def test_metrics_accept_plain_lists():
    assert m.total_return([0.10, -0.10, 0.20, -0.05]) == approx(0.1286)
    assert m.max_drawdown([0.10, -0.10, 0.20, -0.05]) == approx(-0.1)


# =========================================================================
# Metrics from the engine's recorded tables
# =========================================================================

def round_trips(rows):
    """rows: (side, net_pnl, entry_price, exit_price, bars, mae, mfe) per trade, one day apart."""
    start = pd.Timestamp("2024-01-01")
    return pd.DataFrame([{
        "symbol": "BTC", "side": side,
        "entry_time": start + pd.Timedelta(days=i), "exit_time": start + pd.Timedelta(days=i, hours=bars),
        "bars_held": bars, "duration": pd.Timedelta(hours=bars), "qty": 1.0,
        "entry_price": entry, "exit_price": exit_, "pnl": net, "commission": 0.0, "funding": 0.0,
        "net_pnl": net, "return_pct": net / entry, "mae": mae, "mfe": mfe, "fills": 2,
        "entry_label": None, "exit_label": None, "is_liquidation": False,
    } for i, (side, net, entry, exit_, bars, mae, mfe) in enumerate(rows)])


# Three wins (+30, +10, +20) and two losses (-10, -20).
TRIPS = round_trips([
    ("LONG", 30, 100, 130, 10, -0.02, 0.40),
    ("LONG", -10, 100, 90, 2, -0.12, 0.01),
    ("SHORT", 10, 100, 90, 6, -0.05, 0.15),
    ("LONG", -20, 100, 80, 4, -0.25, 0.02),
    ("SHORT", 20, 100, 80, 8, -0.01, 0.22),
])
NO_TRIPS = TRIPS.iloc[0:0]


# ------------------------------------------------------------------- trades ---

def test_trade_counts_and_win_rates():
    s = m.trade_summary(TRIPS)

    assert s["total_trades"] == 5 and s["long_trades"] == 3 and s["short_trades"] == 2
    assert s["win_rate"] == approx(0.6)
    assert s["win_rate_long"] == approx(1 / 3)
    assert s["win_rate_short"] == approx(1.0)


def test_win_loss_sizes_and_ratios():
    s = m.trade_summary(TRIPS)

    assert s["average_win"] == approx(20) and s["average_loss"] == approx(-15)
    assert s["largest_win"] == 30 and s["largest_loss"] == -20
    assert s["payoff_ratio"] == approx(20 / 15)
    assert s["profit_factor"] == approx(60 / 30)
    assert s["gross_profit"] == 60 and s["gross_loss"] == -30 and s["net_profit"] == 30


def test_expectancy():
    assert m.trade_summary(TRIPS)["expectancy"] == approx(6)


def test_holding_times_and_streaks():
    s = m.trade_summary(TRIPS)

    assert s["average_bars_held"] == approx(6)
    assert s["average_bars_held_winners"] == approx(8)
    assert s["average_bars_held_losers"] == approx(3)
    assert s["longest_winning_streak"] == 1 and s["longest_losing_streak"] == 1

    streaky = round_trips([("LONG", pnl, 100, 100, 1, 0, 0) for pnl in (1, 1, 1, -1, -1, 1)])
    assert m.trade_summary(streaky)["longest_winning_streak"] == 3
    assert m.trade_summary(streaky)["longest_losing_streak"] == 2


def test_profit_concentration():
    s = m.trade_summary(TRIPS)

    assert s["profit_from_best_trade"] == approx(30 / 30)
    assert s["net_profit_without_best_5"] == approx(0)

    many = round_trips([("LONG", pnl, 100, 100, 1, 0, 0) for pnl in (100, 5, 5, 5, 5, 5, 5, -10)])
    spread = m.trade_summary(many)
    assert spread["profit_from_best_trade"] == approx(100 / 120)
    assert spread["net_profit_without_best_5"] == approx(120 - 120)


def test_concentration_is_undefined_for_a_losing_strategy():
    losing = round_trips([("LONG", -5, 100, 95, 1, -0.05, 0), ("LONG", 2, 100, 102, 1, 0, 0.02)])
    assert math.isnan(m.trade_summary(losing)["profit_from_best_trade"])


def test_trade_summary_without_trades_or_losses():
    assert m.trade_summary(NO_TRIPS)["total_trades"] == 0
    assert m.trade_summary(None)["total_trades"] == 0

    winners = m.trade_summary(round_trips([("LONG", 5, 100, 105, 1, 0, 0.05)] * 2))
    assert math.isnan(winners["profit_factor"]) and math.isnan(winners["payoff_ratio"])
    assert math.isnan(winners["average_loss"])


# ------------------------------------------------------------ trade quality ---

def test_quality_summary():
    s = m.quality_summary(TRIPS)

    assert s["average_adverse_excursion"] == approx(-0.09)
    assert s["average_favourable_excursion"] == approx(0.16)
    assert s["liquidations"] == 0
    assert m.quality_summary(NO_TRIPS)["liquidations"] == 0


# -------------------------------------------------------------------- costs ---

def fills(rows):
    return pd.DataFrame([{"qty": q, "executed_price": p, "pnl": pnl, "commission": c, "slippage": s}
                         for q, p, pnl, c, s in rows])


def test_cost_summary_adds_up():
    # Buy 2 at 101 (1 of slippage each), sell 2 at 119 (1 each): realized 36 after slippage.
    table = fills([(2, 101, 0, 2.0, 2.0), (-2, 119, 36, 2.4, 2.0)])
    funding = pd.DataFrame({"amount": [-1.5, 0.5]})
    s = m.cost_summary(table, funding, equity=pd.Series([1_000, 1_000]), years=0.5)

    assert s["total_commission"] == approx(4.4)
    assert s["total_slippage"] == approx(4.0)
    assert s["funding_paid"] == approx(1.5) and s["funding_received"] == approx(0.5)
    assert s["total_costs"] == approx(4.4 + 4.0 + 1.0)
    assert s["profit_before_costs"] == approx(40)         # 2 * (120 - 100) at clean prices
    assert s["cost_share"] == approx(9.4 / 40)
    assert s["traded_value"] == approx(2 * 101 + 2 * 119)
    assert s["turnover"] == approx(440 / (1_000 * 0.5))


def test_cost_summary_edge_cases():
    assert m.cost_summary(pd.DataFrame())["total_commission"] == 0
    losing = m.cost_summary(fills([(1, 100, 0, 1, 0), (-1, 90, -10, 1, 0)]))
    assert math.isnan(losing["cost_share"])
    assert math.isnan(losing["turnover"])


# ----------------------------------------------------------------- exposure ---

def test_exposure_summary():
    account = pd.DataFrame({
        "cash": [1_000] * 4, "equity": [1_000, 1_000, 1_100, 1_000],
        "margin": [0, 500, 500, 0], "gross_exposure": [0, 2_000, 2_200, 0],
        "net_exposure": [0, 2_000, 2_200, 0], "leverage": [0, 2.0, 2.0, 0],
    })
    position = pd.DataFrame({
        "qty": [0, 20, 20, 0], "close": [100, 100, 110, 100],
        "liquidation_price": [np.nan, 80, 80, np.nan],
    })
    s = m.exposure_summary(account, {"BTC": position})

    assert s["time_in_market"] == approx(0.5)
    assert s["average_leverage"] == approx(1.0)
    assert s["average_leverage_in_market"] == approx(2.0)
    assert s["maximum_leverage"] == approx(2.0)
    assert s["average_margin_usage"] == approx((0.5 + 500 / 1_100) / 4)
    assert s["peak_margin_usage"] == approx(0.5)
    assert s["closest_distance_to_liquidation"] == approx(0.20)


def test_distance_to_liquidation_for_a_short():
    account = pd.DataFrame({"cash": [1_000], "equity": [1_000], "margin": [100],
                            "gross_exposure": [1_000], "net_exposure": [-1_000], "leverage": [1.0]})
    position = pd.DataFrame({"qty": [-10], "close": [100], "liquidation_price": [125]})
    s = m.exposure_summary(account, {"BTC": position})

    assert s["closest_distance_to_liquidation"] == approx(0.25)


def test_exposure_summary_when_never_in_the_market():
    account = pd.DataFrame({"cash": [1_000] * 2, "equity": [1_000] * 2, "margin": [0, 0],
                            "gross_exposure": [0, 0], "net_exposure": [0, 0], "leverage": [0, 0]})
    s = m.exposure_summary(account, {})

    assert s["time_in_market"] == 0
    assert math.isnan(s["closest_distance_to_liquidation"])
    assert math.isnan(s["average_leverage_in_market"])


# ------------------------------------------------------------------- orders ---

def test_order_summary():
    orders = pd.DataFrame({
        "status": ["FILLED", "FILLED", "CANCELLED", "REJECTED", "OPEN"],
        "qty": [1, -1, 1, 1, 1], "executed_price": [100, 110, 0, 0, 0],
        "slippage": [1.0, 1.1, 0, 0, 0],
    })
    s = m.order_summary(orders)

    assert s["orders_placed"] == 5 and s["orders_filled"] == 2
    assert s["orders_cancelled"] == 1 and s["orders_rejected"] == 1 and s["orders_open"] == 1
    assert s["fill_rate"] == approx(2 / 4)                 # open orders are not counted yet
    assert s["average_slippage"] == approx(2.1 / 210)
    assert m.order_summary(pd.DataFrame())["orders_placed"] == 0


# ------------------------------------------------------------------- tables ---

def test_by_symbol():
    trips = pd.concat([TRIPS, TRIPS.assign(symbol="ETH", net_pnl=[5, 5, -4, 0, 0])])
    held = {"BTC": pd.DataFrame({"qty": [1, 0, 0, 0]}), "ETH": pd.DataFrame({"qty": [1, 1, 0, 0]})}
    table = m.by_symbol(trips, held)

    assert list(table.index) == ["BTC", "ETH"]            # most profitable first
    assert table.loc["BTC", "trades"] == 5 and table.loc["BTC", "net_pnl"] == 30
    assert table.loc["ETH", "win_rate"] == approx(0.4)
    assert table.loc["ETH", "profit_factor"] == approx(10 / 4)
    assert table["share_of_profit"].sum() == approx(1.0)
    assert table.loc["ETH", "time_held"] == approx(0.5)
    assert m.by_symbol(NO_TRIPS).empty


def test_by_weekday_and_hour():
    weekday = m.by_weekday(TRIPS)   # entries on 2024-01-01 (a Monday) to 01-05
    hour = m.by_hour(TRIPS)

    assert list(weekday.index) == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    assert weekday.loc["Monday", "net_pnl"] == 30
    assert list(hour.index) == [0] and hour.loc[0, "trades"] == 5


# --------------------------------------------------- from a real backtest ---

class Swing(Strategy):
    def init(self):
        pass

    def next(self, i):
        qty = float(self.get_position(SYMBOL).qty)
        if i % 10 == 0 and qty == 0:
            self.market_order(SYMBOL, 2 if (i // 10) % 2 == 0 else -2)
        elif i % 10 == 6 and qty != 0:
            self.market_order(SYMBOL, -qty)


@pytest.fixture(scope="module")
def backtest():
    rng = np.random.default_rng(7)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 24 * 40))
    frame = make_frame([(c, c * 1.005, c * 0.995, c) for c in closes])
    frame["funding_rate"] = np.where(np.arange(len(frame)) % 8 == 0, 0.0001, 0.0)
    strategy = Swing(data={SYMBOL: frame}, initial_capital=10_000, commission_rate=0.0005,
                     slippage=0.0005)
    returns = strategy.run()
    return strategy, returns


def test_costs_reconcile_with_the_account(backtest):
    strategy, _ = backtest
    costs = m.cost_summary(strategy.fetch_trades(), strategy.fetch_funding())
    change = strategy.get_balance() - 10_000

    assert strategy.get_position(SYMBOL).qty == 0
    assert costs["profit_before_costs"] - costs["total_costs"] == approx(change)
    assert costs["total_slippage"] > 0 and costs["funding_paid"] > 0
    assert m.trade_summary(strategy.fetch_round_trips())["net_profit"] == approx(change)


def test_report_gains_the_recorded_sections(backtest, capsys):
    strategy, returns = backtest
    strategy.plot(returns)
    out = capsys.readouterr().out

    for section in ("Trades", "Trade Quality", "Costs", "Exposure and Margin", "Orders", "By Symbol"):
        assert f"\n{section}\n" in out
    assert "Winning Trades" not in out            # the fill-based fallback is replaced
    assert out.index("\nTrades\n") < out.index("\nBy Symbol\n") < out.index("\nWorst Drawdowns\n")


def test_report_values_match_the_metric_functions(backtest):
    strategy, returns = backtest
    report = Report(
        returns, strategy.broker.trades, 10_000,
        round_trips=strategy.fetch_round_trips(), orders=strategy.fetch_orders(),
        funding=strategy.fetch_funding(), account_history=strategy.fetch_account_history(),
        position_history={SYMBOL: strategy.fetch_position_history(SYMBOL)},
    )
    trades = m.trade_summary(strategy.fetch_round_trips())

    assert report.metrics["Total Trades"] == trades["total_trades"] == 96
    assert report.metrics["Win Rate"] == approx(trades["win_rate"])
    assert report.metrics["Orders Filled"] == 192
    assert report.metrics["Time in Market"] == approx(0.6, abs=0.01)
    assert report.summary_df.loc["Win Rate", "Value"].endswith("%")
    assert report.summary_df.loc["Total Commission", "Section"] == "Costs"
    assert list(report.symbol_table.index) == [SYMBOL]


def test_report_sections_follow_the_data_given(backtest):
    strategy, returns = backtest
    plain = Report(returns, strategy.broker.trades, 10_000)
    partial = Report(returns, strategy.broker.trades, 10_000, orders=strategy.fetch_orders())

    assert "Winning Trades" in plain.metrics.index and "Win Rate" not in plain.metrics.index
    assert set(plain.summary_df["Section"]) == {
        "Overview", "Returns", "Risk", "Drawdown", "Sharpe Ratio Reliability", "Trades"}
    assert "Orders" in set(partial.summary_df["Section"])
    assert "Costs" not in set(partial.summary_df["Section"])
