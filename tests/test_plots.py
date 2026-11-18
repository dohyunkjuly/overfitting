"""Every chart draws, carries the right data, and survives awkward input."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from conftest import make_frame
from overfitting import Strategy
from overfitting.analysis import metrics, plots

approx = pytest.approx


def daily(days=400, seed=0, mean=0.001):
    index = pd.date_range("2023-01-01", periods=days, freq="D")
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.02, days), index=index)


RETURNS = daily()
BENCH = daily(seed=1)

RETURN_CHARTS = [
    plots.cumulative_returns, plots.cumulative_returns_log, plots.monthly_heatmap,
    plots.yearly_returns, plots.monthly_distribution, plots.return_quantiles,
    plots.rolling_sharpe, plots.rolling_volatility, plots.drawdown,
]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()] if ax.get_legend() else []


class Swing(Strategy):
    """Alternates long and short on two symbols, so every recorded table has data."""

    def init(self):
        pass

    def next(self, i):
        for k, symbol in enumerate(self.data.symbols):
            qty = float(self.get_position(symbol).qty)
            if i % 12 == k and qty == 0:
                self.market_order(symbol, 2 if (i // 12) % 2 == 0 else -2)
            elif i % 12 == 7 + k and qty != 0:
                self.market_order(symbol, -qty)


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(3)
    frames = {}
    for symbol in ("BTC", "ETH"):
        closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 24 * 60))
        frame = make_frame([(c, c * 1.004, c * 0.996, c) for c in closes])
        frame["funding_rate"] = np.where(np.arange(len(frame)) % 8 == 0, 0.0001, 0.0)
        frames[symbol] = frame
    strategy = Swing(data=frames, initial_capital=10_000, commission_rate=0.0005, slippage=0.0005)
    strategy.set_leverage("BTC", 5)
    returns = strategy.run()
    return {
        "returns": metrics.aggregate_returns(returns, "D"),
        "round_trips": strategy.fetch_round_trips(),
        "fills": strategy.fetch_trades(),
        "funding": strategy.fetch_funding(),
        "positions": {s: strategy.fetch_position_history(s) for s in ("BTC", "ETH")},
    }


def recorded(run):
    return {key: run[key] for key in ("round_trips", "fills", "funding", "positions")}


# ----------------------------------------------------------- every chart ---

@pytest.mark.parametrize("chart", RETURN_CHARTS, ids=lambda f: f.__name__)
def test_return_chart_draws(chart, tmp_path):
    fig = chart(RETURNS)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title()
    fig.savefig(tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 5_000


@pytest.mark.parametrize("chart", RETURN_CHARTS + [plots.rolling_beta], ids=lambda f: f.__name__)
@pytest.mark.parametrize("returns", [
    daily(days=3),                                                  # almost no history
    pd.Series(0.0, index=pd.date_range("2024-01-01", periods=40)),  # never traded
    daily(days=40, mean=0.05).abs(),                                # never lost
], ids=["three_days", "flat", "only_gains"])
def test_return_charts_survive_awkward_input(chart, returns, tmp_path):
    fig = chart(returns, returns * 0.5) if chart is plots.rolling_beta else chart(returns)
    fig.savefig(tmp_path / "chart.png")


def test_recorded_charts_draw(run, tmp_path):
    for fig in (plots.profit_per_trade(run["round_trips"]),
                plots.cost_breakdown(run["fills"], run["funding"]),
                plots.liquidation_distance(run["positions"]),
                plots.rolling_beta(RETURNS, BENCH)):
        fig.savefig(tmp_path / "chart.png")
        assert fig.axes[0].has_data()


def test_charts_need_dated_returns():
    with pytest.raises(TypeError):
        plots.cumulative_returns(pd.Series([0.01, 0.02]))


def test_missing_data_is_explained_on_the_chart(run):
    ax = plots.rolling_sharpe(daily(days=10)).axes[0]
    assert not ax.get_lines()
    assert "Not enough history" in ax.texts[0].get_text()

    empty = run["round_trips"].iloc[0:0]
    assert "No completed trades" in plots.profit_per_trade(empty).axes[0].texts[0].get_text()
    assert "No completed trades" in plots.profit_per_trade(None).axes[0].texts[0].get_text()
    assert "No fills" in plots.cost_breakdown(pd.DataFrame()).axes[0].texts[0].get_text()


def test_charts_draw_on_given_axes():
    fig, (left, right) = plt.subplots(1, 2)

    assert plots.drawdown(RETURNS, ax=left) is fig
    assert plots.cumulative_returns(RETURNS, ax=right) is fig
    assert left.get_title() == "Daily Drawdown"
    assert len(fig.axes) == 2


# -------------------------------------------------------- return charts ---

def test_cumulative_returns_plots_growth_and_benchmark():
    ax = plots.cumulative_returns(RETURNS, BENCH, label="SMA cross").axes[0]
    strategy_line, benchmark_line = ax.get_lines()[:2]

    assert labels(ax) == ["SMA cross", "Benchmark"]
    assert strategy_line.get_ydata()[-1] == approx(1 + metrics.total_return(RETURNS))
    assert to_hex(strategy_line.get_color()) == "#656ef2"
    assert benchmark_line.get_color() == "grey"
    assert ax.get_title() == "Cumulative Returns"
    log = plots.cumulative_returns_log(RETURNS).axes[0]
    assert log.get_yscale() == "log" and log.get_title() == "Cumulative Returns (log scale)"


def test_monthly_heatmap_is_in_percent_and_centred_on_zero():
    ax = plots.monthly_heatmap(RETURNS).axes[0]
    low, high = ax.images[0].get_clim()

    assert low == -high                                 # so white means zero
    assert [t.get_text() for t in ax.get_yticklabels()] == ["2023", "2024"]
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["1", "2", "3"]
    first = metrics.aggregate_returns(RETURNS, "ME").iloc[0]
    assert ax.texts[0].get_text() == f"{first * 100:.1f}"
    assert ax.get_title() == "Monthly returns (%)"


def test_yearly_returns_per_calendar_year():
    ax = plots.yearly_returns(RETURNS, BENCH).axes[0]

    assert [t.get_text() for t in ax.get_xticklabels()] == ["2023", "2024"]
    assert ax.patches[0].get_height() == approx(metrics.total_return(RETURNS["2023"]))
    assert labels(ax) == ["Simulation", "Benchmark"]
    assert len(ax.patches) == 4


def test_monthly_distribution_counts_months_and_marks_the_average():
    monthly = metrics.aggregate_returns(RETURNS, "ME")
    ax = plots.monthly_distribution(RETURNS).axes[0]

    assert sum(p.get_height() for p in ax.patches) == len(monthly)
    assert labels(ax) == [f"Average: {monthly.mean():.3f}"]
    assert ax.get_title() == "Distribution of Monthly Returns"


def test_return_quantiles_has_a_box_per_period():
    ax = plots.return_quantiles(RETURNS).axes[0]
    boxes = [p for p in ax.patches]

    assert [t.get_text() for t in ax.get_xticklabels()] == ["Daily", "Weekly", "Monthly"]
    assert len(boxes) == 3
    assert ax.get_title() == "Return Quantiles"
    # The line inside the first box is the median daily return.
    medians = [line for line in ax.get_lines() if line.get_color() == "black" and len(line.get_ydata()) == 2
               and line.get_ydata()[0] == line.get_ydata()[1]]
    assert any(line.get_ydata()[0] == approx(RETURNS.median()) for line in medians)


def test_rolling_charts_show_averages_and_the_benchmark():
    ax = plots.rolling_sharpe(RETURNS, BENCH).axes[0]
    rolling = metrics.rolling_sharpe(RETURNS, 180).dropna()

    assert ax.get_title() == "Rolling Sharpe (180 days)"
    assert labels(ax) == ["Simulation", f"Sim Avg: {rolling.mean():.3f}", "Benchmark",
                          f"Bench Avg: {metrics.rolling_sharpe(BENCH, 180).dropna().mean():.3f}"]
    assert ax.get_lines()[0].get_ydata()[-1] == approx(rolling.iloc[-1])
    assert to_hex(ax.get_lines()[2].get_color()) == "#ff914d"

    last = plots.rolling_volatility(RETURNS).axes[0].get_lines()[0].get_ydata()[-1]
    assert last == approx(metrics.volatility(RETURNS.iloc[-180:]))


def test_rolling_beta_of_a_levered_benchmark():
    ax = plots.rolling_beta(2 * BENCH, BENCH).axes[0]

    assert ax.get_title() == "Rolling Beta (180 days)"
    assert ax.get_lines()[0].get_ydata()[-1] == approx(2.0)
    assert labels(ax) == ["Simulation", "Sim Avg: 2.000"]


def test_rolling_window_shrinks_when_history_is_short():
    assert plots.rolling_sharpe(daily(days=120)).axes[0].get_title() == "Rolling Sharpe (90 days)"
    assert plots.rolling_sharpe(daily(days=50)).axes[0].get_title() == "Rolling Sharpe (30 days)"
    assert plots.rolling_sharpe(RETURNS, window=60).axes[0].get_title() == "Rolling Sharpe (60 days)"


def test_drawdown_matches_the_drawdown_series():
    ax = plots.drawdown(RETURNS, label="SMA cross").axes[0]

    assert min(ax.get_lines()[0].get_ydata()) == approx(metrics.max_drawdown(RETURNS))
    assert to_hex(ax.get_lines()[0].get_color()) == "#ff6666"
    assert labels(ax) == ["SMA cross"]


# --------------------------------------------------------- recorded data ---

def test_profit_per_trade_has_a_bar_per_trade_coloured_by_result(run):
    trips = run["round_trips"].sort_values("exit_time")
    bars = plots.profit_per_trade(trips).axes[0].patches

    assert [b.get_height() for b in bars] == approx(list(trips["net_pnl"]))
    for bar, pnl in zip(bars, trips["net_pnl"]):
        assert to_hex(bar.get_facecolor()) == ("#656ef2" if pnl > 0 else "#ff6666")


def test_cost_breakdown_steps_from_gross_to_net(run):
    costs = metrics.cost_summary(run["fills"], run["funding"])
    ax = plots.cost_breakdown(run["fills"], run["funding"]).axes[0]
    names = [t.get_text() for t in ax.get_xticklabels()]

    assert names == ["Profit before costs", "Commission", "Slippage", "Funding", "Net profit"]
    assert ax.patches[0].get_height() == approx(costs["profit_before_costs"])
    assert ax.patches[1].get_height() == approx(costs["total_commission"])
    assert ax.patches[-1].get_height() == approx(costs["profit_before_costs"] - costs["total_costs"])


def test_cost_breakdown_leaves_out_costs_that_did_not_occur(run):
    free = run["fills"].assign(slippage=0.0)
    names = [t.get_text() for t in plots.cost_breakdown(free).axes[0].get_xticklabels()]
    assert names == ["Profit before costs", "Commission", "Net profit"]


def test_liquidation_distance_is_positive_while_safe(run):
    ax = plots.liquidation_distance(run["positions"]).axes[0]

    assert labels(ax) == ["BTC", "ETH", "Liquidation"]
    btc = ax.get_lines()[0].get_ydata()
    assert 0 < np.nanmin(btc) < 0.2 < np.nanmax(btc) < 0.4   # 5x leverage: about 20% away at entry
    flat = {"BTC": run["positions"]["BTC"].assign(qty=0.0)}
    assert "No positions" in plots.liquidation_distance(flat).axes[0].texts[0].get_text()


# ------------------------------------------------------------------ charts ---

def test_there_are_thirteen_charts():
    assert plots.CHART_NAMES == [
        "cumulative_returns", "cumulative_returns_log", "monthly_heatmap", "yearly_returns",
        "monthly_distribution", "return_quantiles", "rolling_sharpe", "rolling_volatility",
        "drawdown", "rolling_beta", "profit_per_trade", "cost_breakdown", "liquidation_distance"]
    assert set(plots.__all__) == set(plots.CHART_NAMES) | {"charts", "CHART_NAMES"}


def test_charts_gives_all_thirteen_in_order_with_full_data(run):
    charts = plots.charts(run["returns"], BENCH, **recorded(run))

    assert list(charts) == plots.CHART_NAMES
    assert plt.get_fignums() == []                      # nothing is drawn until asked
    assert isinstance(charts["drawdown"](), Figure)
    assert len(plt.get_fignums()) == 1


def test_charts_skip_what_the_data_cannot_support(run):
    returns_only = plots.charts(RETURNS)
    no_benchmark = plots.charts(run["returns"], **recorded(run))
    pnl_only = plots.charts(RETURNS, fills=pd.DataFrame({"pnl": [1.0]}))

    assert list(returns_only) == plots.CHART_NAMES[:9]
    assert list(no_benchmark) == [name for name in plots.CHART_NAMES if name != "rolling_beta"]
    assert "cost_breakdown" not in pnl_only            # needs a full trade log


def test_every_chart_draws_from_a_real_backtest(run, tmp_path):
    for name, draw in plots.charts(run["returns"], BENCH, **recorded(run)).items():
        fig = draw()
        fig.savefig(tmp_path / f"{name}.png")
        plt.close(fig)
    assert len(list(tmp_path.iterdir())) == 13


def test_charts_pass_the_strategy_name_on():
    fig = plots.charts(RETURNS, BENCH, label="Breakout")["cumulative_returns"]()
    assert labels(fig.axes[0]) == ["Breakout", "Benchmark"]
