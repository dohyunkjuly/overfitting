"""
Charts for a backtest, drawn with matplotlib.

Each chart is a function that returns the matplotlib Figure it drew on. Charts
of returns take daily simple returns (a pandas Series indexed by date); the
trade, cost and liquidation charts take the tables from the `Strategy.fetch_*`
methods.

    from overfitting.analysis import plots

    plots.drawdown(daily_returns).savefig("drawdown.png")
    plots.profit_per_trade(strategy.fetch_round_trips())

Pass `ax=` to draw on axes you already have. `charts()` gives the charts for a
report, in order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from overfitting.analysis import metrics

# Colours
STRATEGY_COLOR = "#656EF2"
BENCHMARK_GREY = "grey"            # benchmark on the cumulative and yearly charts
BENCHMARK_ORANGE = "#FF914D"       # benchmark on the rolling charts
BENCHMARK_AVERAGE = "#F2C0A1"
LOSS_COLOR = "#FF6666"             # drawdowns, losing trades, costs
HISTOGRAM_COLOR = "#577CBB"
HEATMAP_COLORS = ["#8B0000", "white", "#96B0C1"]   # loss, zero, gain

WIDE = (12, 6)
STANDARD = (10, 6)
ROLLING_WINDOW = 180
FALLBACK_WINDOWS = (90, 30)        # used when there is less than `window` days of history


# ----------------------------------------------------------------- helpers ---

def _axes(ax, figsize):
    """The axes to draw on: the one given, or a new figure of `figsize`."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    return ax.figure, ax


def _message(ax, title: str, text: str) -> Figure:
    """A titled chart that says why there is nothing to draw."""
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="grey")
    return ax.figure


def _finish(ax, title: str, x_label: str, y_label: str, *, legend: bool = True,
            grid: str | bool = "dashed") -> Figure:
    """Title, axis labels, legend and grid in the house style."""
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()
    if grid == "dashed":
        ax.grid(True, linestyle="--", linewidth=0.5)
    else:
        ax.grid(bool(grid))
    return ax.figure


def _daily(returns) -> pd.Series:
    r = pd.Series(returns, dtype=float).dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("Charts need returns indexed by date.")
    return r.sort_index()


def _pair(returns, benchmark) -> tuple[pd.Series, pd.Series]:
    frame = pd.concat([_daily(returns), _daily(benchmark)], axis=1, join="inner",
                      keys=["r", "b"]).dropna()
    return frame["r"], frame["b"]


def _window(n: int, window: int) -> int | None:
    """`window` if there is enough history for it, else the longest fallback that fits."""
    for candidate in (window, *[w for w in FALLBACK_WINDOWS if w < window]):
        if n >= candidate + 5:
            return candidate
    return None


def _rolling(ax, name: str, label: str, strategy: pd.Series, benchmark: pd.Series | None,
             window: int | None) -> Figure:
    """A rolling statistic with its average, and the benchmark's if given."""
    fig, ax = _axes(ax, STANDARD)
    if window is None:
        return _message(ax, name, "Not enough history for a rolling window.")
    strategy = strategy.dropna()
    ax.plot(strategy, label=label, color=STRATEGY_COLOR)
    ax.axhline(strategy.mean(), color="grey", linestyle=":", label=f"Sim Avg: {strategy.mean():.3f}")
    if benchmark is not None:
        benchmark = benchmark.dropna()
        ax.plot(benchmark, label="Benchmark", color=BENCHMARK_ORANGE, alpha=0.85)
        ax.axhline(benchmark.mean(), color=BENCHMARK_AVERAGE, linestyle=":",
                   label=f"Bench Avg: {benchmark.mean():.3f}")
    ax.tick_params(axis="x", rotation=45)
    _finish(ax, f"{name} ({window} days)", "Date", "Value")
    if ax.figure.get_axes() == [ax]:
        fig.tight_layout()
    return fig


def _trips(round_trips) -> pd.DataFrame:
    """Round trips in the order they closed, numbered from 1."""
    if round_trips is None or len(round_trips) == 0:
        return pd.DataFrame()
    table = round_trips.sort_values("exit_time").reset_index(drop=True)
    table.index = table.index + 1
    return table


# ------------------------------------------------------------------ returns ---

def cumulative_returns(returns, benchmark=None, log: bool = False, label: str = "Simulation",
                       ax=None) -> Figure:
    """Growth of 1 unit of capital, with the benchmark if given. `log=True` uses a log scale."""
    r = _daily(returns)
    fig, ax = _axes(ax, WIDE)
    ax.plot(metrics.equity_curve(r), label=label, color=STRATEGY_COLOR)
    if benchmark is not None:
        ax.plot(metrics.equity_curve(_daily(benchmark)), label="Benchmark",
                color=BENCHMARK_GREY, alpha=0.9)
    if log:
        ax.set_yscale("log")
    return _finish(ax, "Cumulative Returns (log scale)" if log else "Cumulative Returns", "Date",
                   "Cumulative Returns (log)" if log else "Cumulative Returns", grid=True)


def cumulative_returns_log(returns, benchmark=None, label: str = "Simulation", ax=None) -> Figure:
    """Growth of 1 unit of capital on a log scale, so equal percentage moves look equal."""
    return cumulative_returns(returns, benchmark, log=True, label=label, ax=ax)


def monthly_heatmap(returns, ax=None) -> Figure:
    """Monthly returns in percent as a year-by-month grid. Blue months gained, red months lost."""
    table = metrics.calendar_returns(_daily(returns)).dropna(axis=1, how="all") * 100
    fig, ax = _axes(ax, (12, 5))
    values = table.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.0
    limit = limit or 1.0
    cmap = LinearSegmentedColormap.from_list("custom", HEATMAP_COLORS, N=100).with_extremes(bad="white")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    fig.colorbar(image, ax=ax)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                # White text on the darkest cells, as dark text is unreadable there.
                ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                        color="white" if value < -0.6 * limit else "#262626")

    ax.set_xticks(range(len(table.columns)), [str(m) for m in table.columns])
    ax.set_yticks(range(len(table)), [str(year) for year in table.index])
    return _finish(ax, "Monthly returns (%)", "month", "year", legend=False, grid=False)


def yearly_returns(returns, benchmark=None, label: str = "Simulation", ax=None) -> Figure:
    """Return for each calendar year, next to the benchmark's if given."""
    fig, ax = _axes(ax, STANDARD)
    series = [(label, metrics.aggregate_returns(_daily(returns), "YE"), STRATEGY_COLOR, 1.0)]
    if benchmark is not None:
        series.append(("Benchmark", metrics.aggregate_returns(_daily(benchmark), "YE"),
                       BENCHMARK_GREY, 0.9))

    years = sorted({year for _, values, _, _ in series for year in values.index.year})
    width = 0.8 / len(series)
    for k, (name, values, color, alpha) in enumerate(series):
        by_year = dict(zip(values.index.year, values))
        offset = (k - (len(series) - 1) / 2) * width
        ax.bar([i + offset for i in range(len(years))], [by_year.get(y, np.nan) for y in years],
               width=width, color=color, alpha=alpha, label=name)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.axhline(0, color="grey", linewidth=0.8)
    return _finish(ax, "Yearly Returns", "Year", "Return")


def monthly_distribution(returns, ax=None) -> Figure:
    """How often each size of monthly return occurred, with the average marked."""
    monthly = metrics.aggregate_returns(_daily(returns), "ME")
    fig, ax = _axes(ax, STANDARD)
    ax.hist(monthly, color=HISTOGRAM_COLOR)
    if len(monthly):
        ax.axvline(monthly.mean(), color="grey", linestyle="--", label=f"Average: {monthly.mean():.3f}")
    return _finish(ax, "Distribution of Monthly Returns", "Return", "Frequency")


def return_quantiles(returns, ax=None) -> Figure:
    """
    Spread of daily, weekly and monthly returns as box plots: the box is the
    middle half of the returns, the line inside it the median, the whiskers the
    usual range, and the dots the outliers.
    """
    r = _daily(returns)
    fig, ax = _axes(ax, STANDARD)
    groups = [metrics.aggregate_returns(r, freq) for freq in ("D", "W", "ME")]
    boxes = ax.boxplot(groups, patch_artist=True, medianprops=dict(color="black"),
                       flierprops=dict(marker=".", markerfacecolor=STRATEGY_COLOR,
                                       markeredgecolor=STRATEGY_COLOR))
    for box in boxes["boxes"]:
        box.set(facecolor=STRATEGY_COLOR, alpha=0.6)
    ax.set_xticks([1, 2, 3], ["Daily", "Weekly", "Monthly"])
    ax.axhline(0, color="grey", linewidth=0.8)
    return _finish(ax, "Return Quantiles", "Period", "Return", legend=False)


# --------------------------------------------------------------------- risk ---

def rolling_sharpe(returns, benchmark=None, window: int = ROLLING_WINDOW,
                   label: str = "Simulation", ax=None) -> Figure:
    """Annualized Sharpe ratio over a moving window, with the benchmark's if given."""
    r = _daily(returns)
    w = _window(len(r), window)
    bench = None if benchmark is None or w is None else metrics.rolling_sharpe(_daily(benchmark), w)
    return _rolling(ax, "Rolling Sharpe", label, metrics.rolling_sharpe(r, w) if w else r, bench, w)


def rolling_volatility(returns, benchmark=None, window: int = ROLLING_WINDOW,
                       label: str = "Simulation", ax=None) -> Figure:
    """Annualized volatility over a moving window, with the benchmark's if given."""
    r = _daily(returns)
    w = _window(len(r), window)
    bench = None if benchmark is None or w is None else metrics.rolling_volatility(_daily(benchmark), w)
    return _rolling(ax, "Rolling Volatility", label, metrics.rolling_volatility(r, w) if w else r,
                    bench, w)


def rolling_beta(returns, benchmark, window: int = ROLLING_WINDOW, label: str = "Simulation",
                 ax=None) -> Figure:
    """Sensitivity to the benchmark over a moving window."""
    r, b = _pair(returns, benchmark)
    w = _window(len(r), window)
    return _rolling(ax, "Rolling Beta", label, metrics.rolling_beta(r, b, w) if w else r, None, w)


def drawdown(returns, label: str = "Simulation", ax=None) -> Figure:
    """Decline from the highest equity reached so far."""
    series = metrics.drawdown_series(_daily(returns))
    fig, ax = _axes(ax, WIDE)
    ax.fill_between(series.index, series.values, color=LOSS_COLOR, alpha=1)
    ax.plot(series, color=LOSS_COLOR)
    ax.legend([label])
    return _finish(ax, "Daily Drawdown", "Date", "Drawdown", legend=False, grid=False)


# ------------------------------------------------------------------- trades ---

def profit_per_trade(round_trips, ax=None) -> Figure:
    """
    Net profit of each completed trade, in the order they closed. `round_trips`
    is Strategy.fetch_round_trips().
    """
    trips = _trips(round_trips)
    fig, ax = _axes(ax, WIDE)
    title = "Profit per Trade"
    if trips.empty:
        return _message(ax, title, "No completed trades.")
    colors = [STRATEGY_COLOR if pnl > 0 else LOSS_COLOR for pnl in trips["net_pnl"]]
    ax.bar(trips.index, trips["net_pnl"], color=colors, width=0.9)
    ax.axhline(0, color="grey", linewidth=0.8)
    return _finish(ax, title, "Trade", "Net PnL", legend=False)


def cost_breakdown(fills: pd.DataFrame, funding: pd.DataFrame | None = None, ax=None) -> Figure:
    """
    Profit before costs stepped down to net profit by commission, slippage and
    funding. `fills` is Strategy.fetch_trades(), `funding` Strategy.fetch_funding().
    Costs that did not occur are left out.
    """
    fig, ax = _axes(ax, STANDARD)
    title = "From Profit Before Costs to Net Profit"
    if fills is None or len(fills) == 0:
        return _message(ax, title, "No fills.")
    costs = metrics.cost_summary(fills, funding)
    net_funding = costs["funding_paid"] - costs["funding_received"]
    steps = [("Commission", -costs["total_commission"]), ("Slippage", -costs["total_slippage"]),
             ("Funding", -net_funding)]
    steps = [(name, change) for name, change in steps if change != 0]
    start = float(costs["profit_before_costs"])

    names, bottoms, heights, colors = ["Profit before costs"], [0.0], [start], [HISTOGRAM_COLOR]
    level = start
    for name, change in steps:
        names.append(name)
        bottoms.append(min(level, level + change))
        heights.append(abs(change))
        colors.append(LOSS_COLOR if change < 0 else STRATEGY_COLOR)
        level += change
    names.append("Net profit")
    bottoms.append(0.0)
    heights.append(level)
    colors.append(STRATEGY_COLOR)

    bars = ax.bar(names, heights, bottom=bottoms, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{start:,.0f}"] + [f"{c:+,.0f}" for _, c in steps] + [f"{level:,.0f}"])
    ax.axhline(0, color="grey", linewidth=0.8)
    return _finish(ax, title, "", "Profit", legend=False)


def liquidation_distance(positions: dict, ax=None) -> Figure:
    """
    How far each symbol's close was from its liquidation price, as a share of
    the close, while a position was open. Lower is closer to liquidation.
    `positions` is {symbol: Strategy.fetch_position_history(symbol)}.
    """
    fig, ax = _axes(ax, WIDE)
    title = "Distance to Liquidation"
    drawn = False
    for symbol, history in positions.items():
        held = history[history["qty"] != 0]
        if held.empty:
            continue
        distance = np.sign(held["qty"]) * (held["close"] - held["liquidation_price"]) / held["close"]
        ax.plot(distance.reindex(history.index), label=symbol)
        drawn = True
    if not drawn:
        return _message(ax, title, "No positions were held.")
    ax.axhline(0, color=LOSS_COLOR, linestyle="--", label="Liquidation")
    return _finish(ax, title, "Date", "Distance / price", grid=True)


# -------------------------------------------------------------- the report ---

@dataclass
class _Chart:
    name: str
    needs: tuple            # names of the data this chart cannot be drawn without
    draw: Callable          # (data) -> Figure


R, B, L = "returns", "benchmark", "label"
_REGISTRY = [
    _Chart("cumulative_returns", (), lambda d: cumulative_returns(d[R], d[B], label=d[L])),
    _Chart("cumulative_returns_log", (), lambda d: cumulative_returns_log(d[R], d[B], label=d[L])),
    _Chart("monthly_heatmap", (), lambda d: monthly_heatmap(d[R])),
    _Chart("yearly_returns", (), lambda d: yearly_returns(d[R], d[B], label=d[L])),
    _Chart("monthly_distribution", (), lambda d: monthly_distribution(d[R])),
    _Chart("return_quantiles", (), lambda d: return_quantiles(d[R])),
    _Chart("rolling_sharpe", (), lambda d: rolling_sharpe(d[R], d[B], label=d[L])),
    _Chart("rolling_volatility", (), lambda d: rolling_volatility(d[R], d[B], label=d[L])),
    _Chart("drawdown", (), lambda d: drawdown(d[R], label=d[L])),
    _Chart("rolling_beta", (B,), lambda d: rolling_beta(d[R], d[B], label=d[L])),
    _Chart("profit_per_trade", ("round_trips",), lambda d: profit_per_trade(d["round_trips"])),
    _Chart("cost_breakdown", ("fills",), lambda d: cost_breakdown(d["fills"], d.get("funding"))),
    _Chart("liquidation_distance", ("positions",), lambda d: liquidation_distance(d["positions"])),
]
CHART_NAMES = [chart.name for chart in _REGISTRY]

__all__ = ["charts", "CHART_NAMES"] + CHART_NAMES


def _has(data: dict, need: str) -> bool:
    value = data.get(need)
    if value is None or len(value) == 0:
        return False
    if need == "fills":         # a full trade log, not just PnL numbers
        return {"commission", "pnl", "qty", "executed_price"} <= set(value.columns)
    return True


def charts(returns, benchmark=None, label: str = "Simulation", *, round_trips=None, fills=None,
           funding=None, positions=None) -> dict:
    """
    The report's charts in order, as {name: draw}. Each `draw()` creates and
    returns one figure, so figures can be made (and closed) one at a time:

        for name, draw in plots.charts(daily, round_trips=trips, ...).items():
            fig = draw()
            fig.savefig(f"{name}.png")
            plt.close(fig)

    A chart is left out if the data it needs was not given: `rolling_beta`
    needs `benchmark` (daily returns), `profit_per_trade` needs `round_trips`,
    `cost_breakdown` needs `fills`, and `liquidation_distance` needs
    `positions`. These are the tables from the Strategy.fetch_* methods.
    """
    data = {
        R: _daily(returns), B: _daily(benchmark) if benchmark is not None else None, L: label,
        "round_trips": round_trips, "fills": fills, "funding": funding, "positions": positions,
    }
    return {
        chart.name: (lambda chart=chart: chart.draw(data))
        for chart in _REGISTRY if all(_has(data, need) for need in chart.needs)
    }
