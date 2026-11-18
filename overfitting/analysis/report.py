import os
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import overfitting.analysis.metrics as metrics
import overfitting.analysis.plots as plots
from overfitting.order import Order

IMAGE_DPI = 120

# Labels printed as percentages, as whole numbers, and as money.
PERCENT = {
    "Total Return", "CAGR", "Annualized Mean Return", "Best Day", "Worst Day", "Best Month",
    "Worst Month", "Positive Days", "Positive Months", "Average Up Day", "Average Down Day",
    "Volatility (annualized)", "Downside Deviation (annualized)", "Daily Value at Risk (95%)",
    "Daily Value at Risk (99%)", "Daily Expected Shortfall (95%)", "Daily Expected Shortfall (99%)",
    "Max Drawdown", "Current Drawdown", "Average Drawdown", "Time Underwater", "Ulcer Index",
    "Pain Index", "Probability Sharpe Above 0", "Alpha (annualized)",
    "Tracking Error (annualized)", "Excess Return (annualized)", "Days Beating Benchmark",
}
WHOLE = {
    "Longest Winning Streak (days)", "Longest Losing Streak (days)", "Longest Drawdown (days)",
    "Drawdowns Deeper Than 5%", "Drawdowns Deeper Than 10%", "Drawdowns Deeper Than 20%",
    "Days Needed To Confirm Sharpe", "Total Trades", "Winning Trades", "Losing Trades",
}
MONEY = {"Initial Balance", "Final Balance", "Gross Profit", "Gross Loss", "Net Profit"}

SECTIONS = [
    ("Overview", [
        "Start Date", "End Date", "Number of Years", "Initial Balance", "Final Balance",
        "Total Return", "CAGR",
    ]),
    ("Returns", [
        "Annualized Mean Return", "Best Day", "Worst Day", "Best Month", "Worst Month",
        "Positive Days", "Positive Months", "Average Up Day", "Average Down Day",
        "Longest Winning Streak (days)", "Longest Losing Streak (days)",
    ]),
    ("Risk", [
        "Volatility (annualized)", "Downside Deviation (annualized)", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Omega Ratio", "Gain to Pain Ratio", "Tail Ratio",
        "Common Sense Ratio", "Daily Value at Risk (95%)", "Daily Value at Risk (99%)",
        "Daily Expected Shortfall (95%)", "Daily Expected Shortfall (99%)", "Skew (daily)",
        "Excess Kurtosis (daily)", "Autocorrelation (1 day)", "Kelly Fraction",
    ]),
    ("Drawdown", [
        "Max Drawdown", "Current Drawdown", "Average Drawdown", "Longest Drawdown (days)",
        "Average Drawdown Duration (days)", "Average Recovery Time (days)", "Time Underwater",
        "Ulcer Index", "Pain Index", "Recovery Factor", "Martin Ratio",
        "Drawdowns Deeper Than 5%", "Drawdowns Deeper Than 10%", "Drawdowns Deeper Than 20%",
    ]),
    ("Sharpe Ratio Reliability", [
        "Sharpe Standard Error", "Sharpe 95% Range (low)", "Sharpe 95% Range (high)",
        "Probability Sharpe Above 0", "Days Needed To Confirm Sharpe",
    ]),
    ("Benchmark", [
        "Alpha (annualized)", "Beta", "Correlation", "R-Squared", "Tracking Error (annualized)",
        "Information Ratio", "Excess Return (annualized)", "Up Capture", "Down Capture",
        "Days Beating Benchmark",
    ]),
    # Used only when the report is built without recorded round trips.
    ("Trades", [
        "Total Trades", "Winning Trades", "Losing Trades", "Win Rate (%)", "Gross Profit",
        "Gross Loss", "Net Profit", "Avg Return (%)", "Avg Profit (%)", "Avg Loss (%)",
    ]),
]

# Metrics from the engine's recorded data: (section, [(key, label, kind)]).
# `key` is the name in overfitting.analysis.metrics; `kind` picks the format.
RECORDED = [
    ("Trades", "trade", [
        ("total_trades", "Total Trades", "whole"),
        ("long_trades", "Long Trades", "whole"),
        ("short_trades", "Short Trades", "whole"),
        ("win_rate", "Win Rate", "percent"),
        ("win_rate_long", "Win Rate (long)", "percent"),
        ("win_rate_short", "Win Rate (short)", "percent"),
        ("average_win", "Average Win", "money"),
        ("average_loss", "Average Loss", "money"),
        ("largest_win", "Largest Win", "money"),
        ("largest_loss", "Largest Loss", "money"),
        ("payoff_ratio", "Payoff Ratio", "number"),
        ("profit_factor", "Profit Factor", "number"),
        ("expectancy", "Expectancy per Trade", "money"),
        ("average_return", "Average Trade Return", "percent"),
        ("average_winning_return", "Average Winning Return", "percent"),
        ("average_losing_return", "Average Losing Return", "percent"),
        ("expectancy_r", "Expectancy (R)", "number"),
        ("system_quality_number", "System Quality Number", "number"),
        ("average_bars_held", "Average Holding Time (bars)", "number"),
        ("median_bars_held", "Median Holding Time (bars)", "number"),
        ("longest_bars_held", "Longest Holding Time (bars)", "whole"),
        ("average_bars_held_winners", "Average Holding Time, Winners (bars)", "number"),
        ("average_bars_held_losers", "Average Holding Time, Losers (bars)", "number"),
        ("longest_winning_streak", "Longest Winning Streak (trades)", "whole"),
        ("longest_losing_streak", "Longest Losing Streak (trades)", "whole"),
        ("trades_per_month", "Trades per Month", "number"),
        ("profit_from_best_trade", "Profit From Best Trade", "percent"),
        ("profit_from_best_5_trades", "Profit From Best 5 Trades", "percent"),
        ("net_profit_without_best_5", "Net Profit Without Best 5 Trades", "money"),
        ("gross_profit", "Gross Profit", "money"),
        ("gross_loss", "Gross Loss", "money"),
        ("net_profit", "Net Profit", "money"),
    ]),
    ("Trade Quality", "quality", [
        ("average_adverse_excursion", "Average Adverse Excursion", "percent"),
        ("average_favourable_excursion", "Average Favourable Excursion", "percent"),
        ("edge_ratio", "Edge Ratio", "number"),
        ("profit_captured", "Profit Captured", "percent"),
        ("average_profit_given_back", "Average Profit Given Back", "percent"),
        ("liquidations", "Liquidations", "whole"),
    ]),
    ("Costs", "cost", [
        ("total_commission", "Total Commission", "money"),
        ("total_slippage", "Total Slippage", "money"),
        ("funding_paid", "Funding Paid", "money"),
        ("funding_received", "Funding Received", "money"),
        ("total_costs", "Total Costs", "money"),
        ("profit_before_costs", "Profit Before Costs", "money"),
        ("cost_share", "Costs as Share of Profit Before Costs", "percent"),
        ("average_cost_per_fill", "Average Cost per Fill", "money"),
        ("break_even_commission", "Break-Even Commission Multiple", "number"),
        ("traded_value", "Traded Value", "money"),
        ("turnover", "Turnover (times equity per year)", "number"),
    ]),
    ("Exposure and Margin", "exposure", [
        ("time_in_market", "Time in Market", "percent"),
        ("average_leverage", "Average Leverage", "number"),
        ("average_leverage_in_market", "Average Leverage in Market", "number"),
        ("maximum_leverage", "Maximum Leverage", "number"),
        ("average_net_exposure", "Average Net Exposure", "percent"),
        ("average_margin_usage", "Average Margin Usage", "percent"),
        ("peak_margin_usage", "Peak Margin Usage", "percent"),
        ("closest_distance_to_liquidation", "Closest Distance to Liquidation", "percent"),
        ("average_distance_to_liquidation", "Average Distance to Liquidation", "percent"),
        ("liquidation_loss", "Liquidation Loss", "money"),
    ]),
    ("Orders", "order", [
        ("orders_placed", "Orders Placed", "whole"),
        ("orders_filled", "Orders Filled", "whole"),
        ("orders_cancelled", "Orders Cancelled", "whole"),
        ("orders_rejected", "Orders Rejected", "whole"),
        ("orders_open", "Orders Still Open", "whole"),
        ("fill_rate", "Fill Rate", "percent"),
        ("average_slippage", "Average Slippage per Fill", "percent"),
    ]),
]

for _section, _, _rows in RECORDED:
    _labels = [label for _, label, _ in _rows]
    _existing = dict(SECTIONS).get(_section)
    if _existing is None:
        SECTIONS.append((_section, _labels))
    else:
        # Recorded metrics first, then the fallback-only labels.
        _existing[:] = _labels + [label for label in _existing if label not in _labels]
    for _, _label, _kind in _rows:
        {"percent": PERCENT, "whole": WHOLE, "money": MONEY}.get(_kind, set()).add(_label)

SYMBOL_COLUMNS = {
    "trades": ("Trades", "{:,.0f}"), "win_rate": ("Win rate", "{:.1%}"),
    "net_pnl": ("Net PnL", "{:,.2f}"), "average_pnl": ("Average PnL", "{:,.2f}"),
    "profit_factor": ("Profit factor", "{:.2f}"), "share_of_profit": ("Share of profit", "{:.1%}"),
    "time_held": ("Time held", "{:.1%}"),
}


def _format_value(label: str, value) -> str:
    """One metric as display text."""
    if isinstance(value, pd.Timestamp):
        return f"{value:%Y-%m-%d %H:%M}"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if label in PERCENT:
        return f"{value * 100:,.2f}%"
    if label in WHOLE:
        return f"{value:,.0f}"
    if label in MONEY:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


class Report:
    def __init__(self,
                 returns_series: pd.Series,
                 trades_list: Sequence[Order],
                 initial_capital: int,
                 benchmark: Optional[pd.DataFrame] = None,
                 save_path: Optional[str] = None,
                 title_prefix: str = "Simulation",
                 round_trips: Optional[pd.DataFrame] = None,
                 orders: Optional[pd.DataFrame] = None,
                 funding: Optional[pd.DataFrame] = None,
                 account_history: Optional[pd.DataFrame] = None,
                 position_history: Optional[dict] = None):
        """
        returns_series: periodic strategy returns indexed by time.
        trades_list: the broker's trade log.
        benchmark: optional price data (a `close` column) to compare against.
        save_path: directory where show() saves each chart as a PNG file.
        title_prefix: the strategy's name on the charts and the report.
        round_trips, orders, funding, account_history, position_history: the
            tables from Strategy.fetch_round_trips(), fetch_orders(),
            fetch_funding(), fetch_account_history() and a dict of
            fetch_position_history() per symbol. Each one given adds its
            section to the report (trades, trade quality, costs, exposure, orders).
        """
        
        if not isinstance(returns_series.index, pd.DatetimeIndex):
            raise ValueError("returns_series must have a DatetimeIndex")

        self.title_prefix = title_prefix
        self.round_trips = round_trips
        self.orders = orders
        self.funding = funding
        self.account_history = account_history
        self.position_history = position_history
        self.symbol_table = None
        self.save_path = save_path.rstrip("/") if save_path else None
        self.initial_capital = float(initial_capital)

        # normalize strategy series
        self.returns = returns_series.sort_index().copy()
        self.daily_returns = (1 + self.returns).resample("D").prod().sub(1).ffill()
        self.cum = (1 + self.daily_returns).cumprod()

        self.start = self.returns.index.min()
        self.end = self.returns.index.max()

        # trades
        self.trades_list = list(trades_list or [])
        self.gross_returns, self.return_percents = self._unpack_trades_list(self.trades_list)

        # optional benchmark
        self.bench_cum, self.bench_daily = self._normalize_benchmark(benchmark)
        self.summary_df, self.drawdown_series, self.drawdown_table = self._compute_summary()

    # ---------- Output ----------

    def charts(self) -> dict:
        """The charts in report order, as {name: draw}. Each draw() returns a new figure."""
        fills = pd.DataFrame(self.trades_list)
        return plots.charts(
            self.daily_returns, self.bench_daily, label=self.title_prefix,
            round_trips=self.round_trips, fills=fills if len(fills) else None,
            funding=self.funding, positions=self.position_history,
        )

    def show(self):
        """
        Print the summary and show the charts one after another.

        In a notebook the charts appear inline; in a script each opens in a
        window. With `save_path`, each chart is also saved there as a PNG file.
        """
        self._print_summary(self.summary_df, self.drawdown_table)
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # A non-interactive backend (a server, a test run) has nothing to show.
        interactive = matplotlib.get_backend().lower() != "agg"
        for name, draw in self.charts().items():
            fig = draw()
            if self.save_path:
                fig.savefig(os.path.join(self.save_path, f"{name}.png"), dpi=IMAGE_DPI,
                            bbox_inches="tight")
            if interactive:
                plt.show()
            plt.close(fig)

        if self.save_path:
            print(f"\nCharts saved to {self.save_path}")

    def _normalize_benchmark(self, benchmark: Optional[pd.DataFrame]):
        if benchmark is None:
            return None, None

        b = benchmark.copy()
        if not isinstance(b, pd.DataFrame):
            return None, None

        if not isinstance(b.index, pd.DatetimeIndex) and "timestamp" in b.columns:
            b.index = pd.to_datetime(b["timestamp"], unit="ms")
        b = b.sort_index()
        b = b.loc[pd.to_datetime(self.start):pd.to_datetime(self.end)]

        if "close" in b.columns:
            bench_cum = (b["close"].astype(float) / b["close"].iloc[0]).ffill()
        else:
            num_cols = b.select_dtypes(include="number").columns
            if len(num_cols) == 0:
                return None, None
            series = b[num_cols[0]].astype(float)
            bench_cum = (series / series.iloc[0]).ffill()

        # Value at the end of each day, so a day's return covers the same
        # hours as the strategy's return for that day.
        bench_cum = bench_cum.resample("D").last().ffill()
        bench_daily = bench_cum.pct_change()
        bench_daily.iloc[0] = bench_cum.iloc[0] - 1.0
        return bench_cum, bench_daily


    def _unpack_trades_list(self, trades_list):
        if not trades_list:
            return np.array([]), np.array([])
        df = pd.DataFrame(trades_list)
        gross_returns = df.get("realized_pnl", pd.Series(dtype=float)).values

        # closed trades only for percentage returns if 'pnl' exists
        if "pnl" in df and "executed_price" in df and "qty" in df:
            closed = df[df["pnl"] != 0].copy()
            notional = (closed["executed_price"].abs() * closed["qty"].abs()).replace(0, np.nan)
            return_percents = (closed["pnl"] / notional).ffill().values
        else:
            return_percents = np.array([])
        return gross_returns, return_percents

    def _compute_summary(self):
        drawdown = self._compute_drawdown()
        stats = self._compute_return_stats()
        if self.round_trips is None:
            stats.update(self._compute_trade_stats())
        stats.update(self._compute_recorded_stats())
        self.metrics = pd.Series(stats, dtype=object)

        summary_df = self._format_summary()
        dd_table = self._worst_drawdowns(self.daily_returns)
        return summary_df, drawdown, dd_table

    def _compute_drawdown(self):
        peak = self.cum.expanding(min_periods=1).max()
        return (self.cum / peak) - 1

    def _compute_return_stats(self):
        """Every return-based metric, keyed by the label it is printed under."""
        daily = self.daily_returns
        stats = metrics.summary(daily, benchmark=self.bench_daily)
        monthly = metrics.aggregate_returns(daily, "ME")
        growth = float(self.cum.iloc[-1])

        out = {
            # overview
            "Start Date": self.start,
            "End Date": self.end,
            "Number of Years": len(daily) / metrics.DAYS_PER_YEAR,
            "Initial Balance": self.initial_capital,
            "Final Balance": self.initial_capital * growth,
            "Total Return": stats["total_return"],
            "CAGR": stats["cagr"],
            # returns
            "Annualized Mean Return": stats["annualized_mean"],
            "Best Day": stats["best_period"],
            "Worst Day": stats["worst_period"],
            "Best Month": metrics.best(monthly),
            "Worst Month": metrics.worst(monthly),
            "Positive Days": stats["positive_share"],
            "Positive Months": metrics.positive_share(monthly),
            "Average Up Day": stats["average_positive"],
            "Average Down Day": stats["average_negative"],
            "Longest Winning Streak (days)": stats["longest_winning_streak"],
            "Longest Losing Streak (days)": stats["longest_losing_streak"],
            # risk
            "Volatility (annualized)": stats["volatility"],
            "Downside Deviation (annualized)": stats["downside_deviation"],
            "Sharpe Ratio": stats["sharpe_ratio"],
            "Sortino Ratio": stats["sortino_ratio"],
            "Calmar Ratio": stats["calmar_ratio"],
            "Omega Ratio": stats["omega_ratio"],
            "Gain to Pain Ratio": stats["gain_to_pain_ratio"],
            "Tail Ratio": stats["tail_ratio"],
            "Common Sense Ratio": stats["common_sense_ratio"],
            "Daily Value at Risk (95%)": stats["value_at_risk_95"],
            "Daily Value at Risk (99%)": stats["value_at_risk_99"],
            "Daily Expected Shortfall (95%)": stats["expected_shortfall_95"],
            "Daily Expected Shortfall (99%)": stats["expected_shortfall_99"],
            "Skew (daily)": stats["skewness"],
            "Excess Kurtosis (daily)": stats["kurtosis"],
            "Autocorrelation (1 day)": stats["autocorrelation_lag_1"],
            "Kelly Fraction": stats["kelly_fraction"],
            # drawdown
            "Max Drawdown": stats["max_drawdown"],
            "Current Drawdown": stats["current_drawdown"],
            "Average Drawdown": stats["average_drawdown"],
            "Longest Drawdown (days)": stats["longest_drawdown"],
            "Average Drawdown Duration (days)": stats["average_drawdown_duration"],
            "Average Recovery Time (days)": stats["average_recovery_time"],
            "Time Underwater": stats["time_underwater"],
            "Ulcer Index": stats["ulcer_index"],
            "Pain Index": stats["pain_index"],
            "Recovery Factor": stats["recovery_factor"],
            "Martin Ratio": stats["martin_ratio"],
            "Drawdowns Deeper Than 5%": stats["drawdowns_over_5pct"],
            "Drawdowns Deeper Than 10%": stats["drawdowns_over_10pct"],
            "Drawdowns Deeper Than 20%": stats["drawdowns_over_20pct"],
            # how much the Sharpe ratio can be trusted
            "Sharpe Standard Error": stats["sharpe_standard_error"],
            "Sharpe 95% Range (low)": stats["sharpe_ci_low"],
            "Sharpe 95% Range (high)": stats["sharpe_ci_high"],
            "Probability Sharpe Above 0": stats["probabilistic_sharpe_ratio"],
            "Days Needed To Confirm Sharpe": stats["minimum_track_record_length"],
        }

        if self.bench_daily is not None:
            out.update({
                "Alpha (annualized)": stats["alpha"],
                "Beta": stats["beta"],
                "Correlation": stats["correlation"],
                "R-Squared": stats["r_squared"],
                "Tracking Error (annualized)": stats["tracking_error"],
                "Information Ratio": stats["information_ratio"],
                "Excess Return (annualized)": stats["excess_return"],
                "Up Capture": stats["up_capture"],
                "Down Capture": stats["down_capture"],
                "Days Beating Benchmark": stats["outperformance_share"],
            })
        return out

    def _compute_trade_stats(self):
        """
        Counts and totals over the trade log.

        gross_returns: realized PnL of every fill, after commission.
        return_percents: PnL over notional, for the fills that closed a position.
        A "trade" here is a closing fill; opening fills only carry commission.
        """
        pnl = np.asarray(self.gross_returns, dtype=float)
        pct = np.asarray(self.return_percents, dtype=float)
        wins, losses = pct[pct > 0], pct[pct < 0]

        def mean_pct(values):
            return float(values.mean() * 100) if len(values) else np.nan

        return {
            "Total Trades": float(len(pct)),
            "Winning Trades": float(len(wins)),
            "Losing Trades": float(len(losses)),
            "Win Rate (%)": float(len(wins) / len(pct) * 100) if len(pct) else np.nan,
            "Gross Profit": float(pnl[pnl > 0].sum()),
            "Gross Loss": float(pnl[pnl < 0].sum()),
            "Net Profit": float(pnl.sum()),
            "Avg Return (%)": mean_pct(pct),
            "Avg Profit (%)": mean_pct(wins),
            "Avg Loss (%)": mean_pct(losses),
        }

    def _compute_recorded_stats(self):
        """Metrics from the engine's recorded tables, for whichever tables were given."""
        fills = pd.DataFrame(self.trades_list)
        results = {}
        if self.round_trips is not None:
            results["trade"] = metrics.trade_summary(self.round_trips)
            results["quality"] = metrics.quality_summary(self.round_trips)
            self.symbol_table = metrics.by_symbol(self.round_trips, self.position_history)
            years = len(self.daily_returns) / metrics.DAYS_PER_YEAR
            equity = None if self.account_history is None else self.account_history["equity"]
            results["cost"] = metrics.cost_summary(fills, self.funding, equity, years)
        if self.account_history is not None:
            results["exposure"] = metrics.exposure_summary(
                self.account_history, self.position_history, self.round_trips)
        if self.orders is not None:
            results["order"] = metrics.order_summary(self.orders)

        out = {}
        for _, group, rows in RECORDED:
            values = results.get(group)
            if values is None:
                continue
            for key, label, _ in rows:
                out[label] = values.get(key, np.nan)
        return out

    def _symbol_display(self):
        """The per-symbol table as display text, or None if there is nothing to show."""
        if self.symbol_table is None or self.symbol_table.empty:
            return None
        table = self.symbol_table
        return pd.DataFrame({
            title: [fmt.format(v) if pd.notna(v) else "n/a" for v in table[column]]
            for column, (title, fmt) in SYMBOL_COLUMNS.items()
        }, index=pd.Index(table.index, name="Symbol"))

    def _worst_drawdowns(self, daily_returns, top: int = 10):
        """The deepest drawdowns as a display table, worst first."""
        table = metrics.drawdown_table(daily_returns, top=top)
        return pd.DataFrame({
            "Net drawdown in %": table["depth"] * -100,
            "Peak date": table["peak"],
            "Valley date": table["valley"],
            "Recovery date": table["recovery"],
            "Duration (days)": table["duration"],
        })

    def _format_summary(self):
        """The metrics as display text, grouped into sections."""
        rows = []
        for section, labels in SECTIONS:
            present = [label for label in labels if label in self.metrics.index]
            for label in present:
                rows.append((section, label, _format_value(label, self.metrics[label])))
        return pd.DataFrame(rows, columns=["Section", "Metric", "Value"]).set_index("Metric")

    def _print_summary(self, summary_df, dd_table):
        print("Performance Summary")
        width = max(len(label) for label in summary_df.index) + 2
        for section, block in summary_df.groupby("Section", sort=False):
            print(f"\n{section}")
            for label, value in block["Value"].items():
                print(f"  {label:<{width}}{value:>22}")
        symbols = self._symbol_display()
        if symbols is not None:
            print("\nBy Symbol")
            print(symbols.to_string())
        print("\nWorst Drawdowns")
        print(dd_table)
