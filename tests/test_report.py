"""The printed report: summary numbers, trade statistics and the drawdown table."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from overfitting.analysis import metrics
from overfitting.analysis.report import Report

approx = pytest.approx


def hourly_returns(days=120, seed=0):
    index = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    return pd.Series(np.random.default_rng(seed).normal(0.0002, 0.004, len(index)), index=index)


def trade(pnl, commission=1.0, price=100.0, qty=1.0):
    return {"pnl": pnl, "realized_pnl": pnl - commission, "executed_price": price, "qty": qty}


# Two round trips: +10 and -5, each with an opening fill that only pays commission.
TRADES = [trade(0), trade(10, qty=-1.0), trade(0), trade(-5, qty=-1.0)]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def report():
    return Report(hourly_returns(), TRADES, initial_capital=10_000)


def test_report_needs_dated_returns():
    with pytest.raises(ValueError):
        Report(pd.Series([0.01, 0.02]), [], 10_000)


def test_risk_numbers_come_from_the_metrics_module(report):
    daily = report.daily_returns
    assert report.metrics["Sharpe Ratio"] == approx(metrics.sharpe_ratio(daily))
    assert report.metrics["Sortino Ratio"] == approx(metrics.sortino_ratio(daily))
    assert report.metrics["Max Drawdown"] == approx(metrics.max_drawdown(daily))
    assert report.metrics["Daily Value at Risk (95%)"] == approx(metrics.value_at_risk(daily, 0.95))
    assert report.metrics["Best Month"] == approx(metrics.aggregate_returns(daily, "ME").max())


def test_every_metric_is_listed_in_a_section(report):
    from overfitting.analysis.report import SECTIONS

    listed = {label for _, labels in SECTIONS for label in labels}
    assert set(report.metrics.index) <= listed
    assert len(report.metrics) == 49  # 57 with a benchmark
    assert len(report.summary_df) == len(report.metrics)


def test_values_are_formatted_by_kind(report):
    text = report.summary_df["Value"]

    assert text["Max Drawdown"].endswith("%")
    assert text["Initial Balance"] == "10,000.00"
    assert text["Total Trades"] == "2"
    assert text["Start Date"] == "2024-01-01 00:00"
    assert "." in text["Sharpe Ratio"]
    low, high = (float(v) for v in text["Sharpe 95% Range"].split(" to "))
    assert low < report.metrics["Sharpe Ratio"] < high


def test_undefined_metrics_print_as_not_available():
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    flat = Report(pd.Series(0.0, index=index), [], 10_000)

    assert flat.summary_df.loc["Sharpe Ratio", "Value"] == "n/a"
    assert flat.summary_df.loc["Sharpe 95% Range", "Value"] == "n/a"
    assert flat.summary_df.loc["Win Rate (%)", "Value"] == "n/a"


def test_benchmark_section_only_appears_with_a_benchmark(report):
    assert "Beta" not in report.metrics.index
    assert "Benchmark" not in set(report.summary_df["Section"])


def test_benchmark_days_line_up_with_strategy_days():
    # A strategy that simply holds the benchmark must show beta 1 and correlation 1.
    index = pd.date_range("2024-01-01", periods=60 * 24, freq="h")
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(5).normal(0, 0.005, len(index))),
                      index=index)
    report = Report(close.pct_change().fillna(0.0), [], 10_000, benchmark=pd.DataFrame({"close": close}))

    assert report.metrics["Beta"] == approx(1.0, abs=1e-6)
    assert report.metrics["Correlation"] == approx(1.0, abs=1e-6)
    assert report.metrics["Tracking Error (annualized)"] == approx(0.0, abs=1e-6)
    assert report.metrics["Up Capture"] == approx(1.0, abs=1e-6)


def test_final_balance_compounds_the_returns(report):
    expected = 10_000 * (1 + metrics.total_return(hourly_returns()))
    assert report.metrics["Final Balance"] == approx(expected)
    assert report.metrics["Total Return"] == approx(metrics.total_return(hourly_returns()))


def test_trade_statistics_count_closing_fills_only(report):
    stats = report._compute_trade_stats()

    assert stats["Total Trades"] == 2
    assert stats["Winning Trades"] == 1
    assert stats["Losing Trades"] == 1
    assert stats["Win Rate (%)"] == 50
    assert stats["Avg Profit (%)"] == approx(10.0)   # 10 on a notional of 100
    assert stats["Avg Loss (%)"] == approx(-5.0)
    assert stats["Avg Return (%)"] == approx(2.5)


def test_profit_totals_are_after_commission(report):
    stats = report._compute_trade_stats()

    assert stats["Gross Profit"] == approx(9)          # the +10 trade less 1 commission
    assert stats["Gross Loss"] == approx(-1 - 1 - 6)   # two entries and the -5 trade
    assert stats["Net Profit"] == approx(10 - 5 - 4)


def test_trade_statistics_without_trades():
    stats = Report(hourly_returns(), [], 10_000)._compute_trade_stats()

    assert stats["Total Trades"] == 0
    assert np.isnan(stats["Win Rate (%)"])
    assert stats["Net Profit"] == 0


def test_drawdown_table_lists_worst_first(report):
    table = report.drawdown_table

    assert list(table.columns) == [
        "Net drawdown in %", "Peak date", "Valley date", "Recovery date", "Duration (days)"]
    assert len(table) <= 10
    assert table["Net drawdown in %"].is_monotonic_decreasing
    assert table["Net drawdown in %"].iloc[0] == approx(-metrics.max_drawdown(report.daily_returns) * 100)
    assert (table["Valley date"] >= table["Peak date"]).all()


def test_benchmark_is_cut_to_the_test_period():
    index = pd.date_range("2023-06-01", periods=400, freq="D")
    benchmark = pd.DataFrame({"close": np.linspace(100, 200, 400)}, index=index)
    report = Report(hourly_returns(), [], 10_000, benchmark=benchmark)

    assert report.bench_cum.index.min() >= pd.Timestamp("2024-01-01")
    assert report.bench_cum.iloc[0] == approx(1.0)


def daily_benchmark(days=130):
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({"close": 100 * np.cumprod(1 + np.random.default_rng(9).normal(0, 0.02, days))},
                        index=index)


def test_show_prints_every_section(report, capsys):
    report.show()
    out = capsys.readouterr().out

    assert "Performance Summary" in out
    for section in ("Overview", "Returns", "Risk", "Drawdown", "Sharpe Ratio Reliability", "Trades"):
        assert f"\n{section}\n" in out
    assert "Net drawdown in %" in out
    assert "Charts saved" not in out          # nothing is written without save_path
    assert plt.get_fignums() == []            # figures are closed afterwards


def test_show_saves_each_chart(tmp_path, capsys):
    target = tmp_path / "out"
    Report(hourly_returns(), TRADES, 10_000, save_path=str(target)).show()

    assert sorted(p.name for p in target.iterdir()) == [
        "cumulative_returns.png", "cumulative_returns_log.png", "drawdown.png",
        "monthly_distribution.png", "monthly_heatmap.png", "return_quantiles.png",
        "rolling_sharpe.png", "rolling_volatility.png", "yearly_returns.png"]
    assert f"Charts saved to {target}" in capsys.readouterr().out


def test_show_displays_charts_one_at_a_time_on_an_interactive_backend(report, monkeypatch):
    open_figures = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: open_figures.append(len(plt.get_fignums())))

    report.show()
    assert open_figures == []                 # the test backend cannot display anything

    monkeypatch.setattr("matplotlib.get_backend", lambda: "MacOSX")
    report.show()
    assert open_figures == [1] * 9            # nine charts without trade data, each shown alone


def test_charts_grow_with_the_data_supplied():
    returns = hourly_returns()
    plain = Report(returns, TRADES, 10_000).charts()
    full = Report(returns, TRADES, 10_000, benchmark=daily_benchmark()).charts()

    assert len(plain) == 9
    assert list(full) == list(plain) + ["rolling_beta"]


def test_charts_carry_the_report_title():
    report = Report(hourly_returns(), TRADES, 10_000, title_prefix="SMA cross")
    legend = report.charts()["cumulative_returns"]().axes[0].get_legend()

    assert [t.get_text() for t in legend.get_texts()] == ["SMA cross"]
