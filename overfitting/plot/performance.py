import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import overfitting.plot.graph as graph # Helper Functions

from typing import Optional, Sequence
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skew, kurtosis

class PerformanceReport:
    def __init__(self,
                 returns_series: pd.Series,
                 trades_list: Sequence[object],
                 initial_capital: int,
                 benchmark: Optional[pd.DataFrame] = None,
                 save_path: Optional[str] = None,
                 title_prefix: str = "Simulation"):
        
        if not isinstance(returns_series.index, pd.DatetimeIndex):
            raise ValueError("returns_series must have a DatetimeIndex")

        self.title_prefix = title_prefix
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

    def show(self):
        # Plot cumulative returns vs benchmark
        self.plot_cumulative()

        # Plot cumulative returns on a log scale vs benchmark
        self.plot_cumulative_log()

        # Plot daily returns over time
        self.plot_daily_returns()

        # Plot heatmap of monthly returns (%)
        self.plot_monthly_heatmap()

        # Plot drawdowns (decimal form)
        self.plot_drawdown()

        # Plot rolling Sharpe ratio (risk-adjusted returns over time) vs benchmark
        self.plot_rolling_sharpe()

        # Plot rolling volatility (risk level over time)
        self.plot_rolling_vol()

        # Plot distribution of monthly returns
        self.plot_monthly_dist()

    def lg(self, x):
        return np.sign(x) * np.log(np.abs(x))

    def _save(self, name: str):
        if self.save_path:
            plt.savefig(self.save_path + name, format="jpg")

    # ---------- Plots ----------

    def plot_cumulative(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.cum, label=self.title_prefix, color="#656EF2")

        if self.bench_cum is not None: # Benchmark
            plt.plot(self.bench_cum, label="Benchmark", color="grey", alpha=0.9)

        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.title("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        self._save("/cumulative_returns.jpg")
        plt.show()
    
    def plot_cumulative_log(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.cum.apply(self.lg), label=self.title_prefix, color="#656EF2")

        if self.bench_cum is not None: # Benchmark
            plt.plot(self.bench_cum.apply(self.lg), label="Benchmark", color="grey", alpha=0.9)

        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns (log)")
        plt.title("Cumulative Returns (log scale)")
        plt.legend()
        plt.grid(True)
        self._save("/cumulative_returns_log.jpg")
        plt.show()

    def plot_daily_returns(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.daily_returns, label=self.title_prefix, color="#656EF2")

        if self.bench_daily is not None: # benchmark
            plt.plot(self.bench_daily, label="Benchmark", color="grey", alpha=0.7)

        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.title("Daily Returns")
        plt.legend()
        plt.grid(False)
        self._save("/daily_returns.jpg")
        plt.show()

    def plot_monthly_heatmap(self):
        monthly_heatmap = graph.monthly_returns_heatmap(self.daily_returns) * 100
        colors = ["#8B0000", "white", "#96B0C1"]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)
        plt.figure(figsize=(12, 5))
        sns.heatmap(monthly_heatmap, cmap=cmap, annot=True, fmt=".1f", center=0)
        plt.title("Monthly returns (%)")
        self._save("/monthly_returns_heatmap.jpg")
        plt.show()

    def plot_drawdown(self):
        dd = self.drawdown_series
        plt.figure(figsize=(12, 6))
        plt.fill_between(dd.index, dd.values, color="#FF6666", alpha=1)
        plt.plot(dd, color="#FF6666")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.title("Daily Drawdown")
        plt.legend([self.title_prefix])
        plt.grid(False)
        self._save("/daily_drawdown.jpg")
        plt.show()

    def plot_rolling_sharpe(self, window_days: int = 180):
        rs = graph.rolling_sharpe(self.daily_returns, rolling_window=window_days)
        plt.figure(figsize=(10, 6))
        plt.plot(rs, label=self.title_prefix, color="#656EF2")
        plt.axhline(rs.mean(), color="grey", linestyle=":", label=f"Sim Avg: {rs.mean():.3f}")

        if self.bench_daily is not None: # Benchmark
            brs = graph.rolling_sharpe(self.bench_daily, rolling_window=window_days)
            plt.plot(brs, label="Benchmark", color="#FF914D", alpha=0.85)
            plt.axhline(brs.mean(), color="#F2C0A1", linestyle=":", label=f"Bench Avg: {brs.mean():.3f}")

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(f"Rolling Sharpe ({window_days} days)")
        plt.legend(); plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        self._save("/rolling_sharpe.jpg")
        plt.show()

    def plot_rolling_vol(self, window_days: int = 180):
        rv = graph.rolling_volatility(self.daily_returns, rolling_window=window_days)
        plt.figure(figsize=(10, 6))
        plt.plot(rv, label=self.title_prefix, color="#656EF2")
        plt.axhline(rv.mean(), color="grey", linestyle=":", label=f"Sim Avg: {rv.mean():.3f}")

        if self.bench_daily is not None: # Benchmark
            brv = graph.rolling_volatility(self.bench_daily, rolling_window=window_days)
            plt.plot(brv, label="Benchmark", color="#FF914D", alpha=0.85)
            plt.axhline(brv.mean(), color="#F2C0A1", linestyle=":", label=f"Bench Avg: {brv.mean():.3f}")

        plt.xlabel("Date"); plt.ylabel("Value")
        plt.title(f"Rolling Volatility ({window_days} days)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        self._save("/rolling_volatility.jpg")
        plt.show()

    def plot_monthly_dist(self):
        mdist = graph.monthly_returns_dist(self.daily_returns)
        meanv = mdist.mean()
        plt.figure(figsize=(10, 6))
        plt.hist(mdist, color="#577CBB")
        plt.axvline(meanv, color="grey", linestyle="--", label=f"Average: {meanv:.3f}")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.title("Distribution of Monthly Returns")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        self._save("/monthly_returns_dist.jpg")
        plt.show()

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

        # daily frequency + window align
        bench_cum = bench_cum.asfreq("D").interpolate(limit_direction="both")
        bench_daily = bench_cum.pct_change().ffill()
        return bench_cum, bench_daily


    def _unpack_trades_list(self, trades_list):
        if not trades_list:
            return np.array([]), np.array([])
        df = pd.DataFrame(trades_list)
        gross_returns = df.get("realized_pnl", pd.Series(dtype=float)).values

        # closed trades only for percentage returns if 'pnl' exists
        if "pnl" in df and "price" in df and "qty" in df:
            closed = df[df["pnl"] != 0].copy()
            notional = (closed["price"].abs() * closed["qty"].abs()).replace(0, np.nan)
            return_percents = (closed["pnl"] / notional).ffill().values
        else:
            return_percents = np.array([])
        return gross_returns, return_percents

    def _compute_summary(self):
        peak = self.cum.expanding(min_periods=1).max()
        drawdown = (self.cum / peak) - 1
        dvt = graph.value_at_risk(self.daily_returns, sigma=2, period=None)

        # monthly stats for skew/kurt
        monthly = (1 + self.daily_returns).resample("ME").prod().sub(1)
        monthly.index = monthly.index.strftime("%Y-%m")
        skew_val = skew(monthly) if len(monthly) > 2 else np.nan
        kurt_val = kurtosis(monthly, fisher=False) if len(monthly) > 2 else np.nan

        years = max(1e-9, (pd.to_datetime(self.end) - pd.to_datetime(self.start)).days / 365)
        cum_return = float(self.cum.iloc[-1])
        cagr = (cum_return ** (1 / years)) - 1 if cum_return > 0 else np.nan

        sharpe = graph.sharpe_ratio(self.daily_returns, risk_free=0, period="daily")
        sortino = graph.sortino_ratio(self.daily_returns, required_return=0, period="daily")

        summary = {
            "Number of Years": round(years, 2),
            "Start Date": self.start,
            "End Date": self.end,
            "Initial Balance": self.initial_capital,
            "Final Balance": self.initial_capital * cum_return,
            "CAGR": cagr,
            "Cumulative Return": cum_return,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": float(drawdown.min()),
            "Daily Value At Risk": float(dvt),
            "Skew": float(skew_val) if pd.notna(skew_val) else np.nan,
            "Kurtosis": float(kurt_val) if pd.notna(kurt_val) else np.nan,
        }

        # trade stats
        stats = graph.trade_summary(self.gross_returns, self.return_percents)
        summary.update(stats.to_dict())

        df = pd.DataFrame.from_dict(summary, orient="index", columns=[""])
        df[""] = df[""].apply(lambda x: f"{x:,.8f}" if isinstance(x, float) else x)

        print("Performance Summary")
        with pd.option_context("display.colheader_justify", "left", "display.width", None):
            print(df.to_string(header=False))

        dd_table = graph.show_worst_drawdown_periods(self.daily_returns)
        print(dd_table)

        return df, drawdown, dd_table