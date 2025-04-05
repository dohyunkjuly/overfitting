import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import overfitting.plot.graph as graph
from scipy.stats import skew, kurtosis
import seaborn as sns
from overfitting.plot.benchmark import backtest_benchmark
from datetime import datetime
from typing import Sequence

def plotting(returns_series: pd.Series, 
             trades_list: Sequence[object], 
             initial_capital: int, 
             save_path: str =None):
    
    start_time = returns_series.index.min()
    end_time = returns_series.index.max()

    cumulative_returns = (1 + returns_series).cumprod()
    cumulative_return = cumulative_returns.iloc[-1]
    final_balance = initial_capital * cumulative_return

    number_of_years = round((pd.to_datetime(end_time) - pd.to_datetime(start_time)).days / 365 , 1)

    def calculate_cagr(cumulative_return, number_of_years):
        if number_of_years <= 0:
            raise ValueError("Number of years should be greater than 0")
        return (cumulative_return) ** (1 / number_of_years) - 1
    
    cagr = calculate_cagr(cumulative_return, number_of_years)

    daily_returns_series = (1+ returns_series).resample('D').prod() -1    
    monthly_returns_series = (1+ returns_series).resample('ME').prod() -1
    monthly_returns_series.index = monthly_returns_series.index.strftime('%Y-%m')

    sharpe_ratio = graph.sharpe_ratio(daily_returns_series, risk_free=0, period='daily')
    sortino_ratio = graph.sortino_ratio(daily_returns_series, required_return=0, period='daily')
    drawdown_table = graph.show_worst_drawdown_periods(daily_returns_series)

    cumulative_returns = (1 + daily_returns_series).cumprod()

    def unpack_trades_list(trades_list):
        df = pd.DataFrame(trades_list)

        # Always include all trades for PnL calc
        gross_returns = df['realized_pnl'].values

        # Only include trades with PnL ≠ 0 (i.e., closed) for return percentages
        closed = df[df['pnl'] != 0].copy()
        notional = (closed['price'].abs() * closed['qty'].abs()).replace(0, np.nan)
        return_percents = (closed['pnl'] / notional).fillna(0).values

        return gross_returns, return_percents
        
    gross_returns, return_percents = unpack_trades_list(trades_list)
    stats = graph.trade_summary(gross_returns, return_percents)

    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns/peak) - 1
    daily_value_at_risk = graph.value_at_risk(daily_returns_series, sigma=2, period=None)
    skew_value = skew(monthly_returns_series)
    kurtosis_value = kurtosis(monthly_returns_series, fisher=False)

    # Create and format the summary dictionary
    summary = {
        "Number of Years": number_of_years,
        "Start Date": start_time,
        "End Date": end_time,
        "Initial Balance": float(initial_capital),
        "Final Balance": final_balance,
        "CAGR": cagr,
        "Cumulative Return": cumulative_returns.iloc[-1],
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": min(drawdown),
        "Daily Value At Risk": daily_value_at_risk,
        "Skew": skew_value,
        "Kurtosis": kurtosis_value,
    }

    # Add trade stats
    summary.update(stats.to_dict())
    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=[''])
    summary_df[''] = summary_df[''].apply(
        lambda x: f"{x:,.8f}" if isinstance(x, float) else x)
    
    print('Performance Summary')
    with pd.option_context('display.colheader_justify', 'left', 'display.width', None):
        print(summary_df.to_string(header=False))
    print(drawdown_table)

    # Helper function for saveing figure
    def save_figure(name, save_path=None):
        if save_path:
            full_path = save_path + name
            plt.savefig(full_path, format='jpg')
            return full_path
        
    ####### Plot the culmulative returns with benchmark (Incomplete)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Simulation', color = 'blue')  
    plt.xlabel('Date')
    plt.ylabel('Culmulative Returns')
    plt.title('Culmulative Returns')
    plt.legend()
    plt.grid(True)
    save_figure('/cumulative_returns.jpg', save_path)
    plt.show()

    def custom_log(x):
        return np.sign(x) * np.log(np.abs(x))
    
    culmulative_returns_log_scale = cumulative_returns.apply(custom_log)

    ####### Plot the culmulative returns on a logartihmic scale with Benchmark (Incomplete)
    plt.figure(figsize=(12,6))
    plt.plot(culmulative_returns_log_scale, label = 'Simulation', color = 'blue')
    plt.xlabel('Date')
    plt.ylabel('Culmulative Returns')
    plt.title('Culmulative Returns on a logartihmic scale')
    plt.legend()
    plt.grid(True)
    save_figure('/culmulative_returns.jpg', save_path)
    plt.show()

    # Plot daily returns
    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns_series, label='Simulation')  
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Daily Returns')
    plt.legend()
    plt.grid(False)
    save_figure('daily_returns.jpg', save_path)
    plt.show()

    # Plot Monthly return heatmap
    monthly_return_heatmap = graph.monthly_returns_heatmap(daily_returns_series)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "white", "blue"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

    plt.figure(figsize=(12, 5))
    # convert to percentage
    monthly_return_heatmap = monthly_return_heatmap * 100
    sns.heatmap(monthly_return_heatmap, cmap=cmap, annot=True, fmt=".1f", center=0)
    plt.title('Monthly retruns (%)')
    save_figure('monthly_returns_heatmap.jpg', save_path)
    plt.show()

    # Plot the Drawdown
    plt.figure(figsize=(12, 6))
    plt.fill_between(drawdown.index, drawdown.values, color='red', alpha=1) 
    plt.plot(drawdown, label='Simulation', color='red') 
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Daily Drawdown')
    plt.legend()
    plt.grid(False)
    save_figure('daily_drawdown.jpg', save_path)
    plt.show()

    # Plot Sharpe Ratio (6 months)
    rolling_sharpe = graph.rolling_sharpe(daily_returns_series, factor_returns=None, rolling_window= 180) #rolling_window is days
    # print(rolling_sharpe)

    # Calculate mean value
    rolling_sharpe_mean_value = rolling_sharpe.mean()

    # Draw graph
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_sharpe, label='Simulation', color='blue')
    plt.axhline(y=rolling_sharpe_mean_value, color='grey', linestyle='--', label=f'Average: {rolling_sharpe_mean_value:.3f}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Rolling Sharpe Ratio (6 months)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_figure('rolling_sharpe.jpg', save_path)
    plt.show()


    # Plot Rolling Volatility (6 months) with benchmark volatility
    rolling_volatility = graph.rolling_volatility(daily_returns_series, factor_returns=None, rolling_window=180)

    # Calculate mean value
    rolling_volatility_mean_value = rolling_volatility.mean()

    # Draw graph
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_volatility, label='Simulation', color='blue')
    plt.axhline(y=rolling_volatility_mean_value, color='grey', linestyle='--', label=f'Simulation Average: {rolling_volatility_mean_value:.3f}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Rolling Volatility (6 months)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_figure('rolling_volatility.jpg', save_path)
    plt.show()


    ######## Plot Distrubtion of Monthly Returns (Incomplete)
    monthly_returns_dist = graph.monthly_returns_dist(daily_returns_series)

    monthly_returns_dist_mean_value = monthly_returns_dist.mean()

    plt.figure(figsize=(10, 6))

    plt.hist(monthly_returns_dist)
    plt.axvline(x=monthly_returns_dist_mean_value, color='grey', linestyle='--', label=f'Average: {monthly_returns_dist_mean_value:.3f}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.title('Distribution of Monthly Returns')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    save_figure('monthly_returns_dist.jpg', save_path)
    plt.show()