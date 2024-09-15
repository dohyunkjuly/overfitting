import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import overfitting.plot.graph as graph
from scipy.stats import skew, kurtosis
import seaborn as sns
from overfitting.plot.benchmark import backtest_benchmark

def custom_log(x):
    return np.sign(x) * np.log(np.abs(x))

def plotting(returns_series : pd.Series,  start_time, end_time, initial_capital : int):

    cumulative_returns = (1 + returns_series).cumprod()
    cumulative_return = cumulative_returns[-1]
    
    final_balance = initial_capital * cumulative_return

    def calculate_cagr(cumulative_return, number_of_years):
        #Parameters:
        #- cumulative_return: The cumulative return as a decimal (e.g., 1.5 for a 150% return)
        #- number_of_years: The number of years over which the return was achieved
        
        #Returns:
        #- The CAGR as a decimal (e.g., 0.1 fo(r a 10% annual return)

        if number_of_years <= 0:
            raise ValueError("Number of years should be greater than 0")
        return (cumulative_return) ** (1 / number_of_years) - 1

    number_of_years = round((pd.to_datetime(end_time) - pd.to_datetime(start_time)).days / 365 , 1)
    print(f'Number of Years : {number_of_years}')

    cagr = calculate_cagr(cumulative_return, number_of_years)

    daily_returns_series = (1+ returns_series).resample('D').prod() -1
    
    monthly_returns_series = (1+ returns_series).resample('M').prod() -1
    monthly_returns_series.index = monthly_returns_series.index.strftime('%Y-%m')


    sharpe_ratio = graph.sharpe_ratio(daily_returns_series, risk_free=0, period='daily')
    sortino_ratio = graph.sortino_ratio(daily_returns_series, required_return=0, period='daily')
    drawdown_table = graph.show_worst_drawdown_periods(daily_returns_series)

    cumulative_returns = (1 + daily_returns_series).cumprod()

    peak = cumulative_returns.expanding(min_periods=1).max()

    drawdown = (cumulative_returns/peak) - 1
    daily_value_at_risk = graph.value_at_risk(daily_returns_series, sigma=2, period=None)
    skew_value = skew(monthly_returns_series)
    kurtosis_value = kurtosis(monthly_returns_series, fisher=False)

    print(f"Asset: USDT")
    print(f"Start Date: {start_time}")
    print(f"End Date: {end_time}")
    print(f"Initial Balnace: {initial_capital}")
    print(f"Final Balance: {final_balance}")
    print(f'CAGR {cagr}')
    print(f'Culmulative Return: {cumulative_returns[-1]}')
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Sortino: {sortino_ratio}")
    print(f"Max Drawdown: {min(drawdown)} ")
    print(f"Daily Value At Risk: {daily_value_at_risk}")
    print(f"Skew: {skew_value}")
    print(f"Kurtosis: {kurtosis_value}")

    print(drawdown_table)

    ####### Plot the culmulative returns with benchmark (Incomplete)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Simulation', color = 'green')  
    plt.xlabel('Date')
    plt.ylabel('Culmulative Returns')
    plt.title('Culmulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('../graphs/culmulative_returns.jpg', format = 'jpg')
    plt.show()


    culmulative_returns_log_scale = cumulative_returns.apply(custom_log)

    ####### Plot the culmulative returns on a logartihmic scale with Benchmark (Incomplete)
    plt.figure(figsize=(12,6))
    plt.plot(culmulative_returns_log_scale, label = 'Simulation', color = 'green')
    plt.xlabel('Date')
    plt.ylabel('Culmulative Returns')
    plt.title('Culmulative Returns on a logartihmic scale')
    plt.legend()
    plt.grid(True)
    plt.savefig('../graphs/culmulative_returns_log_scale.jpg', format = 'jpg')
    plt.show()

    # Plot daily returns
    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns_series, label='Simulation')  
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Daily Returns')
    plt.legend()
    plt.grid(False)
    plt.savefig('../graphs/daily_returns.jpg', format = 'jpg')
    plt.show()


    # Plot Monthly return heatmap
    monthly_return_heatmap = graph.monthly_returns_heatmap(daily_returns_series)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "white", "green"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

    plt.figure(figsize=(12, 5))
    # convert to percentage
    monthly_return_heatmap = monthly_return_heatmap * 100
    sns.heatmap(monthly_return_heatmap, cmap=cmap, annot=True, fmt=".1f", center=0)
    plt.title('Monthly retruns (%)')
    plt.savefig('../graphs/monthly_returns_heatmap.jpg', format = 'jpg')
    plt.show()

    # Plot the Drawdown
    plt.figure(figsize=(12, 6))
    plt.fill_between(drawdown.index, drawdown.values, color='orange', alpha=1) 
    plt.plot(drawdown, label='Simulation', color='orange') 
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Daily Drawdown')
    plt.legend()
    plt.grid(False)
    plt.savefig('../graphs/daily_drawdown.jpg', format = 'jpg')
    plt.show()

    # Plot Sharpe Ratio (6 months)
    rolling_sharpe = graph.rolling_sharpe(daily_returns_series, factor_returns=None, rolling_window= 180) #rolling_window is days
    # print(rolling_sharpe)

    # Calculate mean value
    rolling_sharpe_mean_value = rolling_sharpe.mean()

    # Draw graph
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_sharpe, label='Simulation', color='green')
    plt.axhline(y=rolling_sharpe_mean_value, color='red', linestyle='--', label=f'Average: {rolling_sharpe_mean_value:.3f}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Rolling Sharpe Ratio (6 months)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('../graphs/rolling_sharpe.jpg', format = 'jpg')
    plt.show()


    # Plot Rolling Volatility (6 months) with benchmark volatility
    rolling_volatility = graph.rolling_volatility(daily_returns_series, factor_returns=None, rolling_window=180)

    # Calculate mean value
    rolling_volatility_mean_value = rolling_volatility.mean()

    # Draw graph
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_volatility, label='Simulation', color='green')
    plt.axhline(y=rolling_volatility_mean_value, color='red', linestyle='--', label=f'Simulation Average: {rolling_volatility_mean_value:.3f}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Rolling Volatility (6 months)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('../graphs/rolling_volatility.jpg', format = 'jpg')
    plt.show()


    ######## Plot Distrubtion of Monthly Returns (Incomplete)
    monthly_returns_dist = graph.monthly_returns_dist(daily_returns_series)

    monthly_returns_dist_mean_value = monthly_returns_dist.mean()

    plt.figure(figsize=(10, 6))

    plt.hist(monthly_returns_dist)
    plt.axvline(x=monthly_returns_dist_mean_value, color='red', linestyle='--', label=f'Average: {monthly_returns_dist_mean_value:.3f}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.title('Distribution of Monthly Returns')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('../graphs/monthly_returns_dist.jpg', format = 'jpg')
    plt.show()