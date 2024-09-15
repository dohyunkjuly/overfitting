import pandas as pd
import matplotlib.pyplot as plt

import sys
import overfitting.plot.graph as graph

pd.set_option('display.float_format', '{:.8f}'.format)

def backtest_benchmark(start_time : str, end_time : str, initial_capital : int, data : pd.DataFrame):

    minute_data = data
    minute_data['open_time'] = pd.to_datetime(minute_data['open_time'], unit='ms')
    minute_data['close_time'] = pd.to_datetime(minute_data['close_time'], unit= 'ms')
    minute_data['primary_key'] = range(1, len(minute_data) + 1)
    minute_data['index_datetime'] = minute_data['open_time']

    minute_data.set_index('index_datetime', inplace=True)
    minute_data = minute_data.loc[start_time:end_time]

    minute_data.drop(minute_data.index[-1], inplace=True)

    total_capital = [initial_capital]
    minute_returns = (minute_data['open'] - minute_data['open'].shift(1)) / minute_data['open'].shift(1)

    minute_returns.fillna(0, inplace=True)

    daily_returns = (1+minute_returns).resample("D").prod() - 1

    monthly_returns = (1+minute_returns).resample("M").prod() - 1
    monthly_returns.index = monthly_returns.index.strftime('%Y-%m')

    # benchmark_asset_balance = ( 1 + minute_returns ).cumprod()*initial_capital

    benchmark_cumulative_returns = (1 + daily_returns).cumprod()

    rolling_volatility = graph.rolling_volatility(daily_returns, factor_returns=None, rolling_window=180)
    
    return benchmark_cumulative_returns, rolling_volatility
    
if(__name__ == "__main__"):

    start_time = "2020-01-01 00:00:00"   # ex) "2020-02-13 13:59:00"
    end_time = "2023-10-01 00:00:00"     # ex) "2023-02-13 13:59:00"
    initial_capital = 1_000_000
    minute_data = pd.read_csv('./csv_files/binance_spot_BTCUSDT_1m_2020-01-01-2023-08-16.csv')

    benchmark_cumulative_returns, rolling_volatility = backtest_benchmark(start_time, end_time, 1000000, minute_data)
    
    print(rolling_volatility)

    plt.figure(figsize=(10, 6))
    plt.plot(rolling_volatility)
    plt.title("Rolling Volatility Over Time")
    plt.xlabel("Time")
    plt.ylabel("Rolling Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_cumulative_returns)
    plt.title("Cumulative Returns Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()