import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from overfitting.functions import Data
from overfitting.plot.plot import plotting
from overfitting.broker import Broker
from overfitting.error import InitializationError

class Strategy:
    def __init__(self, data: pd.DataFrame, *,
                 initial_capital=1000000,
                 commission_rate=0.0002,
                 slippage_rate=0):
        # Handles dataframe
        self.data = Data()
        self._conviert_into_numpy(data)
        
        # Initialize Broker class
        self.broker = Broker(self.data, 
                             initial_capital, 
                             commission_rate, 
                             slippage_rate)
        
        self.balances = []
        self.returns = []
        self.init()

    def _conviert_into_numpy(self, data):
        """
        Convert the input DataFrame columns into Numpy Arrays and
        store them as attributes of the DoDict
        """
        if not isinstance(data, pd.DataFrame):
            raise InitializationError("data must be a pd.DataFrame")

        self.data['timestamp'] = data.index.to_numpy()
        self.data.update({col: data[col].to_numpy() for col in data.columns})

    def __repr__(self):
        return (f"Strategy("
                f"initial_capital={self.broker.initial_captial}, "
                f"commission_rate={self.broker.commission_rate}, "
                f"slippage_rate={self.broker.slippage_rate}, "
                f"balances={self.balances}, "
                f"returns={self.returns})")

    @abstractmethod
    def init(self):
        """
        Abstract method to be implemented by the user.

        Intended for initializing any settings specific to the trading strategy. 
        It is called once when the strategy is instantiated.
        """
    
    @abstractmethod
    def next(self, i):
        """
        Abstract method to be implemented by the user.

        It defines the logic of the strategy that will be executed on each step 
        (i.e., for each time period in the dataset). The parameter `i` represents 
        the index of the current time period. This method is called in a loop 
        within the `run` method.
        """

    def limit_order(self, symbol: str, qty: float, price: float):
        # Place a new limit order using the broker class.
        return self.broker.order(symbol, qty, price, type='limit')

    def market_order(self,symbol: str, qty: float):
        # Place a market order using the broker class.
        return self.broker.order(symbol, qty, None, type='market')
    
    def set_leverage(self, symbol, leverage):
        """
        Sets the leverage for a specific symbol.

        Raises an exception if the updated liquidation price would result 
        in the position being liquidated after changing the leverage.
        """
        self.broker.set_leverage(symbol, leverage)

    def get_position(self, symbol):
        """Fetch the current position of a specific symbol"""
        return self.broker.get_position(symbol)

    def run(self):
        """
        Executes the strategy over the dataset.

        It handles the iteration over each time period in the data. It calls the 
        user-defined `next` method on each iteration to apply the strategy's logic. 
        Additionally, it checks for expired contracts and settles them if necessary, 
        updates account balances, and calculates the returns for each time period.
    
        Returns:
            A pandas Series containing the returns, indexed by the corresponding timestamps.
        """
        t = pd.to_datetime(self.data['timestamp'])
        # Pre-allocate lists for balance and returns
        b = np.zeros(len(t))
        r = np.zeros(len(t))

        for i in range(len(t)):
            self.next(i)
            self.broker.next()

            # Update Balance
            b[i] = self.broker.cash

            if i > 0:
                # Updates the Returns
                pb = b[i-1] # previous balance
                r[i] = (b[i] - pb) / pb

        self.balances = b.tolist()
        self.returns = r.tolist()

        return pd.Series(self.returns, index=t.tolist())

    def plot(self, returns: pd.Series, start_time, end_time, save_path=None):
        """
        Generates a full performance analysis of the strategy, including trade statistics,
        performance metrics, and visualizations. Outputs are optionally saved to disk.

        Parameters
        ----------
        returns : pd.Series
            A series of periodic strategy returns indexed by datetime.
        start_time : datetime
            The datetime representing the start of the backtest period.
        end_time : datetime
            The datetime representing the end of the backtest period.
        save_path : str, optional
            The directory path where plots and visual outputs will be saved.
            If None, plots will only be shown but not saved.
        """
        trades_list = self.broker.trades
        captial = self.broker.initial_captial

        plotting(returns, trades_list, start_time, end_time, captial, save_path)

    def fetch_trades(self):
        """
        Returns the trade history as a pandas DataFrame.

        Returns:
            A pandas DataFrame where each row represents a trade.
        """
        return pd.DataFrame(self.broker.trades)
    
    def save_trades_to_csv(self, path='', filename="trade_history"):
        """
        Save the trade history to a CSV file.

        Parameters
        ----------
        path (str): 
            The directory path where the CSV file will be saved.
        filename (str):
            The name of the CSV file to save the trade history to.
        """
        full_path = os.path.join(path, filename + '.csv')
        trade_history_df = self.fetch_trades()
        
        trade_history_df.to_csv(full_path, index=False)
