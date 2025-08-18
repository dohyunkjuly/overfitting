import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import List, Optional
from overfitting.functions import Data
from overfitting.broker import Broker
from overfitting.order import Order
from overfitting.position import Position
from overfitting.plot.plot import plotting

class Strategy:
    def __init__(self, data: pd.DataFrame, *,
                 initial_capital=100000,
                 commission_rate=0.0002,
                 maint_maring_rate=0.005,
                 maint_amount=0):
        
        self.data = Data(data)
        self.broker = Broker(self.data, 
                             initial_capital, 
                             commission_rate,
                             maint_maring_rate,
                             maint_amount)
        self.balances = []
        self.returns = []
        self.init()

    def __repr__(self):
        return (f"Strategy("
                f"initial_capital={self.broker.initial_captial}, "
                f"commission_rate={self.broker.commission_rate}, "
                f"balances={self.balances}, "
                f"returns={self.returns})")

    @abstractmethod
    def init(self):
        """
        Intended for initializing any parameters specific to the trading strategy. 
        """
    
    @abstractmethod
    def next(self, i):
        """
        It defines the logic of the strategy that will be executed on each step 
        (i.e., for each time period in the dataset). The parameter `i` represents 
        the index of the current time period. This method is called in a loop 
        within the `run` method.
        """

    def limit_order(self, symbol: str, qty: float, price: float) -> Order:
        # Place a new LIMIT order using the broker class.
        return self.broker.order(symbol, qty, price, type='LIMIT')

    def market_order(self, symbol: str, qty: float) -> Order:
        # Place a new MARKET order using the broker class.
        return self.broker.order(symbol, qty, None, type='MARKET')
    
    def stop_limit_order(self, symbol: str, qty: float, price: float, stop_price: float) -> Order:
        # Place a new STOP LIMIT order using the broker class.
        return self.broker.order(symbol, qty, price, type='STOP', stop_price=stop_price)

    def stop_market_order(self, symbol: str, qty: float, stop_price: float) -> Order:
        # Place a new STOP MARKET order using the broker class.
        return self.broker.order(symbol, qty, None, type='STOP', stop_price=stop_price)

    def cancel_order(self, order_id: str) -> Optional[Order]:
        return self.broker.cancel_order(order_id)
    
    def set_leverage(self, symbol: str, leverage: int):
        """
        Sets the leverage for a specific symbol.
        Raises an exception if the updated liquidation price would result 
        in the position being liquidated after changing the leverage.
        """
        self.broker.set_leverage(symbol, leverage)

    def get_position(self, symbol: str) -> Position:
        """
        Fetch the current position of a specific symbol
        """
        return self.broker.get_position(symbol)

    def get_balance(self) -> float:
        """
        Fetch the current balance
        """
        return self.broker.cash

    def get_open_orders(self) -> List[Order]:
        """
        Fetch the current open orders
        """
        return list(self.broker.open_orders)
    
    def run(self) -> pd.Series:
        """
        Executes the strategy over the dataset.

        It handles the iteration over each time period in the data. It calls the 
        user-defined `next` method on each iteration to apply the strategy's logic. 
        Additionally, it updates account balances, and calculates the returns.
    
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

    def plot(self, returns: pd.Series, save_path=None):
        """
        Generates a full performance analysis of the strategy, including trade statistics,
        performance metrics, and visualizations. Outputs are optionally saved to disk.

        Parameters
        ----------
        returns : pd.Series
            A series of periodic strategy returns indexed by datetime.
        save_path : str, optional
            The directory path where plots and visual outputs will be saved.
            If None, plots will only be shown but not saved.
        """
        trades_list = self.broker.trades
        captial = self.broker.initial_captial

        plotting(returns, trades_list, captial, save_path)

    def fetch_trades(self) -> pd.DataFrame:
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
        path : str
            The directory path where the CSV file will be saved.
        filename : str
            The name of the CSV file to save the trade history to.
        """
        full_path = os.path.join(path, filename + '.csv')
        trade_history_df = self.fetch_trades()
        
        trade_history_df.to_csv(full_path, index=False)
