import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import List, Optional, Union, Dict
from overfitting.data import Data, MultiCurrency
from overfitting.broker import Broker
from overfitting.order import Order
from overfitting.position import Position
from overfitting.plot.performance import PerformanceReport
from overfitting.execution.slippage import SlippageModel

class Strategy:
    def __init__(self, 
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                 *,
                 benchmark: Optional[pd.DataFrame] = None,
                 initial_capital: float =100000,
                 commission_rate: float =0.0002,
                 maint_maring_rate: float =0.005,
                 maint_amount: float=0,
                 slippage_model: Optional[SlippageModel] = None):
        
        self.benchmark = benchmark
        self.data = MultiCurrency(data) if isinstance(data, dict) else Data(data)
        self.broker = Broker(
            data=self.data, 
            cash=initial_capital, 
            commission_rate=commission_rate,
            maint_maring_rate=maint_maring_rate,
            maint_amount=maint_amount,
            slippage_model=slippage_model
        )
        self.balances = []
        self.returns= []
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
    
    def open(self, symbol: str, i: int):
        return self.broker._open(symbol, i)
    
    def high(self, symbol: str, i: int):
        return self.broker._high(symbol, i)
    
    def low(self, symbol: str, i: int):
        return self.broker._low(symbol, i)
    
    def close(self, symbol: str, i: int):
        return self.broker._close(symbol, i)

    def bars(self, symbol: str, i: int) -> tuple:
        """
        Returns Tuple - open, high, low, close
        """
        return self.broker._bars(symbol, i)

    def val(self, symbol: str, i: int, col: str):
        """
        Fetch the target column from target index
        """
        d: pd.DataFrame = self.broker._d(symbol)
        target_column = getattr(d, col, None)
        if target_column is None:
            raise AttributeError(f"Col '{col}' not found for {symbol}. Available: {', '.join(d.columns)}")
        return target_column[i]

    def run(self) -> pd.Series:
        """
        Executes the strategy over the dataset.

        It handles the iteration over each time period in the data. It calls the 
        user-defined `next` method on each iteration to apply the strategy's logic. 
        Additionally, it updates account balances, and calculates the returns.
    
        Returns:
            A pandas Series containing the returns, indexed by the corresponding timestamps.
        """
        t = pd.to_datetime(self.data.index)
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

    def plot(self, returns: pd.Series, save_path=None, title="Simulation"):
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

        p = PerformanceReport(
            returns_series=returns, 
            trades_list=trades_list, 
            initial_capital=captial, 
            benchmark=self.benchmark,
            save_path=save_path,
            title_prefix=title
        )
        p.show()

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
