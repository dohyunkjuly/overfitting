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
                 initial_captial=1000000,
                 commission_rate=0.0002,
                 slippage_rate=0):
        # Handles dataframe
        self.data = Data()
        self._conviert_into_numpy(data)
        
        # Initialize Broker class
        self.broker = Broker(self.data, 
                             initial_captial, 
                             commission_rate, 
                             slippage_rate)
        
        self.balances = []
        self.returns = []
        # Initialize iterator
        self._i = 0
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

    def limit_order(self, symbol, qty, price, direction):
        # Place a new limit order using the broker class.
        t = self.data['timestamp'][self._i]
        return self.broker.order(t, symbol, qty, price, direction, type='limit')

    def market_order(self,symbol, qty, direction):
        # Place a market order using the broker class.
        t = self.data['timestamp'][self._i]
        return self.broker.order(t, symbol, qty, None, direction, type='market')
    

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
            pass
