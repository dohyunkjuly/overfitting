import inspect
import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import List, Optional, Dict, Type
from overfitting.data import Data
from overfitting.broker import Broker
from overfitting.order import Order
from overfitting.position import Position
from overfitting.analysis.report import Report
from overfitting.slippage import SlippageModel
from overfitting.indicators import Indicator

_OHLCV_NAMES = ("open", "high", "low", "close", "volume")

class Strategy:
    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 *,
                 benchmark: Optional[pd.DataFrame] = None,
                 initial_capital: float =100000,
                 commission_rate: float =0.0002,
                 maint_margin_rate: float =0.005,
                 maint_amount: float=0,
                 slippage_model: Optional[SlippageModel] = None):

        self.benchmark = benchmark
        self.data = Data(data)
        self.broker = Broker(
            data=self.data, 
            cash=initial_capital, 
            commission_rate=commission_rate,
            maint_margin_rate=maint_margin_rate,
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

    def limit_order(self, symbol: str, qty: float, price: float, label: Optional[str] = None) -> Order:
        return self.broker.order(symbol, qty, price, type="LIMIT", label=label)

    def market_order(self, symbol: str, qty: float, label: Optional[str] = None) -> Order:
        return self.broker.order(symbol, qty, None, type="MARKET", label=label)

    def stop_limit_order(
        self,
        symbol: str,
        qty: float,
        price: float,
        stop_price: float,
        label: Optional[str] = None,
    ) -> Order:
        return self.broker.order(symbol, qty, price, type="STOP", stop_price=stop_price, label=label)

    def stop_market_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        label: Optional[str] = None,
    ) -> Order:
        return self.broker.order(symbol, qty, None, type="STOP", stop_price=stop_price, label=label)

    def cancel_order(self, symbol, order_id: str) -> Optional[Order]:
        return self.broker.cancel_order(symbol, order_id)
    
    def cancel_all_orders(self, symbol):
        """
        Cancel all open orders for a specific symbol
        """
        self.broker.cancel_all_orders(symbol)
    
    def close_all_positions(self, symbol):
        """
        Close all open positions through market order for a specific symbol
        """
        self.broker.close_all_positions(symbol)

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

    def get_open_orders(self, symbol: str) -> Dict[str, Order]:
        """
        Fetch the current open orders
        """
        return dict(self.broker.open_orders.get(symbol, {}))

    def open(self, symbol: str, i: Optional[int] = None):
        return self._ohlcv(symbol, "open", i)

    def high(self, symbol: str, i: Optional[int] = None):
        return self._ohlcv(symbol, "high", i)

    def low(self, symbol: str, i: Optional[int] = None):
        return self._ohlcv(symbol, "low", i)

    def close(self, symbol: str, i: Optional[int] = None):
        return self._ohlcv(symbol, "close", i)

    def _ohlcv(self, symbol: str, name: str, i: Optional[int]):
        """Scalar at bar ``i`` when ``i`` is given; otherwise the full pandas Series."""
        arr = getattr(self.broker._d(symbol), name)
        if i is None:
            return pd.Series(arr, name=name)
        return arr[i]

    def indicator(self, cls: Type[Indicator], symbol: str, *, source: str = "close", **kwargs) -> Indicator:
        """
        Build an indicator by binding the symbol's OHLCV series to ``cls``'s constructor.

        - Constructor params named one of ``open / high / low / close / volume``
          are filled with the matching series for ``symbol``.
        - A param named ``series`` is filled with the ``source`` column
          (default ``"close"``; override per-call with ``source=`` ).
        - Any other kwargs are forwarded as-is.
        """
        sig = inspect.signature(cls.__init__)
        bound = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in _OHLCV_NAMES:
                bound[name] = self._ohlcv(symbol, name, None)
            elif name == "series":
                if source not in _OHLCV_NAMES:
                    raise ValueError(
                        f"source={source!r} must be one of {_OHLCV_NAMES}"
                    )
                bound[name] = self._ohlcv(symbol, source, None)
            elif name in kwargs:
                bound[name] = kwargs.pop(name)
            # else: rely on the indicator's default

        if kwargs:
            raise TypeError(
                f"{cls.__name__}: unexpected keyword arguments {list(kwargs)}"
            )

        return cls(**bound)

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
            raise AttributeError(f"Col '{col}' not found for {symbol}.")
        
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

        p = Report(
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


