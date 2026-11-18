import inspect
import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Dict, Optional
from overfitting.data import Data
from overfitting.broker import Broker
from overfitting.order import Order
from overfitting.position import Position
from overfitting.analysis.report import Report
from types import SimpleNamespace
from overfitting.indicators import Line, Lines

class Strategy:
    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 *,
                 benchmark: Optional[pd.DataFrame] = None,
                 initial_capital: float =100000,
                 commission_rate: float =0.0002,
                 maint_margin_rate: float =0.005,
                 maint_amount: float=0,
                 slippage=None,
                 params: Optional[dict] = None,
                 record: bool = True):
        """
        data: {symbol: DataFrame with open, high, low, close}.
        params: values for the strategy's parameters, which are its class
            attributes. `MyStrategy(data, params={"fast": 10})` runs with
            `self.fast == 10` whatever the class says. Unknown names raise.
        record: False skips recording round trips and per-bar history. The
            run is faster and returns are identical; the fetch_* tables for
            round trips, account and positions come back empty.
        slippage: a number (0.001 means every fill is 0.1% worse), a function
            `(order, bar) -> executed price`, or None. See overfitting.slippage.
        """
        for name, value in (params or {}).items():
            if not hasattr(type(self), name):
                raise AttributeError(
                    f"{type(self).__name__} has no parameter {name!r}. Parameters are class "
                    f"attributes, for example `{name} = {value!r}` in the class body.")
            setattr(self, name, value)
        self.params = dict(params or {})

        self.benchmark = benchmark
        self.data = data if isinstance(data, Data) else Data(data)
        self.broker = Broker(
            data=self.data, 
            cash=initial_capital, 
            commission_rate=commission_rate,
            maint_margin_rate=maint_margin_rate,
            maint_amount=maint_amount,
            slippage=slippage,
            record=record,
        )
        self.balances = []           # the curve returns are computed from
        self.equity_curve = []       # cash + unrealized PnL at each bar's close
        self.realized_balances = []  # cash only; moves when trades close
        self.returns= []
        # Position in the data. Bars before `i` have completed; bar `i` is the
        # one orders placed now will trade on.
        self._cursor = SimpleNamespace(i=0)
        # next() may be written as next(self) or next(self, i).
        self._next_takes_index = len(inspect.signature(self.next).parameters) >= 1
        self.init()

    def __repr__(self):
        return (f"Strategy("
                f"initial_capital={self.broker.initial_capital}, "
                f"commission_rate={self.broker.commission_rate}, "
                f"balances={self.balances}, "
                f"returns={self.returns})")

    @abstractmethod
    def init(self):
        """
        Intended for initializing any parameters specific to the trading strategy. 
        """
    
    @abstractmethod
    def next(self):
        """
        The strategy's logic, called once per bar before that bar trades.

        Write it as `next(self)` and read indicators relative to the current
        bar (`self.fast[-1]` is the last completed bar), or as `next(self, i)`
        to receive the bar's index. Orders placed here fill on the current bar:
        a market order fills at its open.
        """

    @property
    def i(self) -> int:
        """Index of the current bar. Bars before it have completed."""
        return self._cursor.i

    def price(self, symbol: str) -> float:
        """The open of the current bar: the price a market order placed now fills at."""
        return self.broker._open(symbol, self._cursor.i)

    def I(self, source, *args, **kwargs):
        """
        Declare an indicator, usually in init().

            self.fast = self.I(sma, close, 20)             # a function and its arguments
            self.bands = self.I(bollinger, close, 20, 2)   # several outputs
            self.closes = self.I(close)                    # a raw series

        `source` is a function that returns a series, a DataFrame or a tuple of
        series as long as the data, or such values directly. Price arrays from
        `self.data[symbol]` are handed to the function as pandas Series, and
        other indicators (`Line`) as their full series, so indicators can be
        built on each other.

        Returns a `Line` (or `Lines` for several outputs), read inside next()
        relative to the current bar: `line[-1]` is the last completed bar.
        """
        n = len(self.data)

        def as_input(value):
            if isinstance(value, Line):
                return value.values
            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == n:
                return pd.Series(value)
            if isinstance(value, pd.Series) and len(value) == n:
                return value.reset_index(drop=True)
            return value

        if callable(source):
            name = getattr(source, "__name__", None)
            result = source(*[as_input(a) for a in args],
                            **{k: as_input(v) for k, v in kwargs.items()})
        else:
            if args or kwargs:
                raise TypeError("Arguments are only accepted together with an indicator function.")
            name, result = None, source

        def check(values):
            if len(values) != n:
                raise ValueError(
                    f"Indicator returned {len(values)} values for {n} bars; it must return one per bar.")
            return values

        if isinstance(result, pd.DataFrame):
            return Lines(check(result).reset_index(drop=True), self._cursor)
        if isinstance(result, (tuple, list)) and result and np.ndim(result[0]) == 1:
            return tuple(Line(check(values), self._cursor) for values in result)
        return Line(check(np.asarray(result, dtype=float)), self._cursor, name)

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
        Fetch the current balance (realized cash, excluding open positions)
        """
        return self.broker.cash

    def get_equity(self) -> float:
        """
        Fetch the current balance plus unrealized PnL of open positions
        """
        return self.broker.equity()

    def get_available_margin(self) -> float:
        """
        Fetch the cash that is not locked as margin in open positions
        """
        return self.broker.available_margin()

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

    def run(self, mark_to_market: bool = True) -> pd.Series:
        """
        Executes the strategy over the dataset.

        It handles the iteration over each time period in the data. It calls the 
        user-defined `next` method on each iteration to apply the strategy's logic. 
        Additionally, it updates account balances, and calculates the returns.

        Parameters
        ----------
        mark_to_market : bool
            True (default): returns are based on equity, which is cash plus the
            unrealized PnL of open positions at each bar's close.
            False: returns are based on cash only, which moves when trades close.

        Returns:
            A pandas Series containing the returns, indexed by the corresponding timestamps.
        """
        t = pd.to_datetime(self.data.index)
        equity = np.zeros(len(t))
        realized = np.zeros(len(t))
        r = np.zeros(len(t))
        b = equity if mark_to_market else realized

        for i in range(len(t)):
            self._cursor.i = i
            if self._next_takes_index:
                self.next(i)
            else:
                self.next()
            self.broker.next()

            # Update Balance
            realized[i] = self.broker.cash
            equity[i] = self.broker.equity(i)

            if i > 0:
                # Updates the Returns
                pb = b[i-1] # previous balance
                r[i] = (b[i] - pb) / pb

        self.balances = b.tolist()
        self.equity_curve = equity.tolist()
        self.realized_balances = realized.tolist()
        self.returns = r.tolist()

        return pd.Series(r, index=t)

    def plot(self, returns: pd.Series, save_path=None, title="Simulation"):
        """
        Generates a full performance analysis of the strategy, including trade statistics,
        performance metrics, and visualizations. Outputs are optionally saved to disk.

        Parameters
        ----------
        returns : pd.Series
            A series of periodic strategy returns indexed by datetime.
        save_path : str, optional
            Directory where each chart shown is also saved as a PNG file.
            If None, the charts are only shown.
        title : str
            The strategy's name on the charts and the report.
        """
        trades_list = self.broker.trades
        captial = self.broker.initial_capital

        p = Report(
            returns_series=returns,
            trades_list=trades_list,
            initial_capital=captial,
            benchmark=self.benchmark,
            save_path=save_path,
            title_prefix=title,
            round_trips=self.fetch_round_trips(),
            orders=self.fetch_orders(),
            funding=self.fetch_funding(),
            account_history=self.fetch_account_history(),
            position_history={s: self.fetch_position_history(s) for s in self.data.symbols},
        )
        p.show()

    def fetch_trades(self) -> pd.DataFrame:
        """
        Returns the trade history as a pandas DataFrame.

        Returns:
            A pandas DataFrame where each row represents a trade.
        """
        return pd.DataFrame(self.broker.trades)
    
    def fetch_round_trips(self) -> pd.DataFrame:
        """
        Returns completed trades, one row each: from opening a position until
        it is flat again or flips to the other side.

        Columns include entry and exit time and price, holding time, gross PnL,
        commission, funding, net PnL, return on the entry notional, and `mae`
        and `mfe`: the furthest the price moved against and in favour of the
        position while it was open, as a fraction of the entry price.

        A position still open at the end of the backtest is not included.
        """
        return self.broker.recorder.round_trips_frame()

    def fetch_orders(self) -> pd.DataFrame:
        """
        Returns every order placed, one row each, with its `status`:
        OPEN, FILLED, CANCELLED or REJECTED.
        """
        return pd.DataFrame([order.to_dict() for order in self.broker.orders])

    def fetch_account_history(self) -> pd.DataFrame:
        """
        Returns the account at the close of every bar: cash, equity, margin in
        use, gross exposure (long plus short), net exposure (long minus short)
        and leverage (gross exposure over equity).
        """
        return self.broker.recorder.account_history()

    def fetch_position_history(self, symbol: str) -> pd.DataFrame:
        """
        Returns one symbol's position at the close of every bar: size, entry
        price, close, exposure, margin, liquidation price and unrealized PnL.
        """
        return self.broker.recorder.position_history(symbol)

    def fetch_funding(self) -> pd.DataFrame:
        """
        Returns the funding payments as a pandas DataFrame, one row per payment.
        Negative amounts were paid, positive amounts were received.
        """
        return pd.DataFrame(self.broker.funding_payments)

    def fetch_rejected_orders(self) -> pd.DataFrame:
        """
        Returns orders that were dropped at fill time because the account
        could no longer afford them.
        """
        return pd.DataFrame(self.broker.rejected_orders)

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


