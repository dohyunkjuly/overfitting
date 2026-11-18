"""
Parameter optimization.

A strategy's parameters are its class attributes:

    class SmaCross(Strategy):
        fast = 20
        slow = 50

        def init(self):
            close = self.data["BTC"].close
            self.fast_ma = self.I(sma, close, self.fast)
            self.slow_ma = self.I(sma, close, self.slow)

`optimize` runs the strategy once for every combination of the values given
and reports the best one:

    result = optimize(
        SmaCross, {"BTC": df},
        params={"fast": range(5, 50, 5), "slow": range(20, 200, 20)},
        maximize="sharpe_ratio",
        constraint=lambda p: p.fast < p.slow,
    )
    print(result)
    result.best_params      # {"fast": 20, "slow": 60}
    result.table            # every combination tried, best first

The best of many combinations always looks better than it is, because some of
it is luck. So by default the search only sees the first 70% of the data
(the train period), and the result shows how the chosen parameters did on the
last 30% (the test period), which played no part in choosing them.
"""
from __future__ import annotations

import itertools
import math
import multiprocessing
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from overfitting.analysis import metrics
from overfitting.data import Data

__all__ = ["optimize", "OptimizeResult", "METRICS"]

# Computed for every combination, on daily returns. Any of them can be maximized.
METRICS = ("total_return", "cagr", "volatility", "sharpe_ratio", "sortino_ratio",
           "max_drawdown", "calmar_ratio", "fills")
MIN_PARALLEL = 16          # fewer combinations than this are not worth starting processes for


class Params(dict):
    """A parameter combination, readable as `p["fast"]` or `p.fast`."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


# ----------------------------------------------------------------- measuring ---

def _daily_groups(index) -> np.ndarray:
    """Positions in `index` where each calendar day starts."""
    days = pd.DatetimeIndex(index).normalize().asi8
    return np.flatnonzero(np.r_[True, days[1:] != days[:-1]])


def _measure(returns: np.ndarray, day_starts: np.ndarray, fills: int) -> dict:
    """The METRICS for one run, from its per-bar returns."""
    if len(returns) == 0:
        return {name: float("nan") for name in METRICS}
    daily = np.multiply.reduceat(1.0 + returns, day_starts) - 1.0
    return {
        "total_return": metrics.total_return(daily),
        "cagr": metrics.cagr(daily),
        "volatility": metrics.volatility(daily),
        "sharpe_ratio": metrics.sharpe_ratio(daily),
        "sortino_ratio": metrics.sortino_ratio(daily),
        "max_drawdown": metrics.max_drawdown(daily),
        "calmar_ratio": metrics.calmar_ratio(daily),
        "fills": fills,
    }


def _objective(maximize, measured: dict) -> float:
    value = maximize(measured) if callable(maximize) else measured[maximize]
    return float(value) if value is not None else float("nan")


# ------------------------------------------------------------------- running ---

# What a worker needs. Set in the parent before the pool starts, so forked
# workers inherit it without anything having to be pickled.
_JOB: dict = {}


def _run_batch(batch: list[dict]) -> list[dict]:
    job = _JOB
    rows = []
    for combination in batch:
        strategy = job["strategy"](data=job["data"], params=combination, record=False,
                                   **job["kwargs"])
        returns = strategy.run().to_numpy()
        measured = _measure(returns, job["day_starts"], len(strategy.broker.trades))
        measured["objective"] = _objective(job["maximize"], measured)
        rows.append({**combination, **measured})
    return rows


def _fork_context():
    """The 'fork' start method where the platform has it, else None."""
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return None


def _run_all(combinations: list[dict], workers: int) -> list[dict]:
    context = _fork_context()
    if workers <= 1 or len(combinations) < MIN_PARALLEL or context is None:
        return _run_batch(combinations)

    workers = min(workers, len(combinations))
    size = max(1, math.ceil(len(combinations) / (workers * 4)))
    batches = [combinations[i:i + size] for i in range(0, len(combinations), size)]
    try:
        with warnings.catch_warnings():
            # Python warns when forking a process that has threads. The workers
            # only run the backtest loop, which is safe to fork into.
            warnings.simplefilter("ignore", DeprecationWarning)
            with context.Pool(workers) as pool:
                results = pool.map(_run_batch, batches)
    except Exception as error:      # a pool that cannot start should not lose the search
        if isinstance(error, (AttributeError, ValueError, TypeError, KeyError)):
            raise                   # an error in the strategy itself: show it
        warnings.warn(f"Parallel run failed ({error!r}); running on one core.", RuntimeWarning)
        return _run_batch(combinations)
    return [row for batch in results for row in batch]


# -------------------------------------------------------------------- result ---

@dataclass
class OptimizeResult:
    """
    best_params   the winning combination
    train         its metrics on the period the search saw
    test          its metrics on the held-out period (None if test_size=0)
    table         every combination tried, best first, with its train metrics
    trials        how many combinations were tried
    strategy      the winning strategy, run over all the data with recording on
    returns       that run's per-bar returns
    """
    strategy_name: str
    best_params: dict
    train: pd.Series
    test: Optional[pd.Series]
    table: pd.DataFrame
    trials: int
    maximize: str
    method: str
    test_size: float
    split_time: Optional[pd.Timestamp]
    strategy: object = field(repr=False, default=None)
    returns: pd.Series = field(repr=False, default=None)

    def heatmap(self, x: str, y: str, metric: str = "objective") -> pd.DataFrame:
        """
        `metric` for every pair of two parameters, as a grid (rows `y`, columns
        `x`). With more than two parameters, each cell is the best value over
        the others.
        """
        for name in (x, y):
            if name not in self.best_params:
                raise KeyError(f"{name!r} is not a parameter. Parameters: {list(self.best_params)}.")
        return self.table.pivot_table(index=y, columns=x, values=metric, aggfunc="max")

    def __str__(self) -> str:
        labels = {
            "total_return": ("Total Return", "{:.2%}"), "cagr": ("CAGR", "{:.2%}"),
            "volatility": ("Volatility", "{:.2%}"), "sharpe_ratio": ("Sharpe Ratio", "{:.3f}"),
            "sortino_ratio": ("Sortino Ratio", "{:.3f}"), "max_drawdown": ("Max Drawdown", "{:.2%}"),
            "calmar_ratio": ("Calmar Ratio", "{:.3f}"), "fills": ("Fills", "{:,.0f}"),
        }

        def cell(series, key, fmt):
            if series is None:
                return ""
            value = series[key]
            return "n/a" if value is None or not np.isfinite(value) else fmt.format(value)

        best = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
        lines = [
            f"Optimization of {self.strategy_name}",
            f"  {'Trials':<18}{self.trials:,} ({self.method})",
            f"  {'Maximized':<18}{self.maximize}",
            f"  {'Best parameters':<18}{best}",
            "",
        ]
        train_share = f"Train ({1 - self.test_size:.0%})"
        test_share = f"Test ({self.test_size:.0%})" if self.test is not None else ""
        lines.append(f"  {'':<18}{train_share:>14}{test_share:>14}")
        for key, (label, fmt) in labels.items():
            lines.append(f"  {label:<18}{cell(self.train, key, fmt):>14}{cell(self.test, key, fmt):>14}")
        lines.append("")
        if self.test is not None:
            lines.append(f"  The search only saw data before {self.split_time:%Y-%m-%d}. The test column is")
            lines.append("  how these parameters did afterwards, on data that played no part in")
            lines.append(f"  choosing them. The train column is the best of {self.trials:,} tries, so expect")
            lines.append("  it to be flattering.")
        else:
            lines.append(f"  No data was held out: these numbers are the best of {self.trials:,} tries on")
            lines.append("  the same data they were chosen on, so expect live results to be worse.")
        return "\n".join(lines)


# ------------------------------------------------------------------ optimize ---

def _combinations(params: dict, constraint, method: str, n: Optional[int], seed) -> list[dict]:
    if not params:
        raise ValueError("Give at least one parameter to optimize, for example params={'fast': range(5, 50)}.")
    values = {}
    for name, candidates in params.items():
        candidates = [candidates] if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Iterable) else list(candidates)
        if not candidates:
            raise ValueError(f"Parameter {name!r} has no values to try.")
        values[name] = candidates

    grid = [dict(zip(values, combo)) for combo in itertools.product(*values.values())]
    if constraint is not None:
        grid = [combo for combo in grid if constraint(Params(combo))]
    if not grid:
        raise ValueError("No parameter combination satisfies the constraint.")

    if method == "grid":
        return grid
    if method == "random":
        if not n or n < 1:
            raise ValueError("method='random' needs n, the number of combinations to try.")
        return grid if n >= len(grid) else random.Random(seed).sample(grid, n)
    raise ValueError("method must be 'grid' or 'random'.")


def optimize(strategy, data: dict, params: dict, *, maximize: Union[str, Callable] = "sharpe_ratio",
             constraint: Optional[Callable] = None, method: str = "grid", n: Optional[int] = None,
             seed: Optional[int] = None, test_size: float = 0.3, workers: Optional[int] = None,
             **strategy_kwargs) -> OptimizeResult:
    """
    Find the parameter values that maximize a metric.

    strategy      the Strategy class (not an instance); its parameters are class attributes
    data          {symbol: DataFrame}, as for the strategy itself
    params        {parameter name: values to try}
    maximize      one of METRICS, or a function of the metrics dict returning a
                  number, e.g. lambda m: m["cagr"] / abs(m["max_drawdown"])
    constraint    function of a combination returning False to skip it,
                  e.g. lambda p: p.fast < p.slow
    method        "grid" tries every combination; "random" tries `n` of them
    n, seed       how many combinations "random" tries, and its random seed
    test_size     share of the data, at the end, kept out of the search and used
                  to check the winner. 0 searches on everything.
    workers       processes to use; default is all cores. 1 runs in this process.
    **strategy_kwargs   passed to the strategy: initial_capital, commission_rate,
                  slippage, and so on

    Returns an OptimizeResult. Combinations whose metric is undefined (for
    example no trades, so no Sharpe ratio) are kept in the table but never win.
    """
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be at least 0 and below 1.")
    if isinstance(maximize, str) and maximize not in METRICS:
        raise ValueError(f"maximize must be one of {METRICS} or a function; got {maximize!r}.")
    unknown = [name for name in params if not hasattr(strategy, name)]
    if unknown:
        raise AttributeError(
            f"{strategy.__name__} has no parameter {unknown[0]!r}. Parameters are class "
            f"attributes, for example `{unknown[0]} = 20` in the class body.")

    combinations = _combinations(params, constraint, method, n, seed)

    bars = len(next(iter(data.values())))
    split = bars - int(round(bars * test_size))
    if split < 2:
        raise ValueError("test_size leaves no data to search on.")
    train_data = Data({symbol: frame.iloc[:split] for symbol, frame in data.items()})
    full_data = Data(data)

    _JOB.update(strategy=strategy, data=train_data, kwargs=strategy_kwargs, maximize=maximize,
                day_starts=_daily_groups(train_data.index))
    try:
        rows = _run_all(combinations, workers if workers is not None else (os.cpu_count() or 1))
    finally:
        _JOB.clear()

    table = pd.DataFrame(rows).sort_values("objective", ascending=False, na_position="last",
                                           kind="stable").reset_index(drop=True)
    if not np.isfinite(table["objective"].iloc[0]):
        name = maximize if isinstance(maximize, str) else "the objective"
        raise ValueError(f"{name} is undefined for every combination (did the strategy trade?).")
    best_params = {name: table[name].iloc[0] for name in params}
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()}

    # Run the winner over everything, with recording, and measure both periods.
    winner = strategy(data=full_data, params=best_params, **strategy_kwargs)
    returns = winner.run()
    values = returns.to_numpy()
    fills = pd.DataFrame(winner.broker.trades)
    split_time = pd.Timestamp(full_data.index[split]) if split < bars else None
    executed = pd.to_datetime(fills["executed_at"]) if len(fills) else pd.Series([], dtype="datetime64[ns]")
    train_fills = int((executed < split_time).sum()) if split_time is not None else len(fills)

    def period(start: int, stop: int, fill_count: int) -> pd.Series:
        measured = _measure(values[start:stop], _daily_groups(full_data.index[start:stop]), fill_count)
        measured["objective"] = _objective(maximize, measured)
        return pd.Series(measured)

    train = period(0, split, train_fills)
    test = period(split, bars, len(fills) - train_fills) if split < bars else None

    return OptimizeResult(
        strategy_name=strategy.__name__, best_params=best_params, train=train, test=test,
        table=table, trials=len(combinations),
        maximize=maximize if isinstance(maximize, str) else getattr(maximize, "__name__", "custom"),
        method=method, test_size=test_size, split_time=split_time, strategy=winner, returns=returns,
    )
