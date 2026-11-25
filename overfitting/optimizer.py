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

By default the search uses all the data. `train` limits it to a window, either
the most recent share of the data (`train=0.6`) or a date range
(`train=("2024-01-01", "2025-01-01")`). The winner is then backtested over all
the data.
"""
from __future__ import annotations

import itertools
import math
import multiprocessing
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

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
    metrics       the winner's metrics over all the data (the full backtest)
    table         every combination tried, best first, with its metrics on the
                  training window
    trials        how many combinations were tried
    train_start, train_end   first and last bar of the training window
    start, end    first and last bar of all the data
    strategy      the winning strategy, run over all the data with recording on
    returns       that run's per-bar returns
    """
    strategy_name: str
    best_params: dict
    metrics: pd.Series
    table: pd.DataFrame
    trials: int
    maximize: str
    method: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
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

        def cell(key, fmt):
            value = self.metrics[key]
            return "n/a" if value is None or not np.isfinite(value) else fmt.format(value)

        best = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
        lines = [
            f"Optimization of {self.strategy_name}",
            f"  {'Trials':<18}{self.trials:,} ({self.method})",
            f"  {'Maximized':<18}{self.maximize}",
            f"  {'Trained on':<18}{self.train_start:%Y-%m-%d} to {self.train_end:%Y-%m-%d}",
            f"  {'Best parameters':<18}{best}",
            "",
            f"  {'Full backtest':<18}{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}",
        ]
        for key, (label, fmt) in labels.items():
            lines.append(f"  {label:<18}{cell(key, fmt)}")
        return "\n".join(lines)


# ------------------------------------------------------------------ optimize ---

def _combinations(params: dict, constraint, method: str, n: int | None, seed) -> list[dict]:
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


def _train_window(index: pd.DatetimeIndex, train) -> tuple[int, int]:
    """Bar positions [first, stop) of the training window."""
    if isinstance(train, (tuple, list)):
        if len(train) != 2:
            raise ValueError('train as a date range is (start, end), e.g. ("2024-01-01", "2025-01-01").')
        bounds = []
        for value in train:
            try:
                stamp = pd.Timestamp(value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"train: cannot read {value!r} as a date.") from error
            if pd.isna(stamp):
                raise ValueError(f"train: cannot read {value!r} as a date.")
            bounds.append(stamp.tz_convert(None) if stamp.tzinfo is not None else stamp)
        if bounds[1] <= bounds[0]:
            raise ValueError(f"train ends ({bounds[1]:%Y-%m-%d}) before it starts "
                             f"({bounds[0]:%Y-%m-%d}); write it as (start, end).")
        first, stop = index.searchsorted(bounds[0]), index.searchsorted(bounds[1])
    elif isinstance(train, (int, float)) and not isinstance(train, bool):
        if not 0 < train <= 1:
            raise ValueError("train must be a share of the data above 0 and up to 1, "
                             'or a (start, end) date range.')
        first, stop = len(index) - int(round(len(index) * train)), len(index)
    else:
        raise ValueError('train must be a share of the data, such as 0.6, or a (start, end) '
                         'date range, such as ("2024-01-01", "2025-01-01").')
    if stop - first < 2:
        raise ValueError(f"The training window holds {max(stop - first, 0)} bars; it needs at least 2.")
    return int(first), int(stop)


def optimize(strategy, data: dict, params: dict, *, maximize: str | Callable = "sharpe_ratio",
             constraint: Callable | None = None, method: str = "grid", n: int | None = None,
             seed: int | None = None, train: float | tuple = 1.0,
             workers: int | None = None, **strategy_kwargs) -> OptimizeResult:
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
    train         the data the search uses: a share of the most recent data
                  (default 1.0, all of it), or a (start, end) date range, end
                  not included. The winner is then backtested over all the data.
    workers       processes to use; default is all cores. 1 runs in this process.
    **strategy_kwargs   passed to the strategy: initial_capital, commission_rate,
                  slippage, and so on

    Returns an OptimizeResult. Combinations whose metric is undefined (for
    example no trades, so no Sharpe ratio) are kept in the table but never win.
    """
    if isinstance(maximize, str) and maximize not in METRICS:
        raise ValueError(f"maximize must be one of {METRICS} or a function; got {maximize!r}.")
    unknown = [name for name in params if not hasattr(strategy, name)]
    if unknown:
        raise AttributeError(
            f"{strategy.__name__} has no parameter {unknown[0]!r}. Parameters are class "
            f"attributes, for example `{unknown[0]} = 20` in the class body.")

    combinations = _combinations(params, constraint, method, n, seed)

    full_data = Data(data)
    first, stop = _train_window(pd.DatetimeIndex(full_data.index), train)
    train_data = Data({symbol: frame.iloc[first:stop] for symbol, frame in data.items()})

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

    # Backtest the winner over all the data, with recording.
    winner = strategy(data=full_data, params=best_params, **strategy_kwargs)
    returns = winner.run()
    measured = _measure(returns.to_numpy(), _daily_groups(full_data.index), len(winner.broker.trades))
    measured["objective"] = _objective(maximize, measured)

    index = full_data.index
    return OptimizeResult(
        strategy_name=strategy.__name__, best_params=best_params, metrics=pd.Series(measured),
        table=table, trials=len(combinations),
        maximize=maximize if isinstance(maximize, str) else getattr(maximize, "__name__", "custom"),
        method=method, train_start=pd.Timestamp(index[first]), train_end=pd.Timestamp(index[stop - 1]),
        start=pd.Timestamp(index[0]), end=pd.Timestamp(index[-1]), strategy=winner, returns=returns,
    )
