import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import math
import matplotlib.pyplot as plt

nanmean = np.nanmean
nanstd = np.nanstd
nansum = np.nansum
nanmax = np.nanmax
nanmin = np.nanmin
nanargmax = np.nanargmax
nanargmin = np.nanargmin


APPROX_BDAYS_PER_MONTH = 30
APPROX_BDAYS_PER_YEAR = 365

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quarterly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QTRS_PER_YEAR,
    YEARLY: 1
}


def cum_returns(returns, starting_value=0, out=None):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::

            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

         - Also accepts two dimensional data. In this case, each column is
           cumulated.

    starting_value : float, optional
       The starting returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out


def _flatten(arr):
    return arr if not isinstance(arr, pd.Series) else arr.values

def _adjust_returns(returns, adjustment_factor):
    """
    Returns the returns series adjusted by adjustment_factor. Optimizes for the
    case of adjustment_factor being 0 by returning returns itself, not a copy!

    Parameters
    ----------
    returns : pd.Series or np.ndarray
    adjustment_factor : pd.Series or np.ndarray or float or int

    Returns
    -------
    adjusted_returns : array-like
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor

def annualization_factor(period, annualization):
    """
    Return annualization factor from period entered or if a custom
    value is passed in.

    Parameters
    ----------
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annualization_factor : float
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def downside_risk(returns,
                  required_return=0,
                  period=DAILY,
                  annualization=None,
                  out=None):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    downside_deviation : float or pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details, specifically why using the
    standard deviation of the negative returns is not correct.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    downside_diff = np.clip(
        _adjust_returns(
            np.asanyarray(returns),
            np.asanyarray(required_return),
        ),
        -np.inf,
        0,
    )

    np.square(downside_diff, out=downside_diff)
    nanmean(downside_diff, axis=0, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, np.sqrt(ann_factor), out=out)

    if returns_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out, index=returns.columns)
    return out


def rolling_window(array, length, mutable=False):
    """
    Restride an array of shape

        (X_0, ... X_N)

    into an array of shape

        (length, X_0 - length + 1, ... X_N)

    where each slice at index i along the first axis is equivalent to

        result[i] = array[length * i:length * (i + 1)]

    Parameters
    ----------
    array : np.ndarray
        The base array.
    length : int
        Length of the synthetic first axis to generate.
    mutable : bool, optional
        Return a mutable array? The returned array shares the same memory as
        the input array. This means that writes into the returned array affect
        ``array``. The returned array also uses strides to map the same values
        to multiple indices. Writes to a single index may appear to change many
        values in the returned array.

    Returns
    -------
    out : np.ndarray

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(25).reshape(5, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    >>> rolling_window(a, 2)
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    <BLANKLINE>
           [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]]])
    """
    if not length:
        raise ValueError("Can't have 0-length window")

    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] < length:
        raise IndexError(
            "Can't restride array of shape {shape} with"
            " a window length of {len}".format(
                shape=orig_shape,
                len=length,
            )
        )

    num_windows = (orig_shape[0] - length + 1)
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    out = as_strided(array, new_shape, new_strides)
    out.setflags(write=mutable)
    return out


def _create_unary_vectorized_roll_function(function):
    def unary_vectorized_roll(arr, window, out=None, **kwargs):
        """
        Computes the {human_readable} measure over a rolling window.

        Parameters
        ----------
        arr : array-like
            The array to compute the rolling {human_readable} over.
        window : int
            Size of the rolling window in terms of the periodicity of the data.
        out : array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.
        **kwargs
            Forwarded to :func:`~empyrical.{name}`.

        Returns
        -------
        rolling_{name} : array-like
            The rolling {human_readable}.
        """
        allocated_output = out is None

        if len(arr):
            out = function(
                rolling_window(_flatten(arr), min(len(arr), window)).T,
                out=out,
                **kwargs
            )
        else:
            out = np.empty(0, dtype='float64')

        if allocated_output and isinstance(arr, pd.Series):
            out = pd.Series(out, index=arr.index[-len(out):])

        return out

    unary_vectorized_roll.__doc__ = unary_vectorized_roll.__doc__.format(
        name=function.__name__,
        human_readable=function.__name__.replace('_', ' '),
    )

    return unary_vectorized_roll

def trade_summary(gross_returns, return_percents):
    """
    Computes summary statistics from trade-level realized PnL and return percentages.

    Parameters
    ----------
    gross_returns : list or np.ndarray
        List of realized PnL values (can include open positions, i.e., non-zero even if not closed).
        Typically corresponds to `realized_pnl` from all orders.
    
    return_percents : list or np.ndarray
        List of return percentages (only for closed trades with non-zero PnL).
        Should be filtered to exclude entry-only trades.

    Returns
    -------
    pd.Series
        Summary statistics for trades, including:
         - total trades (closed trades only)
         - win/loss counts
         - win rate
         - gross/net profit
         - average returns for winning/losing trades
    """
    gross_returns = np.asarray(gross_returns)
    return_percents = np.asarray(return_percents)

    # Stats from gross_returns (all trades including entries)
    gross_profit = gross_returns[gross_returns > 0].sum()
    gross_loss = gross_returns[gross_returns <= 0].sum()
    net_profit = gross_profit + gross_loss

    # Stats from return_percents (closed trades only)
    closed_trades = len(return_percents)
    winning_percents = return_percents[return_percents > 0]
    losing_percents = return_percents[return_percents <= 0]

    win_rate = (len(winning_percents) / closed_trades) * 100 if closed_trades else 0
    avg_profit_percent = winning_percents.mean() * 100 if len(winning_percents) else 0
    avg_loss_percent = losing_percents.mean() * 100 if len(losing_percents) else 0
    avg_return_percent = return_percents.mean() * 100 if closed_trades else 0

    return pd.Series({
        "Total Trades": closed_trades,
        "Winning Trades": len(winning_percents),
        "Losing Trades": len(losing_percents),
        "Win Rate (%)": win_rate,
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Net Profit": net_profit,
        "Avg Return (%)": avg_return_percent,
        "Avg Profit (%)": avg_profit_percent,
        "Avg Loss (%)": avg_loss_percent
    })


def sharpe_ratio(returns,
                 risk_free=0,
                 period=DAILY,
                 annualization=None,
                 out=None): # it only supports daily returns

    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant daily risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sharpe_ratio : float
        nan if insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(period, annualization)

    with np.errstate(divide='ignore', invalid='ignore'):
        np.multiply(
            np.divide(
                nanmean(returns_risk_adj, axis=0),
                nanstd(returns_risk_adj, ddof=1, axis=0),
                out=out,
            ),
            np.sqrt(ann_factor),
            out=out,
        )
    if return_1d:
        out = out.item()

    return out


roll_sharpe_ratio = _create_unary_vectorized_roll_function(sharpe_ratio)

def sortino_ratio(returns,
                  required_return=0,
                  period=DAILY,
                  annualization=None,
                  out=None,
                  _downside_risk=None):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    _downside_risk : float, optional
        The downside risk of the given inputs, if known. Will be calculated if
        not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sortino_ratio : float or pd.Series

        depends on input type
        series ==> float
        DataFrame ==> pd.Series

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details.

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    adj_returns = np.asanyarray(_adjust_returns(returns, required_return))

    ann_factor = annualization_factor(period, annualization)

    average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
    annualized_downside_risk = (
        _downside_risk
        if _downside_risk is not None else
        downside_risk(returns, required_return, period, annualization)
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(average_annual_return, annualized_downside_risk, out=out)
        
    if return_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


roll_sortino_ratio = _create_unary_vectorized_roll_function(sortino_ratio)


def get_max_drawdown_underwater(underwater):
    """
    Determines peak, valley, and recovery dates given an 'underwater'
    DataFrame.

    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.

    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.

    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """

    valley = underwater.index[np.argmin(underwater)] # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break

    return drawdowns


def show_worst_drawdown_periods(returns, top=10):
    """
    Places top drawdowns in a table.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """

    df_cum = cum_returns(returns, 1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['Net drawdown in %',
                                         'Peak date',
                                         'Valley date',
                                         'Recovery date',
                                         'Duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'Duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                              .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'Recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Net drawdown in %'] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
    df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
    df_drawdowns['Recovery date'] = pd.to_datetime(
        df_drawdowns['Recovery date'])

    return df_drawdowns


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    aggregated_returns : pd.Series
    """

    def cumulate_returns(x):
        return cum_returns(x).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month/3.))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)

def value_at_risk(returns, period=None, sigma=2.0):
    """
    Get value at risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        Period over which to calculate VaR. Set to 'weekly',
        'monthly', or 'yearly', otherwise defaults to period of
        returns (typically daily).
    sigma : float, optional
        Standard deviations of VaR, default 2.
    """
    if period is not None:
        returns_agg = aggregate_returns(returns, period)
    else:
        returns_agg = returns.copy()

    value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
    return value_at_risk

def monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    # if ax is None:
    #     ax = plt.gca()

    monthly_ret_table = aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    return monthly_ret_table

    # sns.heatmap(
    #     monthly_ret_table.fillna(0) *
    #     100.0,
    #     annot=True,
    #     annot_kws={"size": 9},
    #     alpha=1.0,
    #     center=0.0,
    #     cbar=False,
    #     cmap=plt.cm.RdYlGn,
    #     ax=ax, **kwargs)
    # ax.set_ylabel('Year')
    # ax.set_xlabel('Month')
    # ax.set_title("Monthly returns (%)")
    # return ax

def _rolling_sharpe(returns, rolling_sharpe_window):
    """
    Determines the rolling Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_sharpe_window : int
        Length of rolling window, in days, over which to compute.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    return returns.rolling(rolling_sharpe_window).mean() \
        / returns.rolling(rolling_sharpe_window).std() \
        * np.sqrt(APPROX_BDAYS_PER_YEAR)

def rolling_sharpe(returns, factor_returns=None,
                        rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                        legend_loc='best', ax=None, **kwargs):
    
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for
        which the benchmark rolling Sharpe is computed. Usually
        a benchmark such as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    # if ax is None:
    #     ax = plt.gca()

    # y_axis_formatter = FuncFormatter(utils.two_dec_places)
    # ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        rolling_sharpe_ts_factor = _rolling_sharpe(factor_returns, rolling_window)

        return rolling_sharpe_ts_factor
    else:
        rolling_sharpe_ts = _rolling_sharpe(returns, rolling_window)
        return rolling_sharpe_ts
        
    # ax.set_title('Rolling Sharpe ratio')
    # ax.axhline(
    #     rolling_sharpe_ts.mean(),
    #     color='steelblue',
    #     linestyle='--',
    #     lw=3)
    # ax.axhline(0.0, color='black', linestyle='-', lw=3)

    # ax.set_ylabel('Sharpe ratio')
    # ax.set_xlabel('')
    # if factor_returns is None:
    #     ax.legend(['Sharpe', 'Average'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    # else:
    #     ax.legend(['Sharpe', 'Benchmark Sharpe', 'Average'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)

    # return ax


def _rolling_volatility(returns, rolling_vol_window):
    """
    Determines the rolling volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_vol_window : int
        Length of rolling window, in days, over which to compute.

    Returns
    -------
    pd.Series
        Rolling volatility.
    """

    return returns.rolling(rolling_vol_window).std() \
        * np.sqrt(APPROX_BDAYS_PER_YEAR)

def rolling_volatility(returns, factor_returns=None,
                            rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                            legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling volatility versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the volatility.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    # if ax is None:
    #     ax = plt.gca()

    # y_axis_formatter = FuncFormatter(utils.two_dec_places)
    # ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        rolling_vol_ts_factor = _rolling_volatility(factor_returns, rolling_window)
        return rolling_vol_ts_factor
    else:
        rolling_vol_ts = _rolling_volatility(returns, rolling_window)
        return rolling_vol_ts

    # ax.set_title('Rolling volatility (6-month)')
    # ax.axhline(
    #     rolling_vol_ts.mean(),
    #     color='steelblue',
    #     linestyle='--',
    #     lw=3)

    # ax.axhline(0.0, color='black', linestyle='-', lw=2)

    # ax.set_ylabel('Volatility')
    # ax.set_xlabel('')
    # if factor_returns is None:
    #     ax.legend(['Volatility', 'Average volatility'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    # else:
    #     ax.legend(['Volatility', 'Benchmark volatility', 'Average volatility'],
    #               loc=legend_loc, frameon=True, framealpha=0.5)
    # return ax



def monthly_returns_dist(returns, ax=None, **kwargs):

    """
    Plots a distribution of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    # if ax is None:
    #     ax = plt.gca()

    # x_axis_formatter = FuncFormatter(utils.percentage)
    # ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    # ax.tick_params(axis='x', which='major')

    monthly_ret_table = aggregate_returns(returns, 'monthly')

    return monthly_ret_table

    # ax.hist(
    #     100 * monthly_ret_table,
    #     color='orangered',
    #     alpha=0.80,
    #     bins=20,
    #     **kwargs)

    # ax.axvline(
    #     100 * monthly_ret_table.mean(),
    #     color='gold',
    #     linestyle='--',
    #     lw=4,
    #     alpha=1.0)

    # ax.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
    # ax.legend(['Mean'], frameon=True, framealpha=0.5)
    # ax.set_ylabel('Number of months')
    # ax.set_xlabel('Returns')
    # ax.set_title("Distribution of monthly returns")
    # return ax


"""

sharpe_ratio = graph.timeseries.sharpe_ratio(daily_returns_series, risk_free=0, period='daily')
sortino_ratio = graph.timeseries.sortino_ratio(daily_returns_series, required_return=0, period='daily')
drawdown_table = graph.plotting.show_worst_drawdown_periods(daily_returns_series)
daily_value_at_risk = graph.timeseries.value_at_risk(daily_returns_series, sigma=2, period=None)
monthly_return_heatmap = graph.plotting.plot_monthly_returns_heatmap(daily_returns_series)
rolling_sharpe = graph.plotting.plot_rolling_sharpe(daily_returns_series, factor_returns=None, rolling_window= 180) #rolling_window is days
rolling_volatility = graph.plotting.plot_rolling_volatility(daily_returns_series, factor_returns=None, rolling_window=180)
monthly_returns_dist = graph.plotting.plot_monthly_returns_dist(daily_returns_series)

"""