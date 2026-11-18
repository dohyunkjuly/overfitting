"""
Performance metrics computed from a series of periodic returns.

Every function takes simple returns (0.01 means +1%) as a pandas Series, one
value per period, and is independent of plotting and of the backtest engine.
`periods_per_year` says how many of those periods make a year: 365 for daily
crypto returns (the default), 252 for daily stock returns, 8760 for hourly.

Functions return NaN when a metric is undefined for the input (too few
observations, zero volatility, no losses, and so on) instead of raising.

The second half of the module works on what the engine recorded instead of on
returns: `trade_summary`, `quality_summary`, `cost_summary`, `exposure_summary`,
`order_summary`, and the `by_symbol`, `by_weekday` and `by_hour` tables.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_YEAR = 365

__all__ = [
    # returns
    "equity_curve", "total_return", "cagr", "annualized_mean", "aggregate_returns",
    "calendar_returns", "trailing_return", "best", "worst", "positive_share",
    "average_positive", "average_negative", "longest_streak",
    # risk
    "volatility", "downside_deviation", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "omega_ratio", "gain_to_pain_ratio", "tail_ratio", "common_sense_ratio",
    "value_at_risk", "expected_shortfall", "skewness", "kurtosis", "jarque_bera",
    "autocorrelation", "kelly_fraction",
    # drawdown
    "drawdown_series", "max_drawdown", "current_drawdown", "drawdown_table",
    "average_drawdown", "longest_drawdown", "average_drawdown_duration",
    "average_recovery_time", "time_underwater", "ulcer_index", "pain_index",
    "recovery_factor", "martin_ratio", "drawdown_count",
    # benchmark
    "beta", "alpha", "correlation", "r_squared", "tracking_error", "information_ratio",
    "excess_return", "up_capture", "down_capture", "outperformance_share",
    # rolling
    "rolling_volatility", "rolling_sharpe", "rolling_sortino", "rolling_beta",
    "rolling_correlation", "rolling_max_drawdown",
    # significance of the Sharpe ratio
    "sharpe_standard_error", "sharpe_confidence_interval", "probabilistic_sharpe_ratio",
    "minimum_track_record_length",
    # everything at once
    "summary",
    # from the engine's recorded tables
    "trade_summary", "quality_summary", "cost_summary", "exposure_summary", "order_summary",
    "by_symbol", "by_weekday", "by_hour",
]


# ------------------------------------------------------------------ helpers ---

def _clean(returns) -> pd.Series:
    """A float Series with missing values dropped."""
    return pd.Series(returns, dtype=float).dropna()


def _aligned(returns, benchmark) -> tuple[pd.Series, pd.Series]:
    """Both series restricted to the periods they share."""
    frame = pd.concat(
        [pd.Series(returns, dtype=float), pd.Series(benchmark, dtype=float)],
        axis=1, join="inner", keys=["r", "b"],
    ).dropna()
    return frame["r"], frame["b"]


def _ratio(numerator: float, denominator: float) -> float:
    """Division that gives NaN instead of raising or returning infinity."""
    if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return float("nan")
    return float(numerator / denominator)


# ------------------------------------------------------------------ returns ---

def equity_curve(returns, start: float = 1.0) -> pd.Series:
    """Growth of `start` invested at the beginning, after each period."""
    return start * (1.0 + _clean(returns)).cumprod()


def total_return(returns) -> float:
    """Compounded return over the whole series."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def cagr(returns, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    growth = (1.0 + r).prod()
    if growth <= 0:
        return float("nan")  # the account was wiped out
    return float(growth ** (periods_per_year / len(r)) - 1.0)


def annualized_mean(returns, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Average period return times the number of periods in a year (no compounding)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(r.mean() * periods_per_year)


def aggregate_returns(returns, freq: str = "D") -> pd.Series:
    """
    Compound returns into longer periods. Needs a DatetimeIndex.

    `freq` is a pandas offset: "D" daily, "W" weekly, "ME" monthly, "YE" yearly.
    Periods with no observations are left out.
    """
    r = _clean(returns)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("aggregate_returns needs returns indexed by time.")
    grouped = (1.0 + r).resample(freq)
    out = grouped.prod() - 1.0
    return out[grouped.count() > 0]


def calendar_returns(returns) -> pd.DataFrame:
    """Monthly returns as a table: one row per year, one column per month (1 to 12)."""
    monthly = aggregate_returns(returns, "ME")
    table = pd.DataFrame({
        "year": monthly.index.year, "month": monthly.index.month, "r": monthly.values,
    }).pivot(index="year", columns="month", values="r")
    return table.reindex(columns=range(1, 13))


def trailing_return(returns, window: str) -> float:
    """Compounded return over the last `window` of time, for example "90D"."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("trailing_return needs returns indexed by time.")
    cutoff = r.index[-1] - pd.Timedelta(window)
    return total_return(r[r.index > cutoff])


def best(returns) -> float:
    """Largest single-period return."""
    r = _clean(returns)
    return float(r.max()) if len(r) else float("nan")


def worst(returns) -> float:
    """Smallest single-period return."""
    r = _clean(returns)
    return float(r.min()) if len(r) else float("nan")


def positive_share(returns) -> float:
    """Fraction of periods with a return above zero."""
    r = _clean(returns)
    return float((r > 0).mean()) if len(r) else float("nan")


def average_positive(returns) -> float:
    """Average return of the periods that gained."""
    r = _clean(returns)
    gains = r[r > 0]
    return float(gains.mean()) if len(gains) else float("nan")


def average_negative(returns) -> float:
    """Average return of the periods that lost."""
    r = _clean(returns)
    losses = r[r < 0]
    return float(losses.mean()) if len(losses) else float("nan")


def longest_streak(returns, positive: bool = True) -> int:
    """Longest run of consecutive gaining (or losing) periods."""
    r = _clean(returns)
    hits = (r > 0) if positive else (r < 0)
    longest = current = 0
    for hit in hits.to_numpy():
        current = current + 1 if hit else 0
        longest = max(longest, current)
    return longest


# --------------------------------------------------------------------- risk ---

def volatility(returns, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Annualized standard deviation of returns."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def downside_deviation(returns, target: float = 0.0,
                       periods_per_year: float = DAYS_PER_YEAR) -> float:
    """
    Annualized deviation below `target` (a per-period return).

    Root mean square of the shortfalls, where periods at or above the target
    count as zero shortfall.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    shortfall = np.minimum(r - target, 0.0)
    return float(math.sqrt((shortfall ** 2).mean()) * math.sqrt(periods_per_year))


def sharpe_ratio(returns, risk_free: float = 0.0,
                 periods_per_year: float = DAYS_PER_YEAR) -> float:
    """
    Annualized Sharpe ratio: mean excess return over its standard deviation.

    `risk_free` is a per-period rate.
    """
    r = _clean(returns) - risk_free
    if len(r) < 2:
        return float("nan")
    return _ratio(r.mean(), r.std(ddof=1)) * math.sqrt(periods_per_year)


def sortino_ratio(returns, target: float = 0.0,
                  periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Annualized Sortino ratio: mean return above `target` over downside deviation."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    downside = downside_deviation(r, target, periods_per_year=1)
    return _ratio(r.mean() - target, downside) * math.sqrt(periods_per_year)


def calmar_ratio(returns, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """CAGR divided by the size of the maximum drawdown."""
    return _ratio(cagr(returns, periods_per_year), abs(max_drawdown(returns)))


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Sum of gains above `threshold` divided by sum of shortfalls below it."""
    r = _clean(returns) - threshold
    if r.empty:
        return float("nan")
    return _ratio(r[r > 0].sum(), -r[r < 0].sum())


def gain_to_pain_ratio(returns) -> float:
    """Sum of all returns divided by the sum of the losing returns' sizes."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return _ratio(r.sum(), -r[r < 0].sum())


def tail_ratio(returns) -> float:
    """Size of the 95th percentile return over the size of the 5th percentile return."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return _ratio(abs(r.quantile(0.95)), abs(r.quantile(0.05)))


def common_sense_ratio(returns) -> float:
    """Tail ratio times the ratio of summed gains to summed losses."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return tail_ratio(r) * _ratio(r[r > 0].sum(), -r[r < 0].sum())


def value_at_risk(returns, level: float = 0.95, method: str = "historical") -> float:
    """
    The return that is only undercut in (1 - level) of periods. Usually negative.

    method "historical" reads it from the observed returns; "parametric"
    assumes a normal distribution with the observed mean and deviation.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    if method == "historical":
        return float(r.quantile(1.0 - level))
    if method == "parametric":
        return float(r.mean() + stats.norm.ppf(1.0 - level) * r.std(ddof=1))
    raise ValueError("method must be 'historical' or 'parametric'.")


def expected_shortfall(returns, level: float = 0.95, method: str = "historical") -> float:
    """
    Average return in the worst (1 - level) of periods (conditional value at risk).
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    if method == "historical":
        tail = r[r <= r.quantile(1.0 - level)]
        return float(tail.mean())
    if method == "parametric":
        z = stats.norm.ppf(1.0 - level)
        return float(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1.0 - level))
    raise ValueError("method must be 'historical' or 'parametric'.")


def skewness(returns) -> float:
    """Asymmetry of the return distribution. Negative means a longer left tail."""
    r = _clean(returns)
    if len(r) < 3 or r.std(ddof=0) == 0:
        return float("nan")
    return float(stats.skew(r, bias=False))


def kurtosis(returns) -> float:
    """Excess kurtosis: 0 for a normal distribution, positive for fat tails."""
    r = _clean(returns)
    if len(r) < 4 or r.std(ddof=0) == 0:
        return float("nan")
    return float(stats.kurtosis(r, fisher=True, bias=False))


def jarque_bera(returns) -> tuple[float, float]:
    """
    Jarque-Bera test for normality: (statistic, p-value).
    A small p-value means the returns are unlikely to be normally distributed.
    """
    r = _clean(returns)
    if len(r) < 4 or r.std(ddof=0) == 0:
        return float("nan"), float("nan")
    result = stats.jarque_bera(r)
    return float(result.statistic), float(result.pvalue)


def autocorrelation(returns, lag: int = 1) -> float:
    """Correlation of returns with themselves `lag` periods earlier."""
    r = _clean(returns)
    if len(r) <= lag + 1 or r.std(ddof=0) == 0:
        return float("nan")
    return float(r.autocorr(lag))


def kelly_fraction(returns) -> float:
    """
    Growth-optimal leverage for these returns: mean divided by variance.
    Treat it as an upper bound; it assumes the future looks like the sample.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return _ratio(r.mean(), r.var(ddof=1))


# ----------------------------------------------------------------- drawdown ---

def drawdown_series(returns) -> pd.Series:
    """
    Decline from the highest equity reached so far, per period (0 at a new high,
    -0.25 when a quarter below the peak). The starting value counts as a peak.
    """
    equity = equity_curve(returns)
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def max_drawdown(returns) -> float:
    """Deepest decline from a peak, as a negative fraction."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def current_drawdown(returns) -> float:
    """Decline from the peak at the end of the series."""
    dd = drawdown_series(returns)
    return float(dd.iloc[-1]) if len(dd) else float("nan")


def drawdown_table(returns, top: int | None = None) -> pd.DataFrame:
    """
    One row per drawdown, deepest first.

    Columns:
        depth          deepest decline (negative fraction)
        peak           last period at the high before the decline (the first
                       period of the series if it starts with a loss)
        valley         period of the deepest point
        recovery       first period back at the high, NaT if not recovered
        duration       periods from the first losing period until recovery,
                       or until the end if not recovered
        to_valley      periods from the first losing period to the valley
        to_recovery    periods from the valley to recovery, NaN if not recovered
    """
    dd = drawdown_series(returns)
    columns = ["depth", "peak", "valley", "recovery", "duration", "to_valley", "to_recovery"]
    if dd.empty:
        return pd.DataFrame(columns=columns)

    underwater = (dd < 0).to_numpy()
    values = dd.to_numpy()
    index = dd.index
    rows = []
    i, n = 0, len(dd)
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        start = i
        while i < n and underwater[i]:
            i += 1
        end = i  # first period back at the high, or n if never recovered
        valley = start + int(np.argmin(values[start:end]))
        recovered = end < n
        rows.append({
            "depth": float(values[valley]),
            "peak": index[start - 1] if start > 0 else index[start],
            "valley": index[valley],
            "recovery": index[end] if recovered else pd.NaT,
            "duration": end - start,
            "to_valley": valley - start + 1,
            "to_recovery": end - valley if recovered else np.nan,
        })

    table = pd.DataFrame(rows, columns=columns).sort_values("depth").reset_index(drop=True)
    return table.head(top) if top else table


def average_drawdown(returns) -> float:
    """Average depth of all drawdowns."""
    table = drawdown_table(returns)
    return float(table["depth"].mean()) if len(table) else 0.0


def longest_drawdown(returns) -> int:
    """Longest time spent below a previous high, in periods."""
    table = drawdown_table(returns)
    return int(table["duration"].max()) if len(table) else 0


def average_drawdown_duration(returns) -> float:
    """Average time spent in a drawdown, in periods."""
    table = drawdown_table(returns)
    return float(table["duration"].mean()) if len(table) else 0.0


def average_recovery_time(returns) -> float:
    """Average periods from a drawdown's deepest point back to the high (recovered ones only)."""
    recovered = drawdown_table(returns)["to_recovery"].dropna()
    return float(recovered.mean()) if len(recovered) else float("nan")


def time_underwater(returns) -> float:
    """Fraction of periods spent below a previous high."""
    dd = drawdown_series(returns)
    return float((dd < 0).mean()) if len(dd) else float("nan")


def ulcer_index(returns) -> float:
    """Root mean square of drawdowns. Punishes deep and long declines."""
    dd = drawdown_series(returns)
    return float(math.sqrt((dd ** 2).mean())) if len(dd) else float("nan")


def pain_index(returns) -> float:
    """Average drawdown depth over all periods, as a positive fraction."""
    dd = drawdown_series(returns)
    return float(-dd.mean()) if len(dd) else float("nan")


def recovery_factor(returns) -> float:
    """Total return divided by the size of the maximum drawdown."""
    return _ratio(total_return(returns), abs(max_drawdown(returns)))


def martin_ratio(returns, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """CAGR divided by the ulcer index."""
    return _ratio(cagr(returns, periods_per_year), ulcer_index(returns))


def drawdown_count(returns, deeper_than: float = 0.0) -> int:
    """Number of drawdowns deeper than `deeper_than` (0.1 means worse than -10%)."""
    table = drawdown_table(returns)
    return int((table["depth"] < -abs(deeper_than)).sum())


# ---------------------------------------------------------------- benchmark ---

def beta(returns, benchmark) -> float:
    """Sensitivity to the benchmark: slope of strategy returns on benchmark returns."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    return _ratio(r.cov(b), b.var(ddof=1))


def alpha(returns, benchmark, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Annualized return not explained by the benchmark (regression intercept)."""
    r, b = _aligned(returns, benchmark)
    slope = beta(r, b)
    if not np.isfinite(slope):
        return float("nan")
    return float((r.mean() - slope * b.mean()) * periods_per_year)


def correlation(returns, benchmark) -> float:
    """Correlation between strategy and benchmark returns."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 2 or r.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return float("nan")
    return float(r.corr(b))


def r_squared(returns, benchmark) -> float:
    """Share of the strategy's variance explained by the benchmark."""
    c = correlation(returns, benchmark)
    return float(c ** 2) if np.isfinite(c) else float("nan")


def tracking_error(returns, benchmark, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Annualized standard deviation of the difference to the benchmark."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    return float((r - b).std(ddof=1) * math.sqrt(periods_per_year))


def information_ratio(returns, benchmark, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Annualized average outperformance divided by tracking error."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    active = r - b
    return _ratio(active.mean(), active.std(ddof=1)) * math.sqrt(periods_per_year)


def excess_return(returns, benchmark, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Strategy CAGR minus benchmark CAGR over the shared periods."""
    r, b = _aligned(returns, benchmark)
    return cagr(r, periods_per_year) - cagr(b, periods_per_year)


def _capture(returns, benchmark, up: bool) -> float:
    r, b = _aligned(returns, benchmark)
    mask = (b > 0) if up else (b < 0)
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    # Compare average (geometric) per-period returns. Comparing compounded
    # totals instead would shrink toward zero as the sample grows.
    strategy_growth, benchmark_growth = (1.0 + r[mask]).prod(), (1.0 + b[mask]).prod()
    if strategy_growth <= 0 or benchmark_growth <= 0:
        return float("nan")
    return _ratio(strategy_growth ** (1.0 / count) - 1.0, benchmark_growth ** (1.0 / count) - 1.0)


def up_capture(returns, benchmark) -> float:
    """Average strategy return over average benchmark return, in the periods the benchmark rose."""
    return _capture(returns, benchmark, up=True)


def down_capture(returns, benchmark) -> float:
    """Average strategy return over average benchmark return, in the periods the benchmark fell."""
    return _capture(returns, benchmark, up=False)


def outperformance_share(returns, benchmark) -> float:
    """Fraction of periods in which the strategy beat the benchmark."""
    r, b = _aligned(returns, benchmark)
    return float((r > b).mean()) if len(r) else float("nan")


# ------------------------------------------------------------------ rolling ---

def rolling_volatility(returns, window: int,
                       periods_per_year: float = DAYS_PER_YEAR) -> pd.Series:
    """Annualized volatility over a moving window of `window` periods."""
    return _clean(returns).rolling(window).std(ddof=1) * math.sqrt(periods_per_year)


def rolling_sharpe(returns, window: int, risk_free: float = 0.0,
                   periods_per_year: float = DAYS_PER_YEAR) -> pd.Series:
    """Annualized Sharpe ratio over a moving window."""
    r = _clean(returns) - risk_free
    deviation = r.rolling(window).std(ddof=1).replace(0.0, np.nan)
    return r.rolling(window).mean() / deviation * math.sqrt(periods_per_year)


def rolling_sortino(returns, window: int, target: float = 0.0,
                    periods_per_year: float = DAYS_PER_YEAR) -> pd.Series:
    """Annualized Sortino ratio over a moving window."""
    r = _clean(returns)
    shortfall_sq = np.minimum(r - target, 0.0) ** 2
    downside = np.sqrt(shortfall_sq.rolling(window).mean()).replace(0.0, np.nan)
    return (r.rolling(window).mean() - target) / downside * math.sqrt(periods_per_year)


def rolling_beta(returns, benchmark, window: int) -> pd.Series:
    """Beta to the benchmark over a moving window."""
    r, b = _aligned(returns, benchmark)
    variance = b.rolling(window).var(ddof=1).replace(0.0, np.nan)
    return r.rolling(window).cov(b) / variance


def rolling_correlation(returns, benchmark, window: int) -> pd.Series:
    """Correlation with the benchmark over a moving window."""
    r, b = _aligned(returns, benchmark)
    return r.rolling(window).corr(b)


def rolling_max_drawdown(returns, window: int) -> pd.Series:
    """Deepest drawdown that happened inside each moving window."""
    r = _clean(returns)
    return r.rolling(window).apply(lambda w: max_drawdown(pd.Series(w)), raw=True)


# ----------------------------------------- significance of the Sharpe ratio ---
#
# From Bailey and Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (2012).
# The estimated Sharpe ratio of n returns has standard error
#
#     sqrt((1 - skew * SR + (kurt - 1) / 4 * SR^2) / (n - 1))
#
# where SR is the per-period (not annualized) Sharpe ratio and kurt is the
# plain kurtosis (3 for a normal distribution).

def _sharpe_inputs(returns) -> tuple[float, float, int]:
    """Per-period Sharpe ratio, its variance numerator, and the sample size."""
    r = _clean(returns)
    n = len(r)
    if n < 4 or r.std(ddof=1) == 0:
        return float("nan"), float("nan"), n
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r, bias=True)
    kurt = stats.kurtosis(r, fisher=False, bias=True)
    return float(sr), float(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2), n


def sharpe_standard_error(returns, periods_per_year: float = DAYS_PER_YEAR) -> float:
    """Standard error of the annualized Sharpe ratio, allowing for skew and fat tails."""
    sr, spread, n = _sharpe_inputs(returns)
    if not np.isfinite(sr) or spread <= 0:
        return float("nan")
    return float(math.sqrt(spread / (n - 1)) * math.sqrt(periods_per_year))


def sharpe_confidence_interval(returns, confidence: float = 0.95,
                               periods_per_year: float = DAYS_PER_YEAR) -> tuple[float, float]:
    """Range that should contain the true annualized Sharpe ratio with the given confidence."""
    error = sharpe_standard_error(returns, periods_per_year)
    centre = sharpe_ratio(returns, periods_per_year=periods_per_year)
    if not np.isfinite(error) or not np.isfinite(centre):
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    return float(centre - z * error), float(centre + z * error)


def probabilistic_sharpe_ratio(returns, threshold: float = 0.0,
                               periods_per_year: float = DAYS_PER_YEAR) -> float:
    """
    Probability that the true Sharpe ratio is above `threshold` (annualized),
    given the length, skew and tails of the track record.
    """
    sr, spread, n = _sharpe_inputs(returns)
    if not np.isfinite(sr) or spread <= 0:
        return float("nan")
    target = threshold / math.sqrt(periods_per_year)
    return float(stats.norm.cdf((sr - target) * math.sqrt(n - 1) / math.sqrt(spread)))


def minimum_track_record_length(returns, threshold: float = 0.0, confidence: float = 0.95,
                                periods_per_year: float = DAYS_PER_YEAR) -> float:
    """
    Number of periods needed before the observed Sharpe ratio is above
    `threshold` (annualized) with the given confidence. NaN if the observed
    ratio is not above the threshold at all.
    """
    sr, spread, _ = _sharpe_inputs(returns)
    target = threshold / math.sqrt(periods_per_year)
    if not np.isfinite(sr) or spread <= 0 or sr <= target:
        return float("nan")
    z = stats.norm.ppf(confidence)
    return float(1.0 + spread * (z / (sr - target)) ** 2)


# ------------------------------------------------------------------ summary ---

def summary(returns, benchmark=None, periods_per_year: float = DAYS_PER_YEAR) -> pd.Series:
    """Every scalar metric in this module, as one labelled Series."""
    r = _clean(returns)
    ppy = periods_per_year
    low, high = sharpe_confidence_interval(r, periods_per_year=ppy)
    jb_stat, jb_p = jarque_bera(r)

    out = {
        # returns
        "periods": len(r),
        "total_return": total_return(r),
        "cagr": cagr(r, ppy),
        "annualized_mean": annualized_mean(r, ppy),
        "best_period": best(r),
        "worst_period": worst(r),
        "positive_share": positive_share(r),
        "average_positive": average_positive(r),
        "average_negative": average_negative(r),
        "longest_winning_streak": longest_streak(r, positive=True),
        "longest_losing_streak": longest_streak(r, positive=False),
        # risk
        "volatility": volatility(r, ppy),
        "downside_deviation": downside_deviation(r, periods_per_year=ppy),
        "sharpe_ratio": sharpe_ratio(r, periods_per_year=ppy),
        "sortino_ratio": sortino_ratio(r, periods_per_year=ppy),
        "calmar_ratio": calmar_ratio(r, ppy),
        "omega_ratio": omega_ratio(r),
        "gain_to_pain_ratio": gain_to_pain_ratio(r),
        "tail_ratio": tail_ratio(r),
        "common_sense_ratio": common_sense_ratio(r),
        "value_at_risk_95": value_at_risk(r, 0.95),
        "value_at_risk_99": value_at_risk(r, 0.99),
        "expected_shortfall_95": expected_shortfall(r, 0.95),
        "expected_shortfall_99": expected_shortfall(r, 0.99),
        "skewness": skewness(r),
        "kurtosis": kurtosis(r),
        "jarque_bera_statistic": jb_stat,
        "jarque_bera_p_value": jb_p,
        "autocorrelation_lag_1": autocorrelation(r, 1),
        "kelly_fraction": kelly_fraction(r),
        # drawdown
        "max_drawdown": max_drawdown(r),
        "current_drawdown": current_drawdown(r),
        "average_drawdown": average_drawdown(r),
        "longest_drawdown": longest_drawdown(r),
        "average_drawdown_duration": average_drawdown_duration(r),
        "average_recovery_time": average_recovery_time(r),
        "time_underwater": time_underwater(r),
        "ulcer_index": ulcer_index(r),
        "pain_index": pain_index(r),
        "recovery_factor": recovery_factor(r),
        "martin_ratio": martin_ratio(r, ppy),
        "drawdowns_over_5pct": drawdown_count(r, 0.05),
        "drawdowns_over_10pct": drawdown_count(r, 0.10),
        "drawdowns_over_20pct": drawdown_count(r, 0.20),
        # significance
        "sharpe_standard_error": sharpe_standard_error(r, ppy),
        "sharpe_ci_low": low,
        "sharpe_ci_high": high,
        "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(r, periods_per_year=ppy),
        "minimum_track_record_length": minimum_track_record_length(r, periods_per_year=ppy),
    }

    if benchmark is not None:
        b = benchmark
        out.update({
            "alpha": alpha(r, b, ppy),
            "beta": beta(r, b),
            "correlation": correlation(r, b),
            "r_squared": r_squared(r, b),
            "tracking_error": tracking_error(r, b, ppy),
            "information_ratio": information_ratio(r, b, ppy),
            "excess_return": excess_return(r, b, ppy),
            "up_capture": up_capture(r, b),
            "down_capture": down_capture(r, b),
            "outperformance_share": outperformance_share(r, b),
        })

    return pd.Series(out, dtype=object)


# =========================================================================
# Metrics from what the engine recorded
# =========================================================================
#
# The functions below take the DataFrames returned by the Strategy.fetch_*
# methods: round trips, fills, funding payments, orders, and the per-bar
# account and position history.

NAN = float("nan")


def _mean(values) -> float:
    values = pd.Series(values, dtype=float).dropna()
    return float(values.mean()) if len(values) else NAN


def _longest_run(flags) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _empty(frame) -> bool:
    return frame is None or len(frame) == 0


# ------------------------------------------------------------------- trades ---

def trade_summary(round_trips: pd.DataFrame) -> pd.Series:
    """
    Results per completed trade. A win is a round trip with positive net PnL
    (after commission and funding).

    payoff_ratio            average win over average loss
    profit_factor           total won over total lost
    expectancy              average net PnL per trade
    expectancy_r            average net PnL in units of the average loss
    system_quality_number   sqrt(trades) * mean(R) / std(R), R = PnL over average loss
    profit_from_best_*      share of net profit that came from the best trades
    """
    rt = round_trips
    if _empty(rt):
        return pd.Series({"total_trades": 0}, dtype=object)

    net = rt["net_pnl"].astype(float)
    wins, losses = net[net > 0], net[net < 0]
    longs, shorts = rt[rt["side"] == "LONG"], rt[rt["side"] == "SHORT"]
    average_loss = abs(float(losses.mean())) if len(losses) else NAN
    r_multiples = net / average_loss if average_loss else pd.Series(dtype=float)
    bars = rt["bars_held"].astype(float)
    total = float(net.sum())
    ranked = net.sort_values(ascending=False)

    months = max((rt["exit_time"].max() - rt["entry_time"].min()).days / 30.4375, 1e-9)

    return pd.Series({
        "total_trades": len(rt),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "win_rate": float((net > 0).mean()),
        "win_rate_long": float((longs["net_pnl"] > 0).mean()) if len(longs) else NAN,
        "win_rate_short": float((shorts["net_pnl"] > 0).mean()) if len(shorts) else NAN,
        "average_win": _mean(wins),
        "average_loss": _mean(losses),
        "largest_win": float(net.max()),
        "largest_loss": float(net.min()),
        "payoff_ratio": _ratio(_mean(wins), average_loss),
        "profit_factor": _ratio(float(wins.sum()), -float(losses.sum())),
        "expectancy": float(net.mean()),
        "average_return": _mean(rt["return_pct"]),
        "average_winning_return": _mean(rt.loc[net > 0, "return_pct"]),
        "average_losing_return": _mean(rt.loc[net < 0, "return_pct"]),
        "expectancy_r": _mean(r_multiples),
        "system_quality_number": (
            _ratio(float(r_multiples.mean()), float(r_multiples.std(ddof=1))) * math.sqrt(len(rt))
            if len(r_multiples) > 1 else NAN),
        "average_bars_held": float(bars.mean()),
        "median_bars_held": float(bars.median()),
        "longest_bars_held": float(bars.max()),
        "average_bars_held_winners": _mean(bars[net > 0]),
        "average_bars_held_losers": _mean(bars[net < 0]),
        "longest_winning_streak": _longest_run(net > 0),
        "longest_losing_streak": _longest_run(net < 0),
        "trades_per_month": len(rt) / months,
        "profit_from_best_trade": _ratio(float(ranked.iloc[0]), total) if total > 0 else NAN,
        "profit_from_best_5_trades": _ratio(float(ranked.head(5).sum()), total) if total > 0 else NAN,
        "net_profit_without_best_5": total - float(ranked.head(5).sum()),
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "net_profit": total,
    }, dtype=object)


def quality_summary(round_trips: pd.DataFrame) -> pd.Series:
    """
    How well trades were entered and exited, from the best and worst price
    each one saw while open.

    average_adverse_excursion      average worst move against the trade
    average_favourable_excursion   average best move in its favour
    edge_ratio                     favourable over adverse; above 1 means
                                   trades run further for you than against you
    profit_captured                price return actually kept, as a share of
                                   the best return that was available
    average_profit_given_back      average gap between a trade's best point and its exit
    """
    rt = round_trips
    if _empty(rt):
        return pd.Series({"liquidations": 0}, dtype=object)

    side = np.where(rt["side"] == "LONG", 1.0, -1.0)
    price_return = side * (rt["exit_price"] - rt["entry_price"]) / rt["entry_price"]
    mae, mfe = rt["mae"].astype(float), rt["mfe"].astype(float)

    return pd.Series({
        "average_adverse_excursion": float(mae.mean()),
        "average_favourable_excursion": float(mfe.mean()),
        "edge_ratio": _ratio(float(mfe.mean()), abs(float(mae.mean()))),
        "profit_captured": _ratio(float(price_return.sum()), float(mfe.sum())),
        "average_profit_given_back": float((mfe - price_return).mean()),
        "liquidations": int(rt["is_liquidation"].sum()),
    }, dtype=object)


# -------------------------------------------------------------------- costs ---

def cost_summary(fills: pd.DataFrame, funding: pd.DataFrame | None = None,
                 equity: pd.Series | None = None, years: float | None = None) -> pd.Series:
    """
    What trading cost, from the fill log (`fetch_trades`) and funding payments.

    profit_before_costs     PnL if every fill had happened at its pre-slippage
                            price with no commission and no funding
    total_costs             commission + slippage + net funding paid
    cost_share              total costs as a share of profit before costs
    break_even_commission   how many times larger commission could be before
                            net profit reaches zero
    turnover                traded value over average equity, per year
    """
    if _empty(fills):
        return pd.Series({"total_commission": 0.0, "total_slippage": 0.0}, dtype=object)

    commission = float(fills["commission"].sum())
    slippage = float(fills["slippage"].sum()) if "slippage" in fills else 0.0
    paid = received = 0.0
    if not _empty(funding):
        amounts = funding["amount"].astype(float)
        paid, received = -float(amounts[amounts < 0].sum()), float(amounts[amounts > 0].sum())

    realized = float(fills["pnl"].sum())              # after slippage, before commission
    net = realized - commission - paid + received
    costs = commission + slippage + paid - received
    before_costs = net + costs
    traded = float((fills["qty"].abs() * fills["executed_price"].abs()).sum())

    turnover = NAN
    if equity is not None and len(equity) and years:
        turnover = _ratio(traded, float(pd.Series(equity).mean()) * years)

    return pd.Series({
        "total_commission": commission,
        "total_slippage": slippage,
        "funding_paid": paid,
        "funding_received": received,
        "total_costs": costs,
        "profit_before_costs": before_costs,
        "cost_share": _ratio(costs, before_costs) if before_costs > 0 else NAN,
        "average_cost_per_fill": costs / len(fills),
        "break_even_commission": _ratio(net + commission, commission) if net > 0 else NAN,
        "traded_value": traded,
        "turnover": turnover,
    }, dtype=object)


# ----------------------------------------------------------------- exposure ---

def exposure_summary(account: pd.DataFrame, positions: dict[str, pd.DataFrame] | None = None,
                     round_trips: pd.DataFrame | None = None) -> pd.Series:
    """
    How much of the account was at risk, from the per-bar history.

    time_in_market               share of bars with an open position
    average_leverage             gross exposure over equity, averaged over all bars
    average_leverage_in_market   the same, over bars with a position only
    margin usage                 margin locked over equity
    distance to liquidation      how far the close was from the liquidation
                                 price, as a share of the close (smaller is closer)
    """
    if _empty(account):
        return pd.Series(dtype=object)

    equity = account["equity"].replace(0.0, np.nan)
    leverage = account["leverage"]
    in_market = account["gross_exposure"] > 0
    margin_usage = account["margin"] / equity

    distances = []
    for history in (positions or {}).values():
        held = history[history["qty"] != 0]
        if len(held):
            side = np.sign(held["qty"])
            distances.append(side * (held["close"] - held["liquidation_price"]) / held["close"])
    distance = pd.concat(distances) if distances else pd.Series(dtype=float)

    liquidation_loss = 0.0
    if not _empty(round_trips):
        liquidation_loss = float(round_trips.loc[round_trips["is_liquidation"], "net_pnl"].sum())

    return pd.Series({
        "time_in_market": float(in_market.mean()),
        "average_leverage": _mean(leverage),
        "average_leverage_in_market": _mean(leverage[in_market]),
        "maximum_leverage": float(leverage.max()) if leverage.notna().any() else NAN,
        "average_net_exposure": _mean(account["net_exposure"] / equity),
        "average_margin_usage": _mean(margin_usage),
        "peak_margin_usage": float(margin_usage.max()) if margin_usage.notna().any() else NAN,
        "closest_distance_to_liquidation": float(distance.min()) if len(distance) else NAN,
        "average_distance_to_liquidation": _mean(distance),
        "liquidation_loss": liquidation_loss,
    }, dtype=object)


# ------------------------------------------------------------------- orders ---

def order_summary(orders: pd.DataFrame) -> pd.Series:
    """
    What became of the orders placed, from `fetch_orders`.

    fill_rate          filled over (filled + cancelled + rejected)
    average_slippage   slippage cost over traded value, per fill
    """
    if _empty(orders):
        return pd.Series({"orders_placed": 0}, dtype=object)

    status = orders["status"]
    filled = orders[status == "FILLED"]
    counts = {name: int((status == name).sum()) for name in ("FILLED", "CANCELLED", "REJECTED", "OPEN")}
    finished = counts["FILLED"] + counts["CANCELLED"] + counts["REJECTED"]
    traded = float((filled["qty"].abs() * filled["executed_price"].abs()).sum())

    return pd.Series({
        "orders_placed": len(orders),
        "orders_filled": counts["FILLED"],
        "orders_cancelled": counts["CANCELLED"],
        "orders_rejected": counts["REJECTED"],
        "orders_open": counts["OPEN"],
        "fill_rate": _ratio(counts["FILLED"], finished),
        "average_slippage": _ratio(float(filled["slippage"].sum()), traded),
    }, dtype=object)


# ------------------------------------------------------------------- tables ---

def _group_table(round_trips: pd.DataFrame, keys) -> pd.DataFrame:
    net = round_trips["net_pnl"].astype(float)
    grouped = net.groupby(keys)
    total = float(net.sum())
    table = pd.DataFrame({
        "trades": grouped.size(),
        "win_rate": grouped.apply(lambda v: float((v > 0).mean())),
        "net_pnl": grouped.sum(),
        "average_pnl": grouped.mean(),
        "profit_factor": grouped.apply(
            lambda v: _ratio(float(v[v > 0].sum()), -float(v[v < 0].sum()))),
    })
    table["share_of_profit"] = table["net_pnl"] / total if total else NAN
    return table


def by_symbol(round_trips: pd.DataFrame, positions: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    Results per symbol: trades, win rate, net PnL, average PnL, profit factor,
    share of total net profit, and (given position history) share of bars held.
    """
    if _empty(round_trips):
        return pd.DataFrame(columns=["trades", "win_rate", "net_pnl", "average_pnl",
                                     "profit_factor", "share_of_profit", "time_held"])
    table = _group_table(round_trips, round_trips["symbol"])
    table["time_held"] = [
        float((positions[s]["qty"] != 0).mean()) if positions and s in positions else NAN
        for s in table.index
    ]
    table.index.name = "symbol"
    return table.sort_values("net_pnl", ascending=False)


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def by_weekday(round_trips: pd.DataFrame) -> pd.DataFrame:
    """Results by the weekday the trade was entered."""
    if _empty(round_trips):
        return pd.DataFrame()
    table = _group_table(round_trips, round_trips["entry_time"].dt.day_of_week.to_numpy())
    table.index = [WEEKDAYS[d] for d in table.index]
    table.index.name = "weekday"
    return table


def by_hour(round_trips: pd.DataFrame) -> pd.DataFrame:
    """Results by the hour of day the trade was entered."""
    if _empty(round_trips):
        return pd.DataFrame()
    table = _group_table(round_trips, round_trips["entry_time"].dt.hour.to_numpy())
    table.index.name = "hour"
    return table
