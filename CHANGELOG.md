# Changelog

## Unreleased

### Added

- **Parameter optimizer.** `optimize(MyStrategy, data, params={...}, maximize="sharpe_ratio")` runs a grid or random search over a strategy's parameters, in parallel on macOS and Linux. By default the search only sees the first 70% of the data, and the result shows the winner's metrics on the held-out 30% next to its train metrics, with the number of trials. `result.table` lists every combination and `result.heatmap(x, y)` gives a grid of two parameters.
- **Strategy parameters.** Parameters are class attributes; `MyStrategy(data, params={"fast": 10})` overrides them for one run.
- `Strategy(record=False)` skips recording round trips and per-bar history, for runs where only the returns matter.
- `Strategy` accepts an already prepared `Data` object, so many runs can share one.

### Removed

- **The deprecated indicator and slippage API.** `Strategy.indicator()`, the indicator classes (`Indicator`, `SMA`, `EMA`, `WMA`, `RSI`, `MACD`, `Stochastic`, `BollingerBands`, `ATR`), the `slippage_model=` argument and `Slippage.FixedPercent / Skewed / Zero` are gone, earlier than the 1.0 announced in 0.5.0. Use `self.I(sma, close, 20)` with the functions in `overfitting.indicators`, and `slippage=0.001` or a function `(order, bar) -> price`.

### Changed

- Backtests run faster: reading indicators inside `next()` and building the returns series do less work. The SMA example went from about 90 ms to 68 ms per run, and to 47 ms with `record=False`. Results are unchanged.

## 0.5.0 (2026-11-18)

This release fixes correctness problems in the engine. **Backtest results will change**: Sharpe, drawdown and other return statistics move for any strategy that holds positions across bars, and strategies that were over-sized or liquidated will end with different balances.

It also replaces the analysis package and redesigns how indicators and slippage are written. The old indicator and slippage style keeps working with a deprecation warning.

### Changed

- **Equity is marked to market.** `Strategy.run()` now bases returns on cash plus the unrealized PnL of open positions at each bar's close. Before, the curve only moved when a trade closed, which hid drawdowns on open positions. Pass `run(mark_to_market=False)` for the old realized-only curve. Both are always available as `strategy.equity_curve` and `strategy.realized_balances`.
- **Orders are checked against available margin.** An order that opens or increases a position needs `price × qty / leverage` of free margin (cash minus margin locked in open positions). `InsufficientMargin` is raised at placement; a resting order that is no longer affordable when it fills is dropped and listed in `strategy.fetch_rejected_orders()`. Reducing and closing orders are always accepted. Before, any size was accepted.
- **Liquidation loses the initial margin, no more.** Before, a liquidation charged initial margin plus maintenance margin plus commission, which is more than was posted for the position. `Position.margin` is now the initial margin only. The liquidation price formula is unchanged.
- **The printed report now shows 74 metrics in seven sections** (Overview, Returns, Risk, Drawdown, Sharpe Ratio Reliability, Benchmark, Trades) instead of 23 lines. The raw values are available as `Report.metrics`.
- **Trade statistics now count completed trades, not fills.** A win is a round trip with positive net PnL after commission and funding; before, each closing fill was counted and judged before commission. `Winning Trades`, `Losing Trades`, `Win Rate (%)`, `Avg Return (%)`, `Avg Profit (%)` and `Avg Loss (%)` are replaced by the new Trades section.
- Report lines that changed meaning: `Cumulative Return` (a growth factor) is replaced by `Total Return` (a percentage); `Skew` and `Kurtosis` are now measured on daily returns and kurtosis is reported as excess kurtosis, where they were measured on monthly returns before; `Daily Value At Risk` (two standard deviations below the mean) is replaced by historical value at risk at 95% and 99%.
- Drawdown duration is counted in calendar days (crypto trades every day) and the column is named `Duration (days)`. It was counted in business days before.
- Minimum pandas version is now 2.2.
- **Charts.** `strategy.plot()` shows 13 charts in the same style as before. The daily returns chart is removed. `save_path` now saves each chart as `.png` (it was `.jpg`). `seaborn` and `plotly` are no longer dependencies.

### Added

- **Funding rate.** An optional `funding_rate` column settles funding at the open of each bar where it is non-zero. Payments are listed in `strategy.fetch_funding()`.
- Liquidations are flagged in the trade log (`is_liquidation_order`).
- `Strategy.get_equity()`, `Strategy.get_available_margin()`, `Broker.equity()`, `Broker.unrealized_pnl()`, `Broker.available_margin()`, `Broker.required_margin()`, `Position.unrealized_pnl()`.
- **Indicators are plain functions, read relative to the current bar.** Declare one with `self.I(sma, close, 20)` and read it in `next()` as `self.fast[-1]` (the last completed bar). Bars that have not finished cannot be read, so indicators no longer shift their output. New functions in `overfitting.indicators`: `sma`, `ema`, `wma`, `rsi`, `macd`, `stochastic`, `bollinger`, `atr`, `donchian`, plus `crossover` and `crossunder`.
- `next()` can be written without the bar index. `self.i` gives the index and `self.price(symbol)` the current bar's open.
- **Slippage is a number or a function.** `slippage=0.001` means every fill is 0.1% worse; any function `(order, bar) -> price` is a custom model. `overfitting.slippage` provides `fixed_percent` and `skewed`.
- **Trade, cost, exposure and order analysis.** `overfitting.analysis.metrics` also computes results per completed trade (win rate by side, payoff ratio, profit factor, expectancy, system quality number, holding times, streaks, share of profit from the best trades), trade quality from adverse and favourable excursion, costs (commission, slippage, funding, break-even commission, turnover), exposure (time in market, leverage, margin usage, distance to liquidation) and order outcomes, plus tables by symbol, weekday and hour.
- The printed report gains five sections (Trades, Trade Quality, Costs, Exposure and Margin, Orders) and a per-symbol table.
- **Round trips.** `strategy.fetch_round_trips()` returns one row per completed trade (flat to flat, or to a flip) with entry and exit, holding time, gross PnL, commission, funding, net PnL, and the maximum adverse and favourable excursion.
- **Per-bar history.** `strategy.fetch_account_history()` (cash, equity, margin, gross and net exposure, leverage) and `strategy.fetch_position_history(symbol)` (size, entry price, exposure, margin, liquidation price, unrealized PnL).
- **Order status.** `strategy.fetch_orders()` lists every order placed with a `status` of `OPEN`, `FILLED`, `CANCELLED` or `REJECTED`. Orders rejected at placement are now kept in this log with a `reject_reason`. Each fill records its `slippage` cost. The trade log gains `status`, `slippage`, `cancelled_at` and `reject_reason` columns.
- **Six new charts**: yearly returns, return quantiles (box plot of daily, weekly and monthly returns), rolling beta (with a benchmark), profit per trade, the cost breakdown (commission, slippage, funding), and distance to liquidation. Every chart is also a function in `overfitting.analysis.plots` that returns a matplotlib figure.
- `overfitting.analysis.metrics`: about 70 performance metrics written from their definitions (returns, risk, drawdown, benchmark comparison, rolling statistics, and the statistical significance of the Sharpe ratio). First part of the new analysis package that will replace the pyfolio-derived code.

### Deprecated

These still work and emit a `DeprecationWarning`. They will be removed in 1.0.

- `Strategy.indicator(...)` and the indicator classes (`SMA`, `EMA`, `WMA`, `RSI`, `MACD`, `Stochastic`, `BollingerBands`, `ATR`): use `Strategy.I` with the functions.
- `slippage_model=` and `Slippage.FixedPercent / Skewed / Zero`: use `slippage=`.

### Fixed

- `Broker.order()` crashed when `type` was omitted. It now defaults to a limit order and also accepts an `OrderType`.
- Timezone-aware timestamps are converted to naive ones as documented.
- `Broker.initial_captial` is renamed `initial_capital`. The old name still works and emits a `DeprecationWarning`.
- Benchmark daily returns were shifted by one day against the strategy's when the benchmark had intraday data. This only affected the benchmark lines on the rolling charts before; it would have made the new benchmark metrics wrong.

### Removed

- `overfitting/analysis/graph.py`, which was derived from pyfolio (Apache 2.0). The package no longer contains third-party code.

### Project

- Added a pytest suite (420 tests) and GitHub Actions CI on Python 3.10 to 3.13.
- `requires-python` raised to 3.10, the lowest version the code imports on.
