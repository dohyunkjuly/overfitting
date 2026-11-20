# Changelog

## Unreleased

### Changed

- **Faster backtests with many symbols.** Each bar now skips symbols with no position and no open orders. A 50-symbol backtest where 3 symbols trade runs about 3x faster; single-symbol backtests about 1.2x. Results are unchanged.
- **Faster fills and funding.** Recording the time of a fill, cancel or funding payment is now about 20x cheaper. Backtests run 1.2x to 1.6x faster, more with many trades or funding payments. Results are unchanged.
- **Less work per bar.** Funding is only checked on bars that have a funding rate, and the per-bar position checks and history are leaner. 5 years of 1-minute bars for 10 symbols: about 57 s to 41 s, or 106 s to 72 s with recording. Results are unchanged.

## 0.6.0 (2026-11-19)

### Added

- **Market data.** `overfitting.market.load(["BTC/USDT", "ETH/USDT"], "1h", start="2023-01-01")` downloads candles, and optionally funding rates, from 19 exchanges. Data is saved locally, so later calls only download what is missing. `stored()` shows what is saved; `downloads()` lists every download.
- **Parameter optimizer.** `optimize(MyStrategy, data, params={...})` runs a grid or random search in parallel. `train=0.6` or `train=("2024-01-01", "2025-01-01")` limits the search to part of the data; the winner is then backtested over all of it. See `result.table` and `result.heatmap(x, y)`.
- **Strategy parameters.** Parameters are class attributes; override them with `MyStrategy(data, params={"fast": 10})`.
- `Strategy(record=False)` skips detailed recording for faster runs.
- `Strategy` accepts a prepared `Data` object, so runs can share one.

### Changed

- **Funding is off by default.** Pass `funding=True` to `Strategy` and to `load()`.
- **Shorter printed summary** (about 120 lines down to 87). Duplicate and rarely used metrics were dropped; they remain available in `overfitting.analysis.metrics`.
- Backtests run about 25% faster, and about 50% faster with `record=False`.
- New dependency: `pyarrow`.

### Fixed

- **Stop-limit fills.** A triggered stop-limit with room in its limit now fills at the trigger price, not the limit price (a buy stop at 100 with a limit of 105 fills at 100, not 105). Stop orders that the bar opens past fill at the open.

### Removed

- `overfitting.types`. Import `OrderType` and `Status` from `overfitting.order`.
- `CustomError` and `InvalidOrderType`. All errors now derive from `OverfittingError`.
- The deprecated indicator classes, `Strategy.indicator()`, `slippage_model=` and `Slippage`. Use `self.I(sma, close, 20)` and `slippage=0.001`.

## 0.5.0 (2026-11-18)

This release fixes correctness problems in the engine, so **backtest results will change**.

### Changed

- **Equity is marked to market.** Returns now include unrealized PnL at each bar's close. Use `run(mark_to_market=False)` for the old realized-only curve.
- **Orders need enough margin.** Orders that open or increase a position raise `InsufficientMargin` when free margin is short.
- **Liquidation loses the initial margin only.** It used to also charge maintenance margin and commission.
- **Trade statistics count completed trades, not fills**, and include commission and funding.
- **New printed report** with seven sections instead of 23 lines. Values are in `Report.metrics`.
- Some report lines changed meaning: `Total Return` replaces `Cumulative Return`; skew and kurtosis use daily returns; value at risk is historical.
- Drawdown duration is counted in calendar days.
- **Charts.** `strategy.plot()` shows 13 charts and saves them as `.png`. `seaborn` and `plotly` are no longer needed.
- Minimum pandas version is 2.2.

### Added

- **Funding rate.** A `funding_rate` column settles funding at the bar's open. See `strategy.fetch_funding()`.
- **Indicators as functions.** `self.I(sma, close, 20)`, read as `self.fast[-1]`. Functions live in `overfitting.indicators`.
- **Slippage as a number or function.** `slippage=0.001`, or any `(order, bar) -> price`.
- **Trade analysis.** `fetch_round_trips()`, `fetch_account_history()`, `fetch_position_history()` and `fetch_orders()`.
- Trade, cost, exposure and order metrics in `overfitting.analysis.metrics`.
- Six new charts, each also a function in `overfitting.analysis.plots`.
- Liquidations are flagged in the trade log.
- Equity and margin helpers: `get_equity()`, `get_available_margin()`.
- `next()` can be written without the bar index.

### Deprecated

- The indicator classes and `Strategy.indicator()`. Use `Strategy.I`.
- `slippage_model=` and `Slippage`. Use `slippage=`.

### Fixed

- `Broker.order()` crashed when `type` was omitted.
- Timezone-aware timestamps are now converted to naive ones.
- `initial_captial` typo renamed to `initial_capital`; the old name still works.
- Benchmark returns were shifted by one day for intraday data.

### Removed

- `analysis/graph.py`. The package no longer contains third-party code.

### Project

- Added a pytest suite and GitHub Actions CI on Python 3.10 to 3.13.
- `requires-python` is now 3.10.
