# Changelog

## 0.5.0 (unreleased)

This release fixes correctness problems in the engine. **Backtest results will change**: Sharpe, drawdown and other return statistics move for any strategy that holds positions across bars, and strategies that were over-sized or liquidated will end with different balances.

### Changed

- **Equity is marked to market.** `Strategy.run()` now bases returns on cash plus the unrealized PnL of open positions at each bar's close. Before, the curve only moved when a trade closed, which hid drawdowns on open positions. Pass `run(mark_to_market=False)` for the old realized-only curve. Both are always available as `strategy.equity_curve` and `strategy.realized_balances`.
- **Orders are checked against available margin.** An order that opens or increases a position needs `price × qty / leverage` of free margin (cash minus margin locked in open positions). `InsufficientMargin` is raised at placement; a resting order that is no longer affordable when it fills is dropped and listed in `strategy.fetch_rejected_orders()`. Reducing and closing orders are always accepted. Before, any size was accepted.
- **Liquidation loses the initial margin, no more.** Before, a liquidation charged initial margin plus maintenance margin plus commission, which is more than was posted for the position. `Position.margin` is now the initial margin only. The liquidation price formula is unchanged.

### Added

- **Funding rate.** An optional `funding_rate` column settles funding at the open of each bar where it is non-zero. Payments are listed in `strategy.fetch_funding()`.
- Liquidations are flagged in the trade log (`is_liquidation_order`).
- `Strategy.get_equity()`, `Strategy.get_available_margin()`, `Broker.equity()`, `Broker.unrealized_pnl()`, `Broker.available_margin()`, `Broker.required_margin()`, `Position.unrealized_pnl()`.

### Fixed

- `Broker.order()` crashed when `type` was omitted. It now defaults to a limit order and also accepts an `OrderType`.
- Timezone-aware timestamps are converted to naive ones as documented.
- `Broker.initial_captial` is renamed `initial_capital`. The old name still works and emits a `DeprecationWarning`.

## 0.4.3

- Added a pytest suite and GitHub Actions CI.
- `requires-python` raised to 3.10, the lowest version the code imports on.
