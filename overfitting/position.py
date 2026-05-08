from overfitting.order import Order

class Position:
    def __init__(self, symbol: str, maint_margin_rate: float = 0.005, maint_amount: float = 0):
        self.symbol = symbol
        self.maint_margin_rate = maint_margin_rate
        self.maint_amount = maint_amount
        self.leverage = 1

        # Open-position state. All zero when flat.
        self.qty = 0.0
        self.price = 0.0
        self.margin = 0.0
        self.liquid_price = 0.0

    def __repr__(self):
        return (f"Position(symbol={self.symbol}, qty={self.qty}, price={self.price}, "
                f"liquid_price={self.liquid_price}, margin={self.margin}, "
                f"leverage={self.leverage})")

    def _recalc_liquidation(self) -> None:
        """
        Liquidation (isolated margin):
            notional       = abs(qty) * entry_price
            initial_margin = notional / leverage
            maint_margin   = max(0, notional * maint_margin_rate - maint_amount)

            long  liquidation = entry - (initial_margin - maint_margin) / abs(qty)
            short liquidation = entry + (initial_margin - maint_margin) / abs(qty)
        """
        size = abs(self.qty)
        if size == 0:
            # Flat — no margin tied up and no liquidation price to track.
            self.margin = 0.0
            self.liquid_price = 0.0
            return

        # Isolated-margin model.
        # initial_margin: what we lock up to open the position at this leverage.
        # maint_margin:   cushion the exchange requires before forcing liquidation.
        notional = self.price * size
        initial_margin = notional / self.leverage
        maint_margin = max(0.0, notional * self.maint_margin_rate - self.maint_amount)

        self.margin = initial_margin + maint_margin

        # How far the price can move against us before we get liquidated.
        # Long: liquidates when price drops by `offset`.
        # Short: liquidates when price rises by `offset`.
        offset = (initial_margin - maint_margin) / size
        self.liquid_price = self.price - offset if self.qty > 0 else self.price + offset

    def _add_to_position(self, order: Order) -> None:
        new_qty = self.qty + order.qty
        self.price = (self.price * self.qty + order.executed_price * order.qty) / new_qty
        self.qty = new_qty
        self._recalc_liquidation()

    def _reset(self) -> None:
        self.qty = 0.0
        self.price = 0.0
        self.margin = 0.0
        self.liquid_price = 0.0

    def set_leverage(self, leverage: int) -> None:
        if leverage <= 0 or leverage > 100:
            raise Exception("Leverage must be between 1 and 100.")
        self.leverage = leverage
        self._recalc_liquidation()

    def process_trade(self, order: Order, liquidation: bool = False) -> float:
        """Apply an executed order. Returns realized PnL (commission excluded)."""
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol {order.symbol!r} doesn't match position {self.symbol!r}.")
        if order.qty == 0:
            raise ValueError("Order quantity cannot be zero.")

        if liquidation: # Force liquidation
            loss = -self.margin
            self._reset()
            return loss

        # Flat or same-side add: update entry price and increase position size.
        if self.qty == 0 or self.qty * order.qty > 0:
            self._add_to_position(order)
            return 0.0
        else:            
            closed_qty = min(abs(self.qty), abs(order.qty))
            direction = 1 if self.qty > 0 else -1
            # Long earns (exit - entry); short earns (entry - exit). Same formula with sign flip.
            pnl = direction * (order.executed_price - self.price) * closed_qty

            self.qty += order.qty

            if self.qty == 0:
                self._reset()
            else:
                # If we flipped sides, the new entry price is the executed price.
                flipped = self.qty * order.qty > 0
                if flipped:
                    self.price = order.executed_price
                self._recalc_liquidation()

            return pnl
