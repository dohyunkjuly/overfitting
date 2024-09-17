from math import copysign

class Position:
    # using __slots__ to save on memory usage.
    __slots__ = ['symbol', "qty", "price", "liquid_price", "symbol", 
                 "leverage", "margin", "maint_margin_rate", "maint_amount"]

    def __init__(self, symbol=None, leverage=1, maint_margin_rate=0.5, 
                 maint_amount=0):

        self.symbol = symbol
        self.qty = 0.0
        self.price = 0.0
        self.liquid_price = 0.0
        self.margin = 0.0
        self.leverage = leverage
        self.maint_margin_rate = maint_margin_rate
        self.maint_amount = maint_amount

    def __getattr__(self, attr):
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        object.__setattr__(self, attr, value)

    def _update_liquid_price(self):
        """
        NOTE: liquidation price is calculated based on ISOLATED mode.

        Liquidation Price Calculation:
        * Initial Margin = price * size / leverage
        * Maint Margin = price * size * margin rate - margin amount
        [LONG]
        Entry Price - (Initial Margin - Maintenance Margin)
        [SHORT]
        Entry Price + (Initial Margin - Maintenance Margin)

        liquidation price and margin calculation can vary significantly between
        exchanges, so `maint_margin_rate` and `maint_amount` parameters are configurable.
        """
        side = copysign(-1, self.qty)
        total_cost = self.price * abs(self.qty)
        initial_margin = total_cost / self.leverage
        maint_margin = total_cost * (self.maint_margin_rate / 100) - self.maint_amount

        # Updates margin and liquidation price
        self.margin = initial_margin + maint_margin
        self.liquid_price = self.price + (initial_margin - maint_margin) * side

    def _calculate_pnl(self, txn):
        side = copysign(1, self.qty)
        if side == -1:
            price_diff = self.price - txn.price
        else:
            price_diff = txn.price - self.price

        return price_diff * abs(txn.qty)  # PnL

    def update(self, txn):
        pnl = 0.0
        if self.symbol != txn.symbol:
            raise Exception("update() updating different symbol.")

        if txn.qty == 0:
            raise Exception("update() txn qty cannot be zero.")

        total_qty = self.qty + txn.qty

        # Position is closed
        if total_qty == 0:
            # Settle PnL
            pnl = self._calculate_pnl(txn)
            self.price, self.liquid_price = 0.0
        else:
            position_side = copysign(1, self.qty)
            txn_side = copysign(1, txn.qty)

            # Partially closing a position
            if position_side != txn_side:
                # Settle PnL
                pnl = self._calculate_pnl(txn)
                # Closing short and opening a long or
                # closing long and opening a short position
                if abs(txn.qty) > abs(self.qty):
                    self.price = txn.price
            else:
                # Update the entry price
                position_cost = self.price * self.qty
                txn_cost = txn.price * txn.qty
                total_cost = position_cost + txn_cost
                self.price = total_cost / total_qty

        # Finally update liquidation price and quantity
        self._update_liquid_price()
        self.qty = total_qty
        
        return pnl

    def liquidate(self):
        """
        Liquidates position by resetting quantity, price, 
        liquidation price, and margin. Returns the loss
        """
        l = -self.margin
        self.qty = 0.0
        self.price = 0.0
        self.liquid_price = 0.0
        self.margin = 0.0

        return l
    
    def set_leverage(self, leverage):
        self.leverage = leverage
        self._update_liquid_price()
        
    def to_dict(self):
        return{
            'symbol': self.symbol,
            'qty': self.qty,
            'price': self.price,
            'liquid_price': self.liquid_price,
            'margin': self.margin,
            'leverage': self.leverage
        }