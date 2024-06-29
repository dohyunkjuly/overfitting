from math import copysign

class Position:
    def __init__(
        self,
        symbol=None,
        leverage=0,
        maintainence_margin_rate=0.5,
        maintainence_amount=0,
    ):
        self.symbol = symbol
        self.leverage = leverage
        self.maintainence_margin_rate = maintainence_margin_rate
        self.maintainence_amount = maintainence_amount

        self.qty = 0.0
        self.price = 0.0
        self.liquid_price = 0.0

    def __getattr__(self, attr):
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        object.__setattr__(self, attr, value)

    def _update_liquid_price(self, price, qty):
        side = copysign(-1, qty)
        position_cost = price * abs(qty)
        
        self.liquid_price = (
            price
            + (
                (position_cost / self.leverage)  # initial_margin
                - (
                    position_cost * self.maintainence_margin_rate
                    - self.maintainence_amount
                )  # maintainence_margin
            ) * side
        )

    def _calculate_pnl(self, txn):
        side = copysign(1, self.qty)
        price_diff = (self.price - txn.price) if side == -1 else (txn.price - self.price)

        return price_diff * abs(txn.qty) # PnL

    def update(self, txn):
        res = { 'pnl': 0 }
        if self.symbol != txn.symbol: 
            raise Exception('update() updating different symbol.')

        total_qty = self.qty + txn.qty
        
        # Position is closed
        if total_qty == 0:
            # Settle PnL
            res['pnl'] = self._calculate_pnl(txn)
            self.price , self.liquid_price = 0.0
        else:
            position_side = copysign(1, self.qty)
            txn_side = copysign(1, txn.qty)

            # Partially closing a position
            if position_side != txn_side:
                # Settle PnL
                res['pnl'] = self._calculate_pnl(txn)
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
        
        # Finally update the liquidation price and the quantity
        self._update_liquid_price()
        self.qty = total_qty

        return res