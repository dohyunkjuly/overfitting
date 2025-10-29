from math import copysign
from overfitting.order import Order

class Position:
    def __init__(self, 
                 symbol:str =None, 
                 maint_margin_rate:float=0.005, 
                 maint_amount:float=0):
        
        self.symbol = symbol
        self.qty = 0.0
        self.price = 0.0
        self.liquid_price = 0.0
        self.margin = 0.0
        self.leverage = 1 # Default Leverage
        self.maint_margin_rate = maint_margin_rate
        self.maint_amount = maint_amount

    def __repr__(self):
        return (f"Position("
                f"symbol='{self.symbol}', "
                f"qty={self.qty}, "
                f"price={self.price}, "
                f"liquid_price={self.liquid_price}, "
                f"margin={self.margin}, "
                f"leverage={self.leverage}, "
                f"maint_margin_rate={self.maint_margin_rate}, "
                f"maint_amount={self.maint_amount})")

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
        [LONG] LP = Entry Price - (Initial Margin - Maintenance Margin)
        [SHORT] LP = Entry Price + (Initial Margin - Maintenance Margin)
        """
        total_cost = self.price * abs(self.qty)
        im = total_cost / self.leverage 
        mm = total_cost * self.maint_margin_rate - self.maint_amount

        # Updates margin and liquidation price
        self.margin = im + mm
        if self.qty > 0: # Long
            self.liquid_price = self.price - (im - mm)
        else: # Short
            self.liquid_price = self.price + (im - mm)
    
    def _liquidate(self):
        """Returns PNL which is position Margin * -1"""
        l = self.margin
        self.qty = 0.0
        self.price = 0.0
        self.liquid_price = 0.0
        self.margin = 0.0
        return -l

    def set_leverage(self, leverage):
        if leverage <= 0 or leverage > 100:
            raise Exception("set_leverage() Invalid Leverage. Please Choose Between 0 and 100")

        self.leverage = leverage
        self._update_liquid_price()

    def _calculate_pnl(self, txn: Order):
        """Assumes this is only called during reducing/closing trades"""
        closing_qty = txn.qty
        trade_size = abs(closing_qty)

        # Closing long position (selling)
        if closing_qty < 0:
            pnl_per_unit = txn.price - self.price
        # Closing short position (buying to cover)
        else:
            pnl_per_unit = self.price - txn.price

        return pnl_per_unit * trade_size

    def update(self, txn: Order, liquidation = False):
        if self.symbol != txn.symbol:
            raise ValueError("Cannot update with a different symbol.")

        if txn.qty == 0:
            raise ValueError("Transaction quantity cannot be zero.")

        if liquidation:
            return self._liquidate()

        pnl = 0.0
        total_qty = self.qty + txn.qty

        # Case 1: Position fully closed
        if total_qty == 0:
            pnl = self._calculate_pnl(txn)
            self.qty = 0
            self.price = 0.0
            self.liquid_price = 0.0

        else:
            txn_side = copysign(1, txn.qty)
            current_side = copysign(1, self.qty) if self.qty != 0 else txn_side

            # Case 2: Partially closing or flipping position
            if txn_side != current_side:
                pnl = self._calculate_pnl(txn)

                # If position flips (e.g., long → short or vice versa)
                if abs(txn.qty) > abs(self.qty):
                    self.price = txn.price  # new position starts at txn price

            else:
                # Case 3: Adding to an existing position
                weighted_cost = (self.price * self.qty) + (txn.price * txn.qty)
                self.price = weighted_cost / total_qty

            self.qty = total_qty
            self._update_liquid_price()

        return pnl
        
    def to_dict(self):
        return{
            'symbol': self.symbol,
            'qty': self.qty,
            'price': self.price,
            'liquid_price': self.liquid_price,
            'margin': self.margin,
            'leverage': self.leverage
        }