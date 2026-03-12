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
                f"symbol={self.symbol}, "
                f"qty={self.qty}, "
                f"price={self.price}, "
                f"liquid_price={self.liquid_price}, "
                f"margin={self.margin}, "
                f"leverage={self.leverage}, "
                f"maint_margin_rate={self.maint_margin_rate}, "
                f"maint_amount={self.maint_amount})")

    def _update_liquid_price(self):
        """
        NOTE: liquidation price is calculated based on ISOLATED mode.

        Liquidation Price Calculation:
        * Initial Margin = price * size / leverage
        * Maint Margin = price * size * margin rate - margin amount
        [LONG] LP = Entry Price - (Initial Margin - Maintenance Margin)
        [SHORT] LP = Entry Price + (Initial Margin - Maintenance Margin)
        """
        q = abs(self.qty)
        if q == 0:
            self.liquid_price = 0.0
            self.margin = 0.0
            return

        notional = self.price * q 
        im = notional / self.leverage
        mm = max(0.0, notional * self.maint_margin_rate - self.maint_amount)       
         
        self.margin = im + mm
        delta_p = (im - mm) / q

        if self.qty > 0:      # long
            self.liquid_price = self.price - delta_p
        else:                 # short
            self.liquid_price = self.price + delta_p
    
    def _liquidate(self):
        """Returns PNL which is position margin * -1"""
        pnl = -self.margin
        self._reset()
        return pnl

    def _reset(self):
        self.qty = 0.0
        self.price = 0.0
        self.liquid_price = 0.0
        self.margin = 0.0

    def set_leverage(self, leverage):
        if leverage <= 0 or leverage > 100:
            raise Exception("set_leverage() Invalid Leverage. Please Choose Between 0 and 100")

        self.leverage = leverage
        self._update_liquid_price()

    def _calculate_realized_pnl(self, entry_price: float, exit_price: float, qty: float, side: float):
        """
        side > 0 : position is long
        side < 0 : position is short
        """
        if side > 0:
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty
    
    def process_trade(self, order: Order, liquidation: bool = False) -> float:
        if self.symbol != order.symbol:
            raise ValueError("Cannot process trade with a different symbol.")
        if order.qty == 0:
            raise ValueError("Transaction quantity cannot be zero.")
        
        if liquidation:
            return self._liquidate()

        same_side = self.qty == 0 or (self.qty * order.qty > 0)

        if same_side:
            total = self.qty + order.qty
            self.price = (self.price * self.qty + order.executed_price * order.qty) / total
            self.qty = total
            self._update_liquid_price()
            return 0.0

        # Opposite side — reduce, close, or flip
        close_qty = min(abs(self.qty), abs(order.qty))
        side = 1 if self.qty > 0 else -1
        pnl = self._calculate_realized_pnl(self.price, order.executed_price, close_qty, side)

        self.qty += order.qty

        if self.qty == 0:
            self._reset()
        else: # flip or reduce
            if self.qty * order.qty > 0:  # flipped
                self.price = order.executed_price
            self._update_liquid_price()

        return pnl