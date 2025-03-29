from math import copysign

class Position:
    # using __slots__ to save on memory usage.
    __slots__ = ['symbol', "qty", "price", "liquid_price", "symbol", 
                 "leverage", "margin", "maint_margin_rate", "maint_amount"]

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
        return (f"Position(symbol='{self.symbol}', qty={self.qty}, price={self.price}, "
                f"liquid_price={self.liquid_price}, leverage={self.leverage}, "
                f"margin={self.margin}, maint_margin_rate={self.maint_margin_rate}, "
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
        

    def _calculate_pnl(self, txn):
        # Closing Long position
        if txn.qty < 0:
            d = txn.price - self.price
        else: # Closing Short
            d = self.price - txn.price

        return d * abs(txn.qty)  # PnL


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
            self.price, self.liquid_price = 0.0, 0.0

        else: 
            # Current Position side & Transaction side
            ts = copysign(1, txn.qty)
            cs = copysign(1, self.qty) 

            if self.qty == 0:
                ts = cs

            # Partially closing a position
            if cs != ts:
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
                
        # Update the quantity
        self.qty = total_qty
        # Then update the liquid price
        self._update_liquid_price()

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
        if leverage <= 0:
            raise Exception("set_leverage() Invalid Leverage")

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