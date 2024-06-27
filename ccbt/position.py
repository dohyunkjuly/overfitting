class Position():
    __slots__ = ['entry_price', 'total_position','initial_margin', 'maintainence_margin', 
                 'leverage', 'breakeven_price', 'liquidation_price', 'commission', 'unrealized']

    def __init__(self,
                 entry_price=0.0,
                 total_position=0.0,
                 initial_margin=0.0,
                 maintainence_margin=0.0,
                 leverage=0,
                 breakeven_price=0.0,
                 liquidation_price=None,
                 commission=0.0,
                 unrealized=0.0
                ):
        for slot, value in zip(self.__slots__, [entry_price, total_position, initial_margin, maintainence_margin, leverage, breakeven_price, liquidation_price, commission, unrealized]):
            setattr(self, slot, value)


    def __getattr__(self, attr):
        if attr in self.__slots__:
            return object.__getattribute__(self, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr in self.__slots__:
            object.__setattr__(self, attr, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")


