class OverfittingError(Exception):
    """Base class for every error raised by overfitting."""


class InitializationError(OverfittingError):
    """The data given to a Strategy cannot be used (missing columns, mismatched timelines)."""


class InvalidOrderParameters(OverfittingError):
    """An order argument has the wrong type or value."""


class EmptyOrderParameters(InvalidOrderParameters):
    """An order is missing a value its type needs, such as the price of a LIMIT order."""


class StopOrderRejected(OverfittingError):
    """A stop order would trigger immediately: its stop price is already crossed."""


class InsufficientMargin(OverfittingError):
    """There is not enough free margin to open or increase a position."""


class InvalidLeverage(OverfittingError):
    """Leverage must be between 1 and 100."""


class LiquidationError(OverfittingError):
    """Changing the leverage would liquidate the open position immediately."""
    

class ExchangeError(OverfittingError):
    """An exchange answered with an error, or could not be reached (overfitting.market)."""
