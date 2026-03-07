class CustomError(Exception):
    def __init__(self, msg, **kwargs):
        self.msg = msg
        self.kwargs = kwargs

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg


class InitializationError(CustomError):
    pass

class EmptyOrderParameters(CustomError):
    pass

class InvalidOrderParameters(CustomError):
    pass

class InvalidOrderType(CustomError):
    pass

class LiquidationError(CustomError):
    pass