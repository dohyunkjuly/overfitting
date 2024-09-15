class CustomError(Exception):
    def __init__(self, msg, **kwargs):
        self.msg = msg
        self.kwargs = kwargs

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg


class EmptyOrderParameters(CustomError):
    pass

class InvalidOrder(CustomError):
    pass