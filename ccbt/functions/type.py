from enum import Enum

def enum(*args):
    """
    Parameters:
    *args (str): A variable number of string arguments representing the enum names.

    Returns:
    Enum: An Enum class with the provided names.
    """
    return Enum({name: name for name in args})