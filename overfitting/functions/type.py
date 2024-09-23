from enum import Enum

def enum(*args):
    """
    Parameters:
    *args (str): A variable number of string arguments representing the enum names.

    Returns:
    Enum: An Enum class with the provided names.
    """
    return Enum('Enum', args)

TYPE = enum('tp', 'sl', 'limit', 'market')
STATUS = enum(
    'OPEN', 
    'CANCELLED', 
    'FILLED', 
    'REJECTED'
)