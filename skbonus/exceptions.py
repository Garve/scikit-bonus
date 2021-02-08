class NoFrequencyError(Exception):
    """
    This exception is thrown when a DatetimeIndex does not have a frequency.
    This can happen if two disjoint DatetimeIndex objects are combined via a union, for example when dealing
    with cross validation.
    """
