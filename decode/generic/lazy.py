import functools


def no_op_on(attr: str):
    """
    Modifies a method to no-op if specified class / instance attribute is None; original
    arguments are then returned. Only works for positional arguments.

    Args:
        attr: attribute to check

    Examples:
        >>> class Dummy:
        >>>    def __init__(self, factor):
        >>>        self._factor = factor
        >>>    @no_op_on("_factor")
        >>>    def multiply(self, /, x):
        >>>        return x * self._factor

    """
    def wrapping_method(fn):
        @functools.wraps(fn)
        def wrapped_method(self, *args):
            if getattr(self, attr) is None:
                return args if len(args) >= 2 else args[0]
            return fn(self, *args)
        return wrapped_method
    return wrapping_method
