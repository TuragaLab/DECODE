import builtins
import sys
from typing import Sequence


def sys_check(version, fn_builtin):
    """

    Args:
        version: threshold version, below which an own implementation of zip will be
         used
        fn_builtin: function to use in case current python version matches or exceeds
         that defined by the version variable

    """

    def inner1(myfunc):
        def inner2(*args, **kwargs):
            if sys.version >= version:
                return fn_builtin(*args, **kwargs)
            return myfunc(*args, **kwargs)

        return inner2

    return inner1


@sys_check(version="3.10", fn_builtin=builtins.zip)
def zip(*args: tuple[Sequence, ...], strict: bool = False) -> builtins.zip:
    """
    Future of zip (py 3.10).

    Note: Different to the original zip, args must be Sequence because we need to check
    the length of the sequences before the actual implementation.

    Args:
        *args:
        strict: whether all sequences must have the same length

    Returns:
        generator
    """
    if strict:
        if not all(len(args[0]) == len(a) for a in args):
            raise ValueError("All arguments must have same length.")
    return builtins.zip(*args)
