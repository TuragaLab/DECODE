import builtins
import sys
from typing import Generator, Sequence


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
    if sys.version >= "3.10":
        return builtins.zip(*args, strict=strict)

    if strict:
        if not all(len(args[0]) == len(a) for a in args):
            raise ValueError("All arguments must have same length.")
    return builtins.zip(*args)
