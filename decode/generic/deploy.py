from importlib import import_module
from typing import Optional
from functools import wraps


def raise_optional_deps(lib: str, msg: Optional[str] = ""):
    """
    Decorator that checks if an import is available or otherwise raises an ImportError
    along with an optional message.

    Args:
        lib: module to be imported
        msg: error message

    Examples:
        ```
        @raise_optional_deps("numpy", "Numpy not installed.")
        def fn_depends_on_numpy():
            ...
        ```
    """
    def wraps_fn(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            try:
                import_module(lib)
            except ImportError:
                raise ImportError(f"{lib} not found. {msg}")
            return fn(*args, **kwargs)
        return wrapped_fn
    return wraps_fn
