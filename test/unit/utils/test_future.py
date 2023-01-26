import sys

import pytest

from decode.utils import future


@pytest.mark.parametrize("args,strict,err", [
    (([1, 2, 3], [4, 5, 6]), True, None),
    (([1, 2, 3], [4, 5]), True, ValueError),
])
def test_zip_strict(args, strict, err):
    if err is None:
        future.zip(*args, strict=strict)
    else:
        with pytest.raises(err):
            if sys.version < "3.10":
                future.zip(*args, strict=strict)
            else:
                list(future.zip(*args, strict=strict))
