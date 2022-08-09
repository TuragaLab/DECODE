import pytest

from decode.generic import lazy


@pytest.mark.parametrize("val", [None, 5])
def test_no_op_on(val):
    class Dummy:
        def __init__(self, factor):
            self._factor = factor

        @lazy.no_op_on("_factor")
        def multiply(self, x):
            return x * self._factor

    d = Dummy(val)

    if val is not None:
        assert d.multiply(3) == 15
    else:
        assert d.multiply(3) == 3
