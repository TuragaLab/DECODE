import pytest
import torch

from decode.generic import test_utils as utils


@pytest.mark.parametrize("a", [None, 1])
@pytest.mark.parametrize("b", [None, 2])
def test_equal_none(a, b):
    out = utils.equal_none(a, b)

    if a is None and b is None:
        assert out
    if (a is not None and b is None) or (a is None and b is not None):
        assert not out
    if a is not None and b is not None:
        assert not out


@pytest.mark.parametrize("a", [None, 1])
@pytest.mark.parametrize("b", [None, 2])
def test_equal_none(a, b):
    out = utils.equal_optional(a, b)

    if a is None and b is None:
        assert out
    if (a is not None and b is None) or (a is None and b is not None):
        assert not out
    if a is not None and b is not None:
        assert out


@pytest.mark.parametrize("a", [None, torch.ones(42)])
@pytest.mark.parametrize("b", [None, torch.ones(42)])
@pytest.mark.parametrize("equal_none", ["both", "either"])
def test_tens_almeq_equal_none(a, b, equal_none):
    equal = utils.tens_almeq(a, b, none=equal_none)