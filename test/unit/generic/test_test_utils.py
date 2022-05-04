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


@pytest.mark.parametrize(
    "a,b,exp_both,exp_either",
    [
        (None, None, True, True),
        (None, torch.ones(5), "raise", False),
        (torch.ones(5), None, "raise", False),
        (torch.ones(5), torch.ones(5), True, True),
    ],
)
def test_tens_almeq_equal_none(a, b, exp_both, exp_either):
    # both
    if exp_both == "raise":
        with pytest.raises(ValueError):
            utils.tens_almeq(a, b, none="both")
    else:
        assert utils.tens_almeq(a, b, none="both") == exp_both
    # either
    assert utils.tens_almeq(a, b, none="either") == exp_either
