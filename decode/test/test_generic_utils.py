import pytest
import torch

from decode.generic import utils


@pytest.mark.parametrize("arr,expct", [((5108, 3239, 3892, 570, 4994, 3428, 800, 2025, 1206, 655, 1707, 3239),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)),
                                       ((5108, 3239, 1707, 3239), (0, 0, 0, 1))])
def test_cum_count_per_group(arr, expct):

    out = utils.cum_count_per_group(torch.Tensor(arr))
    assert isinstance(out, torch.LongTensor)
    assert (out == torch.LongTensor(expct)).all()
