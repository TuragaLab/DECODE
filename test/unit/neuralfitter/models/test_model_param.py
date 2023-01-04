from unittest import mock

import pytest
import torch

from decode.neuralfitter.models import model_param as mp


@pytest.mark.parametrize("ch_in_map,err", [
    ([[0]], None),
    ([[0, 1], [1]], ValueError),
    ([[0, 3], [1, 3], [2, 3]], None),
])
def test_double_munet_init(ch_in_map, err):
    if err is not None:
        with pytest.raises(err):
            mp.DoubleMUnet(ch_in_map=ch_in_map, ch_out=48)
    else:
        mp.DoubleMUnet(ch_in_map=ch_in_map, ch_out=48)


@pytest.mark.parametrize("ch_in,ch_map", [
    (1, [[0]]),
    (3, [[0], [1], [2]]),
    (4, [[0, 3], [1, 3], [2, 3]]),
    (6, [[0, 4, 5], [1, 4, 5], [2, 4, 5]]),
])
def test_double_munet(ch_in, ch_map):
    m = mp.DoubleMUnet(ch_in_map=ch_map, ch_out=48, use_last_nl=False)

    x = torch.rand(2, ch_in, 32, 32)

    y = m.forward(x)
    assert y.size() == torch.Size([2, 48, 32, 32])
