from unittest import mock

import pytest
import torch

from decode.neuralfitter import dataset


@pytest.mark.parametrize(
    "ix,window,pad,ix_expct",
    [
        (0, 1, None, 0),
        (0, 3, None, 1),
        (0, 5, None, 2),
        (0, 1, "same", 0),
        (0, 3, "same", 0),
    ],
)
def test_pad_index(ix, window, pad, ix_expct):
    s = dataset.IxShifter(mode=pad, window=window)
    assert s(ix) == ix_expct


@pytest.mark.parametrize("window,pad,n,len_expct",[
    (1, None, 10, 10),
    (3, None, 10, 8),
    (3, "same", 10, 10),
])
def test_pad_index_len(window, pad, n, len_expct):
    s = dataset.IxShifter(pad, window, n)
    assert len(s) == len_expct


def test_dataset_gmm():
    tar = mock.MagicMock()
    tar.__getitem__.return_value = (torch.rand(100, 4), torch.zeros(100, dtype=torch.bool)), torch.rand(32, 32)
    ds = dataset.DatasetGausianMixture(torch.rand(32, 3, 40, 41), tar)

    assert len(ds) == 32
    x, (tar_em, tar_mask, tar_bg) = ds[14]
