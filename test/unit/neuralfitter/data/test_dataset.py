from unittest import mock

import pytest
import torch

from decode.neuralfitter.data import dataset


@pytest.fixture
def ds_generic():
    x = torch.arange(5)
    y = -torch.arange(5)

    return dataset.DatasetGenericInputTar(x, y)


def test_ds_len(ds_generic):
    assert len(ds_generic) == 5


@pytest.mark.parametrize("em", [None, mock.MagicMock()])
def test_ds_generic_getitem(em):
    x = mock.MagicMock()
    y = mock.MagicMock()

    ds = dataset.DatasetGenericInputTar(x, y, em=em)
    if em is None:
        xx, yy = ds[42]
    else:
        xx, yy, em_sample = ds[42]
        em.iframe.__getitem__.assert_called_once_with(42)

    x.__getitem__.assert_called_once_with(42)
    y.__getitem__.assert_called_once_with(42)
