import copy
from unittest import mock

import torch
import pytest

from decode.neuralfitter import data


@pytest.fixture
def mock_exp():
    m = mock.MagicMock()
    m.sample.return_value = mock.MagicMock(), mock.MagicMock()

    return m


def test_prepare_data(mock_exp):
    mock_exp_val = copy.deepcopy(mock_exp)
    dm = data.datamodel.DataModel(mock_exp, mock_exp_val, 4)

    dm.prepare_data()

    mock_exp.sample.assert_not_called()
    mock_exp_val.sample.assert_called_once()


@pytest.mark.parametrize("dl_method", ["train_dataloader", "val_dataloader"])
def test_dm_dataloader(mock_exp, dl_method):
    dm = data.datamodel.DataModel(mock_exp, mock_exp, 4)

    dl = getattr(dm, dl_method)()
    assert isinstance(dl, torch.utils.data.DataLoader)
    assert dl.num_workers == 4
