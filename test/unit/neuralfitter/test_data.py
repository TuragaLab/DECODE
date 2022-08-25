from unittest import mock

import torch.utils.data

from decode.neuralfitter import data


def test_datamodel():
    ds_train = mock.MagicMock()
    ds_val = mock.MagicMock()

    dm = data.DataModel(ds_train, ds_val, 4)

    for dl in ["train_dataloader", "val_dataloader"]:
        dl = getattr(dm, dl)()
        assert isinstance(dl, torch.utils.data.DataLoader)
        assert dl.num_workers == 4
