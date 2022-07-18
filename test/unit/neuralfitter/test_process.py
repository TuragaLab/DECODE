import numpy as np
import pytest
import torch
from unittest import mock

from decode.neuralfitter import process


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_pre(mode):
    p = process.Processing(mode=mode)

    with mock.patch.object(p, "pre_train") as mock_train:
        with mock.patch.object(p, "pre_inference") as mock_infer:
            p.pre(None)

    if mode == "train":
        mock_train.assert_called_once()
        mock_infer.assert_not_called()
    elif mode == "eval":
        mock_train.assert_not_called()
        mock_infer.assert_called_once()


@pytest.mark.parametrize(
    "ix,window,ix_expct",
    [
        # all assuming len(...): 100
        (0, 1, [0]),
        (0, 3, [0, 0, 1]),
        (0, 5, [0, 0, 0, 1, 2]),
        (10, 3, [9, 10, 11]),
        (99, 3, [98, 99, 99]),
    ],
)
def test_ix_window(ix, window, ix_expct):
    ix_expct = torch.LongTensor(ix_expct)

    s = process.IxWindow(window, 100)
    np.testing.assert_array_equal(s(ix), ix_expct)
