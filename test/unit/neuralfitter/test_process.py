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


def test_process_supervised_input():
    pre = mock.MagicMock()
    p = process.ProcessingSupervised(pre_input=pre)

    p.input(mock.MagicMock(), mock.MagicMock(), mock.MagicMock())

    pre.forward.assert_called_once()


def test_process_supervised_tar():
    pre = mock.MagicMock()
    tar = mock.MagicMock()

    p = process.ProcessingSupervised(pre_tar=pre, tar=tar)
    p.tar(mock.MagicMock(), mock.MagicMock())

    tar.forward.assert_called_once()
    pre.forward.assert_called_once()


def test_process_supervised_post():
    post_model = mock.MagicMock()
    post = mock.MagicMock()

    p = process.ProcessingSupervised(post_model=post_model, post=post)

    p.post(mock.MagicMock())
    post.forward.assert_called_once()
    post_model.forward.assert_called_once()
