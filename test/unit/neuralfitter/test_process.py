import numpy as np
import pytest
import torch
from unittest import mock
from contextlib import nullcontext

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
        (slice(2, 4), 3, [[1, 2, 3], [2, 3, 4]]),
        (slice(2, 5, 2), 3, [[1, 2, 3], [3, 4, 5]]),
        (-1, 3, NotImplementedError),
        (100, 3, IndexError),
    ],
)
def test_ix_window(ix, window, ix_expct):
    window = process.IxWindow(window, 100)

    if isinstance(ix_expct, type) and isinstance(ix_expct(), Exception):
        with pytest.raises(ix_expct):
            window._compute(ix)
    else:
        ix_expct = torch.LongTensor(ix_expct)
        np.testing.assert_array_equal(window(ix), ix_expct)


@pytest.mark.parametrize("alias", ["__call__", "__getitem__"])
def test_ix_window_aliases(alias):
    w = process.IxWindow(3, None)

    with mock.patch.object(w, "_compute") as mock_compute:
        _ = getattr(w, alias)(0)
    mock_compute.assert_called_once()


def test_ix_window_attach():
    x = torch.rand(10, 32, 4)
    windower = process.IxWindow(3, None)
    x_sliced = windower.attach(x)

    assert x_sliced[5].size() == torch.Size([3, 32, 4])


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
