from unittest import mock

import pytest

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


def test_process_supervised_input():
    model_input = mock.MagicMock()

    p = process.ProcessingSupervised(m_input=model_input)

    args = mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    p.input(*args)

    model_input.forward.assert_called_once()


def test_process_supervised_tar():
    tar = mock.MagicMock()
    tar_em = mock.MagicMock()

    p = process.ProcessingSupervised(tar=tar, tar_em=tar_em)
    p.tar(mock.MagicMock(), mock.MagicMock())
    tar.forward.assert_called_once()

    p.tar_em(mock.MagicMock())
    tar_em.forward.assert_called_once()


def test_process_supervised_post():
    post_model = mock.MagicMock()
    post = mock.MagicMock()

    p = process.ProcessingSupervised(post_model=post_model, post=post)

    p.post(mock.MagicMock())
    post.forward.assert_called_once()
    post_model.forward.assert_called_once()
