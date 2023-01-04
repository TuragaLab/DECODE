from unittest import mock

import pytest

from decode.neuralfitter import process


@pytest.mark.parametrize("post_model", [None, mock.MagicMock()])
def test_process_supervised_flow(post_model):
    # very superficial mocked base test to check whether the respective flow is correct
    model_input = mock.MagicMock()
    tar = mock.MagicMock()
    tar_em = mock.MagicMock()
    post = mock.MagicMock()

    p = process.ProcessingSupervised(
        m_input=model_input,
        tar=tar,
        tar_em=tar_em,
        post=post,
        post_model=post_model,
    )

    p.pre_train(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
    model_input.forward.assert_called_once()

    p.tar_em(mock.MagicMock())
    tar_em.forward.assert_called_once()

    p.post(mock.MagicMock())
    post.forward.assert_called_once()
    if post_model is not None:
        post_model.forward.assert_not_called()

    m = p.post_model(mock.MagicMock())
    if post_model is not None:
        post_model.forward.assert_called_once()
    else:
        assert m is not None
