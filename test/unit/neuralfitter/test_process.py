from unittest import mock

from decode.neuralfitter import process


def test_process_supervised_flow():
    # very superficial mocked base test to check whether the respective flow is correct
    model_input = mock.MagicMock()
    tar = mock.MagicMock()
    tar_em = mock.MagicMock()
    post_model = mock.MagicMock()
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
    post_model.forward.assert_not_called()
