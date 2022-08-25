from unittest import mock

import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter import model


@pytest.fixture
def mock_model():
    loss = mock.MagicMock()
    loss.forward.return_value = torch.tensor(17.)

    evaluator = mock.MagicMock()
    evaluator.return_value = {"rmse": 17., "mad": 42.}

    return model.Model(
        model=mock.MagicMock(),
        loss=loss,
        proc=mock.MagicMock(),
        em_val_tar=mock.MagicMock(),
        evaluator=evaluator,
    )


@pytest.mark.parametrize("method", ["training_step", "validation_step"])
def test_model_train_val_shared(method, mock_model):
    x, y = mock.MagicMock(), mock.MagicMock()

    loss = getattr(mock_model, method)((x, y), 42)
    assert loss == 17.
    mock_model._model.forward.assert_called_once_with(x)
    mock_model._proc.post_model.assert_called_once_with(
        mock_model._model.forward.return_value
    )
    mock_model._loss.forward.assert_called_once_with(
        y,
        mock_model._proc.post_model.return_value,
    )


def test_model_validation(mock_model):
    x, y = mock.MagicMock(), mock.MagicMock()

    em_dummy_out = emitter.factory(frame_ix=[1, 2])
    em_dummy_tar = em_dummy_out.clone()
    em_dummy_tar.frame_ix = torch.LongTensor([43, 44])

    mock_model._proc.post.return_value = em_dummy_out

    _ = mock_model.validation_step((x, y), 42)
    assert mock_model._em_val_out == [em_dummy_out]


def test_model_on_validation_epoch_end(mock_model):
    mock_model.on_validation_epoch_end()
