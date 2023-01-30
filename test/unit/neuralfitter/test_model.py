from unittest import mock

import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter import model

from .. import commons


@pytest.fixture
def mock_model():
    model_bb = mock.MagicMock()
    model_bb.forward.return_value = torch.rand(2, 10, 16, 16)

    loss = mock.MagicMock()
    loss.forward.return_value = torch.tensor(17.), {"a": 17}

    opt = torch.optim.Adamw()
    lr_sched = torch.optim.lr_scheduler.StepLR()

    evaluator = mock.MagicMock()
    evaluator.return_value = {"rmse": 17., "mad": 42.}

    m = model.Model(
        model=model_bb,
        loss=loss,
        proc=mock.MagicMock(),
        evaluator=evaluator,
        optimizer=mock.MagicMock(),
        batch_size=16,
    )
    m.log = mock.MagicMock()
    m.trainer = mock.MagicMock()
    m.trainer.loggers = [mock.MagicMock()]
    return m


@pytest.mark.parametrize("batch", [
    (torch.rand(16, 10, 16, 16), torch.rand(16, 10, 16, 16)),
    (
            torch.rand(16, 10, 16, 16),
            torch.rand(16, 10, 16, 16),
            [e.to_dict() for e in emitter.factory(100)],
    ),
])
@pytest.mark.parametrize("device", commons._get_device_types_available())
def test_transfer_batch_to_device(batch, device):
    # tensors should be shipped to device, emitters should not
    m = model.Model(
        model=mock.MagicMock(),
        loss=mock.MagicMock(),
        optimizer=mock.MagicMock(),
        lr_sched=mock.MagicMock(),
        proc=mock.MagicMock(),
        evaluator=mock.MagicMock(),
        batch_size=16,
    )

    batch = m.transfer_batch_to_device(batch, device, 0)
    assert all([b.device == device for b in batch[:2]])
    if len(batch) >= 3:
        assert all([b["xyz"].device == torch.device("cpu") for b in batch[2]])


def test_model_training_step(mock_model):
    x, y = mock.MagicMock(), mock.MagicMock()

    loss = mock_model.training_step((x, y), 42)
    assert loss == 17.
    mock_model._model.forward.assert_called_once_with(x)
    mock_model._proc.post_model.assert_called_once_with(
        mock_model._model.forward.return_value
    )
    mock_model._loss.forward.assert_called_once_with(
        mock_model._proc.post_model.return_value,
        y,
    )


def test_model_validation(mock_model):
    x, y = mock.MagicMock(), mock.MagicMock()
    y_em = emitter.factory(0)

    em_dummy_out = emitter.factory(frame_ix=[1, 2])
    em_dummy_tar = em_dummy_out.clone()
    em_dummy_tar.frame_ix = torch.LongTensor([43, 44])

    mock_model._proc.post.return_value = em_dummy_out

    _ = mock_model.validation_step((x, y, y_em), 42)
    assert mock_model._em_val_out == [em_dummy_out]


def test_model_on_validation_epoch_end(mock_model):
    mock_model._em_val_out = [emitter.factory(0)]
    mock_model._em_val_tar = [emitter.factory(0)]

    mock_model.on_validation_epoch_end()
    mock_model._evaluator.forward.assert_called_once()
