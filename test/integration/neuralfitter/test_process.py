import pytest
import torch
from unittest import mock

from decode.emitter import emitter
from decode.neuralfitter import process


@pytest.fixture
def pre_input():
    m = mock.MagicMock()
    m.forward = lambda x: x / x.max()
    return m


@pytest.fixture
def pre_tar():
    m = mock.MagicMock()
    m.forward = lambda em: em[em.phot > 100]
    return m


@pytest.fixture
def tar():
    m = mock.MagicMock()
    m.forward = lambda em, _: torch.rand(2, 5, 64, 64)
    return m


@pytest.fixture
def post_model():
    m = mock.MagicMock()
    m.forward = lambda x: x * 2
    return m


@pytest.fixture
def post_process():
    m = mock.MagicMock()
    m.forward = lambda x: emitter.factory(len(x))
    return m


@pytest.fixture
def model():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self._param = torch.rand(2, 5, 64, 64)

        def forward(self, x):
            return torch.rand(2, 5, 64, 64) * self._param

    return DummyModel()


@pytest.fixture
def loss():
    m = mock.MagicMock()
    m.forward = lambda x, y: torch.nn.functional.mse_loss(x, y)
    return m


def test_processing_model_input():
    # placeholder if logic gets more complicated
    pass


def test_supervised_model_output(post_model, post_process):
    p = process.ProcessingSupervised(
        post_model=post_model,
        post=post_process,
    )

    model_out = torch.rand(2, 10, 64, 64)

    model_inter = p.post_model(model_out)
    out = p.post(model_inter)

    assert isinstance(out, emitter.EmitterSet)
    assert len(out) == len(model_out)


def test_supervised_total_pipeline(
    pre_input, pre_tar, tar, model, post_model, post_process, loss
):
    p = process.ProcessingSupervised(
        pre_input=pre_input,
        pre_tar=pre_tar,
        tar=tar,
        post_model=post_model,
        post=post_process,
    )

    em = emitter.factory(
        100, phot=torch.rand(100) * 1000, frame_ix=torch.randint(20, size=(100,))
    )
    frames = torch.rand(2, 3, 64, 64)

    # whole forward
    model_in, tar_pre = p.pre_train(frames, em)
    tar = p.tar(tar_pre, None)
    model_out = model.forward(model_in)
    model_inter = p.post_model(model_out)
    em_out = p.post(model_inter)

    # test loss consistency
    loss_val = loss.forward(model_out, tar)

    assert isinstance(em_out, emitter.EmitterSet)
    assert loss_val.dim() == 0
