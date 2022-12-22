from typing import Sequence

import pytest
import torch

from decode.emitter import emitter
from decode.simulation import camera
from decode.neuralfitter import process
from decode.neuralfitter.processing import model_input
from decode.neuralfitter import scale_transform


@pytest.mark.parametrize(
    "frame,bg,n_codes",
    [
        (torch.rand(3, 64, 64), torch.rand(64, 64), 1),  # single channel
        # two channel
        (
            torch.unbind(torch.rand(2, 3, 64, 64), 0),
            torch.unbind(torch.rand(2, 64, 64), 0),
            2,
        ),
    ],
)
def test_process_sup_pre_train(frame, bg, n_codes):
    input_prep = model_input.ModelInputPostponed(
        cam=camera.CameraPerfect()
        if n_codes < 2
        else [camera.CameraPerfect()] * n_codes,
        scaler_frame=scale_transform.ScalerAmplitude(1000.0).forward,
    )

    p = process.ProcessingSupervised(
        m_input=input_prep,
        mode="train",
    )

    x = p.pre_train(
        frame=frame, em=emitter.factory(0), bg=bg, aux=torch.rand(7, 64, 64)
    )

    assert isinstance(x, torch.Tensor)

    if isinstance(frame, Sequence):
        assert x.size() == torch.Size([13, 64, 64])
    else:
        assert x.size() == torch.Size([10, 64, 64])


@pytest.mark.skip("to implement")
def test_process_sup_pre_inference():
    pass


@pytest.mark.skip("to implement")
def test_process_sup_tar():
    pass


@pytest.mark.skip("to implement")
def test_process_sup_tar_em():
    pass


@pytest.mark.skip("to implement")
def test_process_sup_post_model():
    pass


@pytest.mark.skip("to implement")
def test_process_sup_post():
    pass
