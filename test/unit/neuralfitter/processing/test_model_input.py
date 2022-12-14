from unittest import mock

import torch

from decode.emitter import emitter
from decode.neuralfitter import scale_transform
from decode.neuralfitter.processing import model_input
from decode.simulation import camera


def test_model_in():
    noise = mock.MagicMock()
    noise.forward.side_effect = lambda x: x
    noise = [noise] * 2

    m = model_input.ModelInputPostponed(
        cam=noise,
        scaler_frame=scale_transform.ScalerAmplitude(2.0, 0.0).forward,
        scaler_aux=scale_transform.ScalerAmplitude(1.0, 1.0).forward,
    )

    out = m.forward(
        frame=torch.unbind(torch.ones(2, 3, 8) * 1e6),
        em=None,
        bg=torch.unbind(torch.zeros(2, 3, 8)),
        aux=torch.unbind(torch.ones(2, 3, 8)),
    )

    assert out.size() == torch.Size([4, 3, 8])
    assert torch.unique(out[:2]) == 500000.
    assert torch.unique(out[2:]) == 0.
