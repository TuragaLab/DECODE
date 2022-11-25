from unittest import mock

import pytest
import torch

from decode.emitter import emitter
from decode.simulation import camera
from decode.neuralfitter.processing import model_input


def test_model_in():
    noise = [camera.CameraPerfect(), camera.CameraPerfect()]

    m = model_input.ModelInputINeedToFindAName(
        noise=noise,
        cat_frame=torch.stack,
        cat_input=torch.cat,
    )
    x = [torch.rand(3, 8), torch.rand(3, 8)]
    noise = camera.CameraPerfect()
    em = emitter.factory(100)
    aux = {
        "bg": [torch.rand(3, 8), torch.rand(3, 8)],
        "indicator": [torch.zeros(4, 8), torch.rand(3, 8)]
    }

    out = m.forward(x, em=em, aux=aux)
    
    assert out.size() == torch.Size([4, 3, 8])
    out


@pytest.mark.parametrize("frame,bg,size", [
    (torch.rand(3, 4), torch.rand(3, 4), [3, 4]),
    (torch.rand(3, 4), None, [3, 4]),
    (
            [torch.rand(3, 4), torch.rand(7, 8)],
            [torch.rand(3, 4), torch.rand(7, 8)],
            [[3, 4], [7, 8]]
    )
])
def test_merger_frame_bg(frame, bg, size):
    m = model_input.MergerFrameBg()

    out = m.forward(frame, bg)

    if not isinstance(size[0], list):
        assert out.size() == torch.Size(size)
    else:
        for o, s in zip(out, size):
            assert o.size() == torch.Size(s)
