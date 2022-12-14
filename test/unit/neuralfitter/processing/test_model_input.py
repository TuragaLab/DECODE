from unittest import mock

import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter.processing import model_input
from decode.simulation import camera


def test_model_in():
    noise = [camera.CameraPerfect(), camera.CameraPerfect()]

    m = model_input.ModelInputPostponed(
        cam=noise,
        cat_input=None,
    )
    x = [torch.rand(3, 8), torch.rand(3, 8)]
    em = emitter.factory(100)
    bg = [torch.rand(3, 8), torch.rand(3, 8)]
    aux = [torch.zeros(3, 8), torch.rand(3, 8)]

    out = m.forward(x, em=em, bg=bg, aux=aux)
    
    assert out.size() == torch.Size([4, 3, 8])


@pytest.mark.parametrize("cat_impl", [None, mock.MagicMock()])
def test_model_input_postponed_cat_input(cat_impl):
    m = model_input.ModelInputPostponed(None, cat_input=cat_impl)

    out = m._cat_input(
        frame=torch.unbind(torch.rand(4, 3, 8), 0),
        aux=torch.unbind(torch.rand(3, 3, 8), 0)
    )

    if cat_impl is not None:
        cat_impl.assert_called_once()
        return

    assert out.size() == torch.Size([7, 3, 8])
