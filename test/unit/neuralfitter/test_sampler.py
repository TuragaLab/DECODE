import pytest
import torch
from unittest import mock

from decode.emitter import emitter
from decode.neuralfitter import sampler


def test_slicer_delayed():
    def product(x, *, y):
        return x + y

    a = [1, 2, 3]
    b = [4, 5, 6]

    s = sampler._SlicerDelayed(product, a, y=b)
    assert s[0] == 5
    assert s[:] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("prop", ["input", "target"])
def test_sampler_target(prop):
    em, frame, bg = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    proc = mock.MagicMock()

    s = sampler.SamplerSupervised(em, bg, mock.MagicMock(), proc)
    s._frames = frame

    # test sliceability
    _ = s.target[0]
    _ = s.target[:]

    with mock.patch.object(sampler, "_SlicerDelayed") as mock_slicer:
        if prop == "input":
            s.input[0]
            # make sure that iframe is used and not simple emitter indexing
            mock_slicer.assert_called_once_with(proc.input, frame=frame, em=em.iframe, bg=bg)
        elif prop == "target":
            s.target[0]
            mock_slicer.assert_called_once_with(proc.tar, em=em.iframe, bg=bg)
        else:
            raise NotImplementedError
