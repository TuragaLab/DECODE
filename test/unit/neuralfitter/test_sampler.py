from unittest import mock

import pytest
import torch

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
def test_sampler_input_target(prop):
    em, frame, bg = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    frame.__len__.return_value = 100
    proc = mock.MagicMock()

    s = sampler.SamplerSupervised(em, bg, mock.MagicMock(), proc)
    s.frame = frame

    # test sliceability
    _ = getattr(s, prop)[0]
    _ = getattr(s, prop)[:]

    with mock.patch.object(sampler, "_SlicerDelayed") as mock_slicer:
        if prop == "input":
            _ = s.input[0]
            # make sure that iframe is used and not simple emitter indexing
            mock_slicer.assert_called_once_with(
                proc.input, frame=s.frame_samples, em=em.iframe, aux=bg
            )
        elif prop == "target":
            _ = s.target[0]
            mock_slicer.assert_called_once_with(proc.tar, em=em.iframe, aux=bg)
        else:
            raise NotImplementedError


def test_sampler_frame_samples():
    s = sampler.SamplerSupervised(
        em=mock.MagicMock(),
        bg=mock.MagicMock(),
        mic=mock.MagicMock(),
        proc=mock.MagicMock(),
        window=3,
    )
    s._frame = torch.rand(15, 32, 32)
    x = s.frame_samples[5]
    assert x.size() == torch.Size([3, 32, 32])


def test_sampler_len():
    s = sampler.SamplerSupervised(
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )

    s._frame = mock.MagicMock()
    s._frame.__len__.return_value = 42

    assert len(s) == len(s.frame)
    assert len(s) == 42
