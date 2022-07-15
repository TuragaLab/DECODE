from unittest import mock

import numpy as np
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
            _ = s.input[0]
            # make sure that iframe is used and not simple emitter indexing
            mock_slicer.assert_called_once_with(
                proc.input, frame=frame, em=em.iframe, bg=bg
            )
        elif prop == "target":
            _ = s.target[0]
            mock_slicer.assert_called_once_with(proc.tar, em=em.iframe, bg=bg)
        else:
            raise NotImplementedError


@pytest.mark.parametrize(
    "ix,window,pad,ix_expct",
    [
        (0, 1, None, 0),
        (0, 3, None, 1),
        (0, 5, None, 2),
        (0, 1, "same", 0),
        (0, 3, "same", 0),
    ],
)
def test_pad_index(ix, window, pad, ix_expct):
    s = sampler.IxShifter(mode=pad, window=window)
    assert s(ix) == ix_expct


@pytest.mark.parametrize(
    "ix,window,ix_expct",
    [
        (0, 1, [0]),
        (0, 3, [0, 0, 1]),
        (0, 5, [0, 0, 0, 1, 2]),
        (10, 3, [9, 10, 11]),
        (99, 3, [98, 99, 99]),
    ],
)
def test_ix_window(ix, window, ix_expct):
    ix_expct = torch.LongTensor(ix_expct)

    s = sampler.IxWindow(window, 100)
    np.testing.assert_array_equal(s(ix), ix_expct)
