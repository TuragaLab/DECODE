from unittest import mock

import pytest
import torch

from decode.neuralfitter import sampler


def test_delayed_slicer():
    def product(x, *, y):
        return x + y

    a = [1, 2, 3]
    b = [4, 5, 6]

    s = sampler._DelayedSlicer(product, a, y=b)
    assert s[0] == 5
    assert s[:] == [1, 2, 3, 4, 5, 6]


def test_tensor_delayed():
    def unsqueezer(x):
        return x.unsqueeze(0)

    s = sampler._DelayedTensor(
        unsqueezer, size=torch.Size([1, 5]), kwargs={"x": torch.rand(5)}
    )
    assert s.size() == torch.Size([1, 5])
    assert s.size(0) == 1
    assert s.size(1) == 5


@pytest.mark.parametrize("n,args,kwargs,size_expct", [
    (None, [torch.rand(20, 2)], None, [20, 2, 1]),
    (None, None, {"x": torch.rand(20, 2)}, [20, 2, 1]),
    (42, [torch.rand(20, 2)], None, [42, 2, 1]),  # manually set first dim
])
def test_tensor_delayed_auto_size(n, args, kwargs, size_expct):
    s = sampler._DelayedTensor(lambda x: x.view(2, 1), size=None, args=args, kwargs=kwargs)
    s = s.auto_size(n)

    assert s.size() == torch.Size(size_expct)
    assert s.size(0) == size_expct[0]


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

    with mock.patch.object(sampler, "_DelayedTensor") as mock_delayed:
        if prop == "input":
            _ = s.input[0]
            # make sure that iframe is used and not simple emitter indexing
            mock_delayed.assert_called_once_with(
                proc.input,
                kwargs={"frame": s.frame_samples, "em": em.iframe, "aux": bg}
            )
        elif prop == "target":
            _ = s.target[0]
            mock_delayed.assert_called_once_with(
                proc.tar,
                kwargs={"em": em.iframe, "aux": bg}
            )
        else:
            raise NotImplementedError


def test_sampler_frame_samples():
    # tests windowing
    s = sampler.SamplerSupervised(
        em=mock.MagicMock(),
        bg=mock.MagicMock(),
        mic=mock.MagicMock(),
        proc=mock.MagicMock(),
        window=3,
    )
    s.frame = torch.rand(15, 32, 32)
    x = s.frame_samples[5]
    assert x.size() == torch.Size([3, 32, 32])


@pytest.mark.parametrize("bg_mode", ["global", "sample"])
def test_sampler_input(bg_mode):
    mic = mock.MagicMock()

    # tests windowing
    s = sampler.SamplerSupervised(
        em=mock.MagicMock(),
        bg=mock.MagicMock(),
        mic=mic,
        proc=mock.MagicMock(),
        bg_mode=bg_mode,
    )
    s.sample()

    if bg_mode == "global":
        mic.forward.assert_called_once_with(em=s._em, bg=s._bg)
    elif bg_mode == "sample":
        mic.forward.assert_called_once_with(em=s._em, bg=None)


def test_sampler_len():
    s = sampler.SamplerSupervised(
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )

    frame = mock.MagicMock()
    frame.__len__.return_value = 42
    s.frame = frame

    assert len(s) == len(s.frame)
    assert len(s) == 42
