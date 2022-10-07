from unittest import mock

import pytest
import torch

from decode.emitter import emitter
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


@pytest.mark.parametrize(
    "n,args,kwargs,size_expct",
    [
        (None, [torch.rand(20, 2)], None, [20, 2, 1]),
        (None, None, {"x": torch.rand(20, 2)}, [20, 2, 1]),
        (42, [torch.rand(20, 2)], None, [42, 2, 1]),  # manually set first dim
    ],
)
def test_tensor_delayed_auto_size(n, args, kwargs, size_expct):
    s = sampler._DelayedTensor(
        lambda x: x.view(2, 1), size=None, args=args, kwargs=kwargs
    )
    s = s.auto_size(n)

    assert s.size() == torch.Size(size_expct)
    assert s.size(0) == size_expct[0]
    assert len(s) == s.size(0)


@pytest.fixture
def proc():
    class _DummyProc:
        def input(self, frame, em, aux):
            return frame

        def tar(self, em, aux):
            return em

        def tar_em(self, em):
            return em.iframe[0]

    return _DummyProc()


@pytest.fixture
def sampler_sup(proc):
    em = emitter.factory(frame_ix=torch.randint(0, 100, size=(1000,)))
    return sampler.SamplerSupervised(
        em=em,
        bg=torch.rand(100, 32, 32),
        frames=torch.rand(100, 32, 32),
        proc=proc,
        window=5,
    )


def test_sampler_frame_samples(sampler_sup):
    assert sampler_sup.frame_samples[5].size() == torch.Size([5, 32, 32])


def test_sampler_len(sampler_sup):
    assert len(sampler_sup) == len(sampler_sup.frame)
    assert len(sampler_sup) == 100


def test_sampler_input_target(sampler_sup):
    assert sampler_sup.input[5].size() == torch.Size([5, 32, 32])
    assert sampler_sup.target[5] == sampler_sup._em.iframe[5]


@pytest.mark.parametrize("bg_mode", ["sample", "global"])
def test_sampler_sample(bg_mode, proc):
    em_return = emitter.factory(frame_ix=torch.randint(100, size=(10000, )))
    bg_return = torch.rand(100, 32, 32)
    em = mock.MagicMock()
    em.sample.return_value = em_return
    bg = mock.MagicMock()
    bg.sample.return_value = bg_return

    mic = mock.MagicMock()

    s = sampler.SamplerSupervised(
        em=em, bg=bg, mic=mic, frames=None, proc=proc, bg_mode=bg_mode
    )
    s.sample()

    mic.forward.assert_called_once()
    if bg_mode == "sample":
        mic.forward.assert_called_once_with(em=em_return, bg=None)
    elif bg_mode == "global":
        mic.forward.assert_called_once_with(em=em_return, bg=bg_return)
    else:
        raise ValueError

    assert s.emitter is not None
    assert s.emitter != s.emitter_tar
    assert len(s.emitter_tar) < len(s.emitter)
    assert s.bg is not None
    assert s.frame is not None
