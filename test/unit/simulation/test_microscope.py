import pytest
import torch

from decode import emitter_factory
from decode.simulation import microscope
from decode.simulation import noise as noise_lib
from decode.simulation import psf_kernel


@pytest.fixture
def psf():
    return psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 39.5), (32, 40))


@pytest.fixture
def noise():
    return noise_lib.Poisson()


@pytest.mark.parametrize("bg", [None, torch.rand(32, 40)])
def test_microscope(bg, psf, noise):
    m = microscope.Microscope(psf, noise, frame_range=(0, 10))
    em = emitter_factory(10, xy_unit="px")

    frames = m.forward(em, bg)

    assert frames.size() == torch.Size([10, 32, 40])


def test_microscope_multi_channel():
    psf = [
        psf_kernel.DeltaPSF((0., 32.), (0., 32.), (32, 32)),
        psf_kernel.DeltaPSF((0., 32.), (0., 32.), (32, 32))
    ]
    noise = [
        noise_lib.ZeroNoise(),
        noise_lib.ZeroNoise()
    ]

    m = microscope.MicroscopeMultiChannel(psf, noise, (-5, 5), (-2, 0))
    em = emitter_factory(3, frame_ix=[-5, 0, 5], code=[-5, -1, 1], xy_unit="px")

    frames = m.forward(em)

    assert frames.size() == torch.Size([10, 2, 32, 32])
    assert (frames[:5] == 0).all()
    assert (frames[6:] == 0).all()
    assert not (frames[5, 1] == 0).all()
