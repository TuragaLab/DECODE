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
