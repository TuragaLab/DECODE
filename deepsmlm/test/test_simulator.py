import torch
import pytest

import deepsmlm.generic.emitter as emitter
import deepsmlm.simulation.background as background
import deepsmlm.simulation.psf_kernel as psf_kernel
import deepsmlm.simulation.simulator as can  # test candidate


class TestSimulator:

    @pytest.fixture(scope='class', params=[32, 64])
    def sim(self, request):
        psf = psf_kernel.GaussianExpect((-0.5, 31.5), (-0.5, 31.5), (-750., 750.), (request.param, request.param),
                                        sigma_0=1.0)
        bg = background.UniformBackground(10.)
        sim = can.Simulation(psf=psf, background=bg, noise=None, frame_range=(-1, 1))

        return sim

    @pytest.fixture(scope='class')
    def em(self):
        return emitter.RandomEmitterSet(10)

    def test_em_eq(self, sim, em):
        """
        Tests whether input emitter and output emitter are the same

        """
        _, _, em_ = sim.forward(em)

        """Tests"""
        assert em_ == em

    def test_framerange(self, sim, em):
        """
        Tests whether the frames are okay.

        """

        """Run"""
        frames, bg, _ = sim.forward(em)

        """Tests"""
        assert frames.size() == torch.Size([3, *sim.psf.img_shape])
        assert (frames[[0, -1]] == 10.).all(), "Only middle frame is supposed to be active."
        assert frames[1].max() > 10., "Middle frame should be active"
