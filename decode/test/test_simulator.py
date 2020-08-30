import pytest
import torch

import decode.generic.emitter as emitter
import decode.simulation.background as background
import decode.simulation.psf_kernel as psf_kernel
import decode.simulation.simulator as can  # test candidate


class TestSimulator:

    @pytest.fixture(scope='class', params=[32, 64])
    def sim(self, request):
        psf = psf_kernel.GaussianPSF((-0.5, 31.5), (-0.5, 31.5), (-750., 750.), (request.param, request.param),
                                     sigma_0=1.0)
        bg = background.UniformBackground(10.)
        sim = can.Simulation(psf=psf, background=bg, noise=None, frame_range=(-1, 1))

        return sim

    @pytest.fixture(scope='class')
    def em(self):
        return emitter.RandomEmitterSet(10)

    def test_framerange(self, sim, em):
        """
        Tests whether the frames are okay.

        """

        """Run"""
        frames, bg = sim.forward(em)

        """Tests"""
        assert frames.size() == torch.Size([3, *sim.psf.img_shape])
        assert (frames[[0, -1]] == 10.).all(), "Only middle frame is supposed to be active."
        assert frames[1].max() > 10., "Middle frame should be active"

    def test_sampler(self, sim):
        """Setup"""

        def dummy_sampler():
            return emitter.RandomEmitterSet(20)

        sim.em_sampler = dummy_sampler

        """Run"""
        em, frames, bg_frames = sim.sample()

        """Assertions"""
        assert isinstance(em, emitter.EmitterSet)

    @pytest.mark.parametrize("ix_low,ix_high,n", [(None, None, 6),
                                                  (0, None, 4),
                                                  (None, 0, 3),
                                                  (-5, 5, 11)])
    def test_forward(self, sim, ix_low, ix_high, n):
        """Tests the output length of forward method of simulation."""

        """Setup"""
        sim.frame_range = (None, None)
        em = emitter.RandomEmitterSet(2)
        em.frame_ix = torch.tensor([-2, 3]).long()

        """Run"""
        frames, bg_frames = sim.forward(em, ix_low=ix_low, ix_high=ix_high)

        """Assert"""
        assert len(frames) == n, "Wrong number of frames."
    #
    # def test_fill_bg_to_em(self, sim):
    #     """Setup"""
    #     sim.background = background.UniformBackground([10., 20.])
    #     sim.bg2em = background.BgPerEmitterFromBgFrame(1, (-0.5, 31.5), (-0.5, 31.5), (32, 32))
    #
    #     """Run"""
    #     em = emitter.RandomEmitterSet(10)
    #     frames, bg = sim.forward(em)
    #
    #     """Assert"""
    #     assert (em.bg >= 10.).all()
    #     assert (em.bg <= 20.).all()
    #
