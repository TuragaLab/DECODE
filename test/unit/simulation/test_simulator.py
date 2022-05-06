import pytest
import torch
from unittest import mock

from decode.emitter import emitter
from decode.simulation import background
from decode.simulation import psf_kernel
from decode.simulation import simulator  # test candidate


class TestSimulator:
    @pytest.fixture(scope="class", params=[32, 64])
    def sim(self, request):
        psf = psf_kernel.GaussianPSF(
            (-0.5, 31.5),
            (-0.5, 31.5),
            (-750.0, 750.0),
            (request.param, request.param),
            sigma_0=1.0,
        )
        bg = background.UniformBackground(10.0)
        sim = simulator.Simulation(
            psf=psf, background=bg, noise=None, frame_range=(-1, 2)
        )

        return sim

    @pytest.fixture(scope="class")
    def em(self):
        return emitter.factory(10, xy_unit="px")

    def test_sampler(self, sim):
        def dummy_sampler():
            return emitter.factory(20, xy_unit="px")

        sim.em_sampler = dummy_sampler

        em, frames, bg_frames = sim.sample()

        assert isinstance(em, emitter.EmitterSet)
        assert isinstance(frames, torch.Tensor)
        assert bg_frames is None

    @pytest.mark.parametrize(
        "ix_low,ix_high", [(None, None), (0, None), (None, 1), (-5, 6)]
    )
    def test_forward(self, sim, ix_low, ix_high):
        # check that psf has been called with apropriate frame index limits

        sim.frame_range = (None, None)
        sim.background = None
        sim.noise = None

        em = emitter.factory(2, xy_unit="px")
        em.frame_ix = torch.tensor([-2, 3]).long()

        with mock.patch.object(sim.psf, "forward") as mock_forward:
            frames, bg_frames = sim.forward(em, ix_low=ix_low, ix_high=ix_high)

        mock_forward.assert_called_once_with(
            xyz=em.xyz_px,
            weight=em.phot,
            frame_ix=em.frame_ix,
            ix_low=ix_low,
            ix_high=ix_high,
        )
