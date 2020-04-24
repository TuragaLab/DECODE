import numpy as np
import pytest
import torch

import deepsmlm.generic.emitter as em
import deepsmlm.simulation.emitter_generator as emgen
from deepsmlm.generic.utils import test_utils
from deepsmlm.simulation.structure_prior import RandomStructure


class TestEmitterPopperABC:

    @pytest.fixture()
    def em_pop(self):
        class DummyEmitterPopper(emgen.EmitterPopperABC):
            def pop(self):
                return em.RandomEmitterSet(10)

        return DummyEmitterPopper()

    def test_pop_general(self, em_pop):
        """
        A couple of generic tests.

        Args:
            em_pop: fixture

        """

        """Run"""
        out = em_pop()

        """Tests"""
        assert isinstance(out, em.EmitterSet), "Wrong output type."


class TestEmitterPopper(TestEmitterPopperABC):

    @pytest.fixture()
    def structure(self):
        return RandomStructure((10., 20.), (30., 40.), (1000, 2000.))

    @pytest.fixture(params=[[None, 10.], [2., None]], ids=["em_av", "dens"])
    def em_pop(self, request, structure):

        dens, em_av = request.param  # unpack arguments
        cand = emgen.EmitterPopperSingle(structure=structure,
                                         photon_range=(100, 2000),
                                         xy_unit='px',
                                         px_size=(1., 1.),
                                         density=dens,
                                         emitter_av=em_av)

        return cand

    def test_average(self, em_pop):
        """
        Tests whether average number returned by EmitterPopper is roughly okay.

        Args:
            em_pop: fixture

        """

        """Setup"""
        exp_av = em_pop._emitter_av
        em_av_out = []

        """Run"""
        n = 10
        for i in range(n):
            em_av_out.append(len(em_pop().get_subset_frame(0, 0)))

        em_av_out = torch.tensor(em_av_out).float()

        """Assert"""
        assert em_av_out.mean() == pytest.approx(exp_av, exp_av / 10), "Average seems to be off."

    def test_frame_ix(self, em_pop):
        """Make sure that the frame_ix is 0."""

        """Run and Test"""
        n = 100
        for _ in range(n):
            assert (em_pop().frame_ix == 0).all()


class TestEmitterPopperMultiframe(TestEmitterPopper):

    @pytest.fixture(params=[[None, 10.], [2., None]], ids=["em_av", "dens"])
    def em_pop(self, request, structure):
        dens, em_av = request.param  # unpack arguments
        cand = emgen.EmitterPopperMultiFrame(structure=structure,
                                             intensity_mu_sig=(100, 2000),
                                             xy_unit='px',
                                             px_size=(1., 1.),
                                             lifetime=2.,
                                             density=dens,
                                             emitter_av=em_av)

        return cand

    def test_frame_ix(self, em_pop):
        """Run and Test"""
        n = 100
        for _ in range(n):
            assert (em_pop().frame_ix.unique() == torch.tensor([-1, 0, 1])).all()

    def test_frame_specification(self, structure):

        generator = emgen.EmitterPopperMultiFrame(structure=structure,
                                      intensity_mu_sig=(100, 2000),
                                      xy_unit='px',
                                      px_size=(1., 1.),
                                      lifetime=2.,
                                      emitter_av=100,
                                      frames=(-100, 100))

        generator.pop()

    def test_uniformity(self, structure):
        """
        Tests whether there are approx. equal amount of fluorophores on all frames.
        Tested with a high number for statistical reasons. This test can fail by statistical means.
        """

        """Setup"""
        em_gen = emgen.EmitterPopperMultiFrame(structure=structure,
                                               intensity_mu_sig=(100, 2000),
                                               xy_unit='px',
                                               px_size=(1., 1.),
                                               lifetime=2.,
                                               density=None,
                                               emitter_av=10000,
                                               frames=(0, 1000))

        """Run"""
        emitters = em_gen.pop()

        """Asserts"""
        bin_count, _ = np.histogram(emitters.frame_ix, bins=np.arange(1002))
        bin_count = torch.from_numpy(bin_count)

        assert test_utils.tens_almeq(bin_count, torch.ones_like(bin_count) * 10000, 2000)  # plus minus 1000
        assert bin_count.float().mean() == pytest.approx(10000, rel=0.05)
