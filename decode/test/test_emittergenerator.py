import numpy as np
import pytest
import torch

import decode.generic.emitter as em
import decode.simulation.emitter_generator as emgen
from decode.generic import test_utils
from decode.simulation.structure_prior import RandomStructure


class TestEmitterSamplerABC:

    @pytest.fixture()
    def em_pop(self):
        class DummyEmitterPopper(emgen.EmitterSampler):
            def sample(self):
                return em.RandomEmitterSet(10)

        return DummyEmitterPopper(structure=None, xy_unit='px', px_size=None)

    def test_sample(self, em_pop):
        """
        General tests of the sample method.

        Args:
            em_pop: fixture

        """

        """Tests"""
        assert isinstance(em_pop(), em.EmitterSet), "Wrong output type."


class TestEmitterSamplerFrameIndependent(TestEmitterSamplerABC):

    @pytest.fixture()
    def structure(self):
        return RandomStructure((10., 20.), (30., 40.), (1000, 2000.))

    @pytest.fixture(params=[[None, 10.], [2., None]], ids=["em_av", "dens"])
    def em_pop(self, request, structure):

        dens, em_av = request.param  # unpack arguments
        cand = emgen.EmitterSamplerFrameIndependent(structure=structure,
                                                    photon_range=(100, 2000),
                                                    xy_unit='px',
                                                    px_size=(1., 1.),
                                                    density=dens,
                                                    em_avg=em_av)

        return cand

    @pytest.mark.skip("Not implemented.")
    @pytest.mark.parametrize("n", [-1, 0, 1, 10, 100])
    def test_sample_n(self, em_pop, n):

        if n >= 0:
            assert len(em_pop.sample_n(n=n)) == n

        else:
            with pytest.raises(ValueError, match="Negative number of samples is not well-defined."):
                em_pop.sample_n(n)

    def test_average(self, em_pop):
        """
        Tests whether average number returned by EmitterPopper is roughly okay.

        Args:
            em_pop: fixture

        """

        """Run"""
        em_av_out = [len(em_pop().get_subset_frame(0, 0)) for _ in range(10)]
        em_av_out = torch.tensor(em_av_out).float().mean()

        """Assert"""
        assert em_av_out == pytest.approx(em_pop._em_avg, em_pop._em_avg / 10), \
            "Emitter average seems to be off."

    def test_frame_ix(self, em_pop):
        """Make sure that the frame_ix is 0."""

        """Run and Test"""
        n = 100
        for _ in range(n):
            assert (em_pop().frame_ix == 0).all()


class TestEmitterPopperMultiframe(TestEmitterSamplerFrameIndependent):

    @pytest.fixture(params=[[None, 10.], [2., None]], ids=["em_av", "dens"])
    def em_pop(self, request, structure):
        dens, em_av = request.param  # unpack arguments
        cand = emgen.EmitterSamplerBlinking(structure=structure, intensity_mu_sig=(100, 2000), lifetime=2.,
                                            frame_range=(-1, 1), xy_unit='px', px_size=(1., 1.), density=dens,
                                            em_avg=em_av)

        return cand

    def test_frame_ix(self, em_pop):
        """Run and Test"""
        n = 100
        for _ in range(n):
            assert (em_pop().frame_ix.unique() == torch.tensor([-1, 0, 1])).all()

    def test_frame_specification(self, structure):

        generator = emgen.EmitterSamplerBlinking(structure=structure, intensity_mu_sig=(100, 2000), lifetime=2.,
                                                 frame_range=(-100, 100), xy_unit='px', px_size=(1., 1.), em_avg=100)

        generator.sample()

    @pytest.mark.slow()
    def test_uniformity(self, structure):
        """
        Tests whether there are approx. equal amount of fluorophores on all frames.
        Tested with a high number for statistical reasons. This test can fail by statistical means.
        """

        """Setup"""
        em_gen = emgen.EmitterSamplerBlinking(structure=structure, intensity_mu_sig=(100, 2000), lifetime=2.,
                                              frame_range=(0, 1000), xy_unit='px', px_size=(1., 1.), density=None,
                                              em_avg=10000)

        """Run"""
        emitters = em_gen.sample()

        """Asserts"""
        bin_count, _ = np.histogram(emitters.frame_ix, bins=np.arange(1002))
        bin_count = torch.from_numpy(bin_count)

        assert test_utils.tens_almeq(bin_count, torch.ones_like(bin_count) * 10000, 2000)  # plus minus 1000
        assert bin_count.float().mean() == pytest.approx(10000, rel=0.05)
