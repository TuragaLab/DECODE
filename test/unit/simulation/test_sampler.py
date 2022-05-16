import numpy as np
import pytest
import torch

from decode.emitter.emitter import EmitterSet
import decode.emitter as em
import decode.simulation.sampler as emgen
from decode.simulation.structures import RandomStructure


class TestEmitterSamplerABC:
    @pytest.fixture()
    def em_pop(self):
        class DummyEmitterPopper(emgen.EmitterSampler):
            def sample(self):
                return em.factory(10)

        return DummyEmitterPopper(structure=None, xy_unit="px", px_size=None)

    def test_sample(self, em_pop):
        assert isinstance(em_pop(), EmitterSet), "Wrong output type."


class TestEmitterSamplerFrameIndependent(TestEmitterSamplerABC):
    @pytest.fixture()
    def structure(self):
        return RandomStructure((10.0, 20.0), (30.0, 40.0), (1000, 2000.0))

    @pytest.fixture(params=[[None, 10.0], [2.0, None]], ids=["em_av", "dens"])
    def em_pop(self, request, structure):

        dens, em_av = request.param  # unpack arguments
        cand = emgen.EmitterSamplerFrameIndependent(
            structure=structure,
            photon_range=(100, 2000),
            xy_unit="px",
            px_size=(1.0, 1.0),
            density=dens,
            em_avg=em_av,
        )

        return cand

    def test_sample(self, em_pop):
        super().test_sample(em_pop)
        assert (em_pop().frame_ix == 0).all()

    @pytest.mark.skip("Not implemented.")
    @pytest.mark.parametrize("n", [-1, 0, 1, 10, 100])
    def test_sample_n(self, em_pop, n):

        if n >= 0:
            assert len(em_pop.sample_n(n=n)) == n

        else:
            with pytest.raises(
                ValueError, match="Negative number of samples is not well-defined."
            ):
                em_pop.sample_n(n)

    def test_average(self, em_pop):
        """
        Tests whether average number returned by EmitterPopper is roughly okay.
        """

        em_av_out = [len(em_pop().get_subset_frame(0, 0)) for _ in range(10)]
        em_av_out = torch.tensor(em_av_out).float().mean()

        assert em_av_out == pytest.approx(
            em_pop._em_avg, em_pop._em_avg / 10
        ), "Emitter average seems to be off."


class TestEmitterPopperMultiframe(TestEmitterSamplerFrameIndependent):
    @pytest.fixture(params=[[None, 10.0], [2.0, None]], ids=["em_av", "dens"])
    def em_pop(self, request, structure):
        dens, em_av = request.param  # unpack arguments
        cand = emgen.EmitterSamplerBlinking(
            structure=structure,
            intensity_mu_sig=(100, 2000),
            lifetime=2.0,
            frame_range=(-1, 2),
            xy_unit="px",
            px_size=(1.0, 1.0),
            density=dens,
            em_avg=em_av,
        )

        return cand

    def test_frame_lifetime_properties(self, em_pop):
        assert em_pop.num_frames == 3, "Wrong number of frames."
        for l in em_pop._frame_range_plus:
            assert isinstance(l, float), "Frame duration (plus) must be float"
        assert isinstance(
            em_pop._num_frames_plus, float
        ), "Number of frames including lifetime must be float."

    def test_sample(self, em_pop):
        e = em_pop.sample()

        assert isinstance(e, EmitterSet)
        assert (e.frame_ix.unique() == torch.tensor([-1, 0, 1])).all()

    @pytest.mark.slow()
    def test_sample_uniformity(self, structure):
        """
        Tests whether there are approx. equal amount of fluorophores on all frames.
        Tested with a high number for statistical reasons.
        This test could fail by statistical means.
        """
        em_gen = emgen.EmitterSamplerBlinking(
            structure=structure,
            intensity_mu_sig=(100, 2000),
            lifetime=2.0,
            frame_range=(0, 1000),
            xy_unit="px",
            px_size=(1.0, 1.0),
            density=None,
            em_avg=10000,
        )

        emitters = em_gen.sample()

        bin_count, _ = np.histogram(emitters.frame_ix, bins=np.arange(1001))
        bin_count = torch.from_numpy(bin_count)

        np.testing.assert_allclose(
            bin_count, torch.ones_like(bin_count) * 10000, atol=1000
        )
        assert bin_count.float().mean() == pytest.approx(10000, rel=0.05)
