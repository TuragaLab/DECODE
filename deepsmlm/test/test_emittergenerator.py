import pytest
import torch

import deepsmlm.simulation.emitter_generator as emgen
import deepsmlm.generic.emitter as em


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

    @pytest.fixture(params=[[None, 10.], [2., None]], ids=["em_av", "dens"])
    def em_pop(self, request):
        from deepsmlm.simulation.structure_prior import RandomStructure

        dens, em_av = request.param  # unpack arguments

        structure = RandomStructure((10., 20.), (30., 40.), (1000, 2000.))

        cand = emgen.EmitterPopperSingle(structure=structure,
                                         photon_range=(100, 2000),
                                         xy_unit='px',
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
        exp_av = em_pop.emitter_av
        em_av_out = []

        """Run"""
        n = 10
        for i in range(n):
            em_av_out.append(len(em_pop().get_subset_frame(0, 0)))

        em_av_out = torch.tensor(em_av_out).float()

        """Assert"""
        assert em_av_out.mean() == pytest.approx(exp_av, exp_av/10), "Average seems to be off."

    def test_frame_ix(self, em_pop):
        """Make sure that the frame_ix is 0."""

        """Run and Test"""
        n = 10
        for _ in range(n):
            assert (em_pop().frame_ix == 0).all()


@pytest.mark.skip("Deprecation.")
class TestEmitterPopperMultiFrame:

    @pytest.fixture(scope='class')
    def empopper(self):
        structure_prior = RandomStructure((-0.5, 63.5),
                                          (-0.5, 63.5),
                                          (-750., 750.))

        return emgen.EmitterPopperMultiFrame(structure=structure_prior,
                                             density=None,
                                             intensity_mu_sig=(1000., 300.),
                                             lifetime=1.5,
                                             num_frames=3.,
                                             emitter_av=60)

    def test_loose_emset(self, empopper):
        loose_em = empopper.gen_loose_emitter()
        assert loose_em.intensity.mean().item() == pytest.approx(1000., 0.1)
        assert loose_em.intensity.std().item() == pytest.approx(300., 0.1)

    def test_emset_after_distribution(self, empopper):
        emset = empopper.pop()
        assert emset.phot.max().item() <= 1000 + 5 * empopper.intensity_mu_sig[1]
        assert emset.phot.min().item() >= 0
