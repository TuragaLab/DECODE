import torch
import pytest

import deepsmlm.generic.emitter as emitter
import deepsmlm.simulation.emittergenerator as emgen
import deepsmlm.test.utils_ci as tutil
from deepsmlm.simulation.structure_prior import RandomStructure


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
                                             num_frames=1000.,
                                             emitter_av=60)

    def test_loose_emset(self, empopper):
        loose_em = empopper.gen_loose_emitter()
        assert loose_em.intensity.mean().item() == pytest.approx(1000., 0.001)
        assert loose_em.intensity.std().item() == pytest.approx(300., 0.01)

    def test_emset_after_distribution(self, empopper):
        emset = empopper.pop()
        assert emset.phot.max().item() <= 1000 + 5 * empopper.intensity_mu_sig[1]
        assert emset.phot.min().item() >= 0
