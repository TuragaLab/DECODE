
import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt

import deepsmlm.test.utils_ci as tutil
import deepsmlm.neuralfitter.pre_processing as prep
import deepsmlm.neuralfitter.weight_generator as wgen

from deepsmlm.generic.emitter import EmitterSet, CoordinateOnlyEmitter, RandomEmitterSet
from deepsmlm.generic.psf_kernel import DeltaPSF
from deepsmlm.neuralfitter.losscollection import OffsetROILoss
from deepsmlm.neuralfitter.pre_processing import ZasOneHot, OffsetRep, GlobalOffsetRep, ROIOffsetRep
import deepsmlm.neuralfitter.pre_processing as pre
from deepsmlm.generic.plotting.frame_coord import PlotFrame, PlotFrameCoord


class TestDerivePseudobgFromBg:

    @pytest.fixture(scope='class')
    def pseudobg(self):
        return wgen.DerivePseudobgFromBg((-0.5, 63.5), (-0.5, 63.5), (64, 64), [17, 17], 8)

    # @pytest.fixture(scope='class')
    def bg_sampler(self):
        em = RandomEmitterSet(20, 64)
        fix = np.linspace(0, 3, 20).round()
        np.random.shuffle(fix)
        em.frame_ix = torch.from_numpy(fix)
        bg = torch.rand((5, 1, 64, 64))

        return em, bg

    def test_candidate(self, pseudobg):
        em, bg = self.bg_sampler()
        bg[2] *= 100  # test whether this comes out

        pseudobg.forward(None, em, bg)
        em_c = em.get_subset_frame(2, 2)
        assert tutil.tens_almeq(em_c.bg, torch.ones_like(em_c.bg) * bg[2].mean(), 5.)
