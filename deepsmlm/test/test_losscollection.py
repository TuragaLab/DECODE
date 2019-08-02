from unittest import TestCase
import torch
import pytest
import matplotlib.pyplot as plt

import deepsmlm.test.utils_ci as tutil


from deepsmlm.generic.emitter import EmitterSet, CoordinateOnlyEmitter, RandomEmitterSet
import deepsmlm.neuralfitter.losscollection as loss


class TestFocalVoronoiPointLoss:

    @pytest.fixture(scope='class')
    def fvp_loss(self):
        return loss.FocalVoronoiPointLoss(0.012, 0.9)

    def test_run(self, fvp_loss):
        gt = torch.zeros((2, 5, 32, 32))
        gt[0, 0, 0, 0] = 1.

        prediction = gt.clone()
        prediction[0, 0, 0, 0] = 0.01

        loss = fvp_loss(prediction, gt)

        assert True
