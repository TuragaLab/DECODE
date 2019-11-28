import matplotlib.pyplot as plt
import pytest
import torch

import deepsmlm.evaluation.match_emittersets
import deepsmlm.generic.emitter
import deepsmlm.evaluation.evaluation as evaluation
import deepsmlm.test.utils_ci as tutil


class TestCumulantMeter:

    def test_cumulation(self):
        m = evaluation.CumulantMeter()
        m.update(torch.rand(10000))
        m.update(torch.rand(10000))
        assert 0.5 == pytest.approx(m.avg, 1e-2)

    def test_histogram(self):
        m = evaluation.CumulantMeter()
        m.update(torch.rand(10000))
        m.update(torch.rand(10000))
        f = m.hist()
        plt.show()
        assert True


class TestGreedyMatching:

    @pytest.fixture(scope='class')
    def matcher(self):
        return deepsmlm.evaluation.match_emittersets.GreedyHungarianMatching(dist_lat=100.)

    @pytest.fixture(scope='class')
    def set_1ref_4pred(self):
        ref = deepsmlm.generic.emitter.CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.]]))
        ref.id[0] = 42
        pred = deepsmlm.generic.emitter.CoordinateOnlyEmitter(
            torch.tensor([[-50, -50, 0.], [60, -50, 0.], [-40, 50, 0.], [35, 35, 0.]]))

        return [pred, ref]

    @pytest.fixture(scope='class')
    def set_4ref_1pred(self):
        pred = deepsmlm.generic.emitter.CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.]]))
        ref = deepsmlm.generic.emitter.CoordinateOnlyEmitter(
            torch.tensor([[-50, -50, 0.], [60, -50, 0.], [-40, 50, 0.], [35, 35, 0.]]))
        ref.id[-1] = 42

        return [pred, ref]

    def test_oneref_npred(self, matcher, set_1ref_4pred):
        pred, ref = set_1ref_4pred[0], set_1ref_4pred[1]
        tp, fp, fn, tp_match = matcher.forward(pred, ref)

        tutil.tens_almeq(tp.xyz, torch.tensor([35., 35., 0]))
        tutil.tens_almeq(tp.id, tp_match.id)
        assert tp_match == ref  # match must be the ground truth

    def test_oneref_mulpred(self, matcher, set_4ref_1pred):
        pred, ref = set_4ref_1pred[0], set_4ref_1pred[1]
        tp, fp, fn, tp_match = matcher.forward(pred, ref)

        assert tp == pred
        tutil.tens_almeq(tp.id, tp_match.id)
        tutil.tens_almeq(tp.xyz, torch.tensor([35., 35., 0]))



