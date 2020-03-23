from collections import namedtuple
import pytest
import math
import torch

from deepsmlm.evaluation import evaluation
from deepsmlm.generic import emitter as em
from deepsmlm.generic.utils import test_utils


class TestEval:

    @pytest.fixture()
    def evaluator(self):
        class MockEval:
            def __str__(self):
                return "Mock."

        return MockEval()

    def test_str(self, evaluator):
        assert evaluator.__str__() is not None
        assert len(evaluator.__str__()) != 0


class TestSegmentationEval(TestEval):

    @pytest.fixture()
    def evaluator(self):
        return evaluation.SegmentationEvaluation()

    test_data = [
        (em.EmptyEmitterSet(), em.EmptyEmitterSet(), em.EmptyEmitterSet(), (float('nan'), ) * 4),
        (em.EmptyEmitterSet(), em.RandomEmitterSet(1), em.EmptyEmitterSet(), (0., float('nan'), 0., float('nan')))
    ]

    @pytest.mark.parametrize("tp,fp,fn,expect", test_data)
    def test_segmentation(self, evaluator, tp, fp, fn, expect):
        """
        Some handcrafted values

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
            expect(tuple): expected outcome

        """
        out = evaluator.forward(tp, fp, fn)

        for o, e in zip(out, expect):  # check all the outcomes
            if math.isnan(o) or math.isnan(e):
                assert math.isnan(o)
                assert math.isnan(e)

            else:
                assert o == e


class TestDistanceEval(TestEval):

    @pytest.fixture()
    def evaluator(self):
        return evaluation.DistanceEvaluation()

    test_data = [
        (em.EmptyEmitterSet(), em.EmptyEmitterSet(), (float('nan'), ) * 6)
    ]

    @pytest.mark.parametrize("tp,tp_match,expect", test_data)
    def test_distance(self, evaluator, tp, tp_match, expect):
        """

        Args:
            evaluator: fixture
            tp: true positives
            tp_match: matching ground truth
            expect: expected outcome

        """
        out = evaluator.forward(tp, tp_match)

        for o, e in zip(out, expect):  # check all the outcomes
            if math.isnan(o) or math.isnan(e):
                assert math.isnan(o)
                assert math.isnan(e)

            else:
                assert o == e

    def test_distance_excpt(self, evaluator):
        """

        Args:
            evaluator:

        """
        with pytest.raises(ValueError):
            evaluator.forward(em.EmptyEmitterSet(), em.RandomEmitterSet(1))


class TestWeightedErrors(TestEval):

    @pytest.fixture(params=['phot', 'crlb'])
    def evaluator(self, request):
        return evaluation.WeightedErrors(mode=request.param)

    def test_sanity(self, evaluator):

        """Assertions"""
        with pytest.raises(ValueError):
            evaluator.__init__(mode='abc')

        with pytest.raises(ValueError):
            evaluator.__init__(mode=None)

    def test_forward_handcrafted(self, evaluator):
        # if evaluator.mode != 'phot':
        #     return

        """Setup"""
        tp = em.EmitterSet(xyz=torch.zeros((4, 3)), phot=torch.tensor([1050., 1950., 3050., 4050]),
                           frame_ix=torch.tensor([0, 0, 1, 2]), bg=torch.ones((4, )) * 10,
                           xy_unit='px', px_size=(127., 117.))

        ref = tp.clone()
        ref.xyz += 0.5
        ref.phot = torch.tensor([1000., 2000., 3000., 4000.])
        ref.xyz_cr = (torch.tensor([[10., 10., 15], [8., 8., 10], [6., 6., 7], [4., 4., 5.]]) / 100. )** 2
        ref.phot_cr = torch.tensor([10., 12., 14., 16.]) ** 2
        ref.bg_cr = torch.tensor([1., 2., 3., 4]) ** 2

        """Run"""
        dpos, dphot, dbg = evaluator.forward(tp, ref)

        """Assertions"""
        assert (dpos.abs().argsort(0) == torch.arange(4).unsqueeze(1).repeat(1, 3)).all(), "Weighted error for pos." \
                                                                                           "should be monot. increasing"

        assert (dphot.abs().argsort(descending=True) == torch.arange(4)).all(), "Weighted error for photon should be " \
                                                                                "monot. decreasing"

        assert (dbg.abs().argsort(descending=True) == torch.arange(4)).all(), "Weighted error for background should be " \
                                                                                "monot. decreasing"

    data_forward_sanity = [
        (em.EmptyEmitterSet(), em.EmptyEmitterSet(), False, (torch.empty((0, 3)), torch.empty((0, )), torch.empty((0, )))),
        (em.RandomEmitterSet(5), em.EmptyEmitterSet(), True, None)
    ]

    @pytest.mark.parametrize("tp,ref,expt_err,expt_out", data_forward_sanity)
    def test_forward_sanity(self, evaluator, tp, ref, expt_err, expt_out):
        """
        General forward sanity checks.
            1. Both empty sets of emitters
            2. Unequal size

        """

        if expt_err and expt_out is not None:
            raise RuntimeError("Wrong test setup.")

        """Run"""
        if expt_err:
            with pytest.raises(ValueError):
                _ = evaluator.forward(tp, ref)
            return

        else:
            out = evaluator.forward(tp, ref)

        """Assertions"""
        assert isinstance(out, evaluator._return), "Wrong output type"
        for out_i, expt_i in zip(out, expt_out):
            assert test_utils.tens_almeq(out_i, expt_i, 1e-4)
