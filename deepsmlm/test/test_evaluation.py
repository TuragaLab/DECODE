import pytest
import math

import deepsmlm.evaluation.evaluation as test_cand
import deepsmlm.generic.emitter as em


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
        return test_cand.SegmentationEvaluation()

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
        return test_cand.DistanceEvaluation()

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

