import pytest

import deepsmlm.evaluation.metric_library as test_cand
import deepsmlm.generic.emitter as em


class TestSegmentation:

    @pytest.fixture()
    def evaluator(self):
        return test_cand.PrecisionRecallJaccard()

    test_data = [
        (em.EmptyEmitterSet(), em.EmptyEmitterSet(), em.EmptyEmitterSet(), (float('nan'), float('nan'), float('nan'),
                                                                            float('nan'))),
        (em.EmptyEmitterSet(), em.RandomEmitterSet(1), em.EmptyEmitterSet())
    ]

    @pytest.mark.parametrize("tp,fp,fn,expect", test_data)
    def test_segmentation(self, evaluator, tp, fp, fn, expect):
        """
        Some handcrafted values

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
            exp(tuple): expected outcome

        """
        out = evaluator.forward(tp, fp, fn)

        for o, e in zip(out, expect):  # check all the outcomes
            if math.isnan(o) or math.isnan(e):
                assert math.isnan(o)
                assert math.isnan(e)

            else:
                assert o == e
