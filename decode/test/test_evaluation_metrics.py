import pytest
import math
import torch

import decode.evaluation.metric as test_cand


class TestRootMeanAbsoluteDist:

    rmse_mad_testdata = [
        (torch.zeros((0, 3)), torch.zeros((0, 3)), (float('nan'), ) * 6),  # nothing
        (torch.tensor([[2., 0., 0]]), torch.tensor([[0., 0., 0.]]), (2., 0., 2., 2., 0., 2.)),
        (torch.tensor([[5., 6., 0.]]), torch.tensor([[2., 2., 0.]]), (5., 0., 5., 7., 0., 7.))
    ]

    @pytest.mark.parametrize("xyz_tp,xyz_gt,expect", rmse_mad_testdata)
    def test_rmse_mad(self, xyz_tp, xyz_gt, expect):

        out = test_cand.rmse_mad_dist(xyz_tp, xyz_gt)

        for o, e in zip(out, expect):  # check all the outcomes
            if math.isnan(o) or math.isnan(e):  # if at least either outcome or expect is nan check if the other is as well
                assert math.isnan(o)
                assert math.isnan(e)

            else:
                assert o == e

    def test_excpt(self):
        """Exceptions"""

        with pytest.raises(ValueError):
            test_cand.rmse_mad_dist(torch.zeros((0, 3)), torch.zeros((2, 3)))

        with pytest.raises(ValueError):
            test_cand.rmse_mad_dist(torch.zeros((2, 4)), torch.zeros((2, 4)))


class TestPrecisionRecallJaccard:

    test_data = [
        (0, 0, 0, (float('nan'), ) * 4),
        (1, 0, 0, (1., 1., 1., 1.)),
        (0, 1, 0, (0, float('nan'), 0, float('nan')))
    ]

    @pytest.mark.parametrize("tp,fp,fn,expect", test_data)
    def test_prec_rec(self, tp, fp, fn, expect):

        out = test_cand.precision_recall_jaccard(tp, fp, fn)

        for o, e in zip(out, expect):  # check all the outcomes
            if math.isnan(o) or math.isnan(e):  # if at least either outcome or expect is nan check if the other is as well
                assert math.isnan(o)
                assert math.isnan(e)

            else:
                assert o == e


def test_efficiency():
    out = test_cand.efficiency(0.91204, 32.077, 1.0)
    assert out == pytest.approx(0.66739, rel=1e-3)

