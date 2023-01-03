import matplotlib.pyplot as plt
import pytest
import math
import torch
import itertools

from decode.evaluation import evaluation, match_emittersets
from decode.emitter import emitter as em
from decode.emitter import process
from decode.generic import test_utils


class TestEval:
    @pytest.fixture
    def evaluator(self):
        class MockEval:
            def __str__(self):
                return "Mock."

        return MockEval()

    def test_str(self, evaluator):
        assert evaluator.__str__() is not None
        assert len(evaluator.__str__()) != 0


class TestDetectionEvaluation(TestEval):
    @pytest.fixture
    def evaluator(self):
        return evaluation.DetectionEvaluation()

    test_data = [
        (
            em.factory(0),
            em.factory(0),
            em.factory(0),
            (float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, 0),
        ),
        (
            em.factory(0),
            em.factory(1),
            em.factory(0),
            (0.0, float("nan"), 0.0, float("nan"), 0, 1, 0),
        ),
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

        for o, e in itertools.zip_longest(out, expect, fillvalue=None):  # check all the outcomes
            if math.isnan(o) or math.isnan(e):
                assert math.isnan(o)
                assert math.isnan(e)

            else:
                assert o == e


class TestDistanceEvaluation(TestEval):
    @pytest.fixture
    def evaluator(self):
        return evaluation.DistanceEvaluation()

    test_data = [
        (
            em.factory(0, xy_unit="nm"),
            em.factory(0, xy_unit="nm"),
            (float("nan"),) * 6,
        )
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
            evaluator.forward(em.factory(0, xy_unit="nm"), em.factory(1, xy_unit="nm"))


class TestWeightedErrors(TestEval):
    @pytest.fixture(params=["phot", "crlb"])
    def evaluator(self, request):
        return evaluation.WeightedErrors(mode=request.param, reduction="mstd")

    # one mode of paremtr. should not lead to an error because
    # than the reduction type is also checked
    @pytest.mark.parametrize("mode", [None, "abc", "phot"])
    @pytest.mark.parametrize("reduction", ["None", "abc"])
    def test_sanity(self, evaluator, mode, reduction):

        """Assertions"""
        with pytest.raises(ValueError):
            evaluator.__init__(mode=mode, reduction=reduction)

    def test_forward_handcrafted(self, evaluator):
        # if evaluator.mode != 'phot':
        #     return

        """Setup"""
        tp = em.EmitterSet(
            xyz=torch.zeros((4, 3)),
            phot=torch.tensor([1050.0, 1950.0, 3050.0, 4050]),
            frame_ix=torch.tensor([0, 0, 1, 2]),
            bg=torch.ones((4,)) * 10,
            xy_unit="px",
            px_size=(127.0, 117.0),
        )

        ref = tp.clone()
        ref.xyz += 0.5
        ref.phot = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])
        ref.xyz_cr = (
            torch.tensor(
                [[10.0, 10.0, 15], [8.0, 8.0, 10], [6.0, 6.0, 7], [4.0, 4.0, 5.0]]
            )
            / 100.0
        ) ** 2
        ref.phot_cr = torch.tensor([10.0, 12.0, 14.0, 16.0]) ** 2
        ref.bg_cr = torch.tensor([1.0, 2.0, 3.0, 4]) ** 2

        _, _, _, dpos, dphot, dbg = evaluator.forward(
            tp, ref
        )  # test only on non reduced values

        assert (
            dpos.abs().argsort(0) == torch.arange(4).unsqueeze(1).repeat(1, 3)
        ).all(), ("Weighted error for pos." "should be monot. increasing")

        assert (dphot.abs().argsort(descending=True) == torch.arange(4)).all(), (
            "Weighted error for photon should be " "monot. decreasing"
        )
        assert (dbg.abs().argsort(descending=True) == torch.arange(4)).all(), (
            "Weighted error for background should be " "monot. decreasing"
        )

    data_forward_sanity = [
        (
            em.factory(0, xy_unit="nm"),
            em.factory(0, xy_unit="nm"),
            False,
            (torch.empty((0, 3)), torch.empty((0,)), torch.empty((0,))),
        ),
        (em.factory(5), em.factory(0), True, None),
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

        if expt_err:
            with pytest.raises(ValueError):
                _ = evaluator.forward(tp, ref)
            return
        else:
            out = evaluator.forward(tp, ref)

        assert isinstance(out, evaluator._return), "Wrong output type"
        # test only the non-reduced outputs
        for out_i, expt_i in zip(out[3:], expt_out):
            assert test_utils.tens_almeq(out_i, expt_i, 1e-4)

    def test_reduction(self, evaluator):
        # mean and std
        dxyz, dphot, dbg = (
            torch.randn((250000, 3)),
            torch.randn(250000) + 20,
            torch.rand(250000),
        )
        dxyz_, dphot_, dbg_ = evaluator._reduce(dxyz, dphot, dbg, "mstd")

        assert test_utils.tens_almeq(dxyz_[0], torch.zeros((3,)), 1e-2)
        assert test_utils.tens_almeq(dxyz_[1], torch.ones((3,)), 1e-2)

        assert test_utils.tens_almeq(dphot_[0], torch.zeros((1,)) + 20, 1e-2)
        assert test_utils.tens_almeq(dphot_[1], torch.ones((1,)), 1e-2)

        assert test_utils.tens_almeq(dbg_[0], torch.zeros((1,)) + 0.5, 1e-2)
        assert test_utils.tens_almeq(dbg_[1], torch.ones((1,)) * 0.2889, 1e-2)

        # gaussian fit
        dxyz, dphot, dbg = (
            torch.randn((250000, 3)),
            torch.randn(250000) + 20,
            torch.randn(250000),
        )
        dxyz_, dphot_, dbg_ = evaluator._reduce(dxyz, dphot, dbg, "gaussian")

        assert test_utils.tens_almeq(dxyz_[0], torch.zeros((3,)), 1e-2)
        assert test_utils.tens_almeq(dxyz_[1], torch.ones((3,)), 1e-2)

        assert test_utils.tens_almeq(dphot_[0], torch.zeros((1,)) + 20, 1e-2)
        assert test_utils.tens_almeq(dphot_[1], torch.ones((1,)), 1e-2)

        assert test_utils.tens_almeq(dbg_[0], torch.zeros((1,)), 1e-2)
        assert test_utils.tens_almeq(dbg_[1], torch.ones((1,)), 1e-2)

    plot_test_data = [
        (torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0, 3))),
        (torch.randn((25000, 3)), torch.randn(25000), torch.randn(25000)),
    ]

    plot_test_axes = [None, plt.subplots(5)[1]]

    @pytest.mark.plot
    @pytest.mark.parametrize("dxyz,dphot,dbg", plot_test_data)
    @pytest.mark.parametrize("axes", plot_test_axes)
    def test_plot_hist(self, evaluator, dxyz, dphot, dbg, axes):
        axes = evaluator.plot_error(dxyz, dphot, dbg, axes=axes)
        plt.show()


class TestSMLMEval:
    @pytest.fixture
    def evaluator(self):
        return evaluation.SMLMEvaluation()

    def test_result_dictablet(self, evaluator):

        result = evaluator.forward(
            em.factory(10, xy_unit="nm"),
            em.factory(1, xy_unit="nm"),
            em.factory(2, xy_unit="nm"),
            em.factory(10, xy_unit="nm"),
        )

        assert isinstance(result._asdict(), dict)
        assert result.prec == result._asdict()["prec"]

    def test_descriptors(self, evaluator):
        descriptors = {
            "prec": "Precision",
            "rec": "Recall",
            "jac": "Jaccard Index",
            "rmse_lat": "RMSE lateral",
            "rmse_ax": "RMSE axial",
            "rmse_vol": "RMSE volumetric",
            "mad_lat": "Mean average distance lateral",
            "mad_ax": "Mean average distance axial",
            "mad_vol": "Mean average distance in 3D",
            "dx_red_sig": "CRLB normalised error in x",
            "dy_red_sig": "CRLB normalised error in y",
            "dz_red_sig": "CRLB normalised error in z",
            "dx_red_mu": "CRLB normalised bias in x",
            "dy_red_mu": "CRLB normalised bias in y",
            "dz_red_mu": "CRLB normalised bias in z",
        }

        assert isinstance(evaluator.descriptors, dict)
        assert evaluator.descriptors == descriptors


def test_evaluation_smlm():
    matcher = match_emittersets.GreedyHungarianMatching(
        match_dims=2, dist_ax=200.0, dist_lat=100, dist_vol=None
    )
    em_filter = process.EmitterFilterGeneric(prob=lambda p: p >= 0.5)
    e = evaluation.EvaluationSMLM(matcher=matcher, em_filter=em_filter)

    em_out = em.factory(
        6, prob=torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), xy_unit="nm"
    )
    em_tar = em.factory(xyz=torch.tensor([[1.0, 2.0, 3.0]]), xy_unit="nm")

    out = e.forward(em_out, em_tar)

    assert "prec" in out
    assert "rmse_lat" in out
