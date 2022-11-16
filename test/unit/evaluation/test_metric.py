import pytest
import math
import torch

from decode.evaluation import metric

rmse_mad_testdata = [
    # xyz, xyz_ref, (rmse_lat, rmse_ax, rmse_vol, mad_lat, mad_ax, mad_vol)
    (torch.zeros((0, 3)), torch.zeros((0, 3)), (float("nan"),) * 6),  # nothing
    (
        torch.tensor([[2.0, 0.0, 0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        (2.0, 0.0, 2.0, 2.0, 0.0, 2.0),
    ),
    (
        torch.tensor([[5.0, 6.0, 0.0]]),
        torch.tensor([[2.0, 2.0, 0.0]]),
        (5.0, 0.0, 5.0, 7.0, 0.0, 7.0),
    ),
]


@pytest.mark.parametrize("xyz_tp,xyz_gt,expect", rmse_mad_testdata)
def test_rmse_mad(xyz_tp, xyz_gt, expect):

    out_rmse = metric.rmse(xyz_tp, xyz_gt)
    out_mad = metric.mad(xyz_tp, xyz_gt)
    out = torch.cat((torch.tensor(out_rmse), torch.tensor(out_mad)))

    for o, e in zip(out, expect):
        # if at least either outcome or expect is nan check if the other is as well
        if math.isnan(o) or math.isnan(e):
            assert math.isnan(o)
            assert math.isnan(e)
        else:
            assert o == e


@pytest.mark.parametrize("metric_impl", ["rmse", "mad"])
def test_rmse_mad_exceptions(metric_impl):
    with pytest.raises(ValueError):  # inconsistent number of points
        getattr(metric, metric_impl)(torch.zeros((0, 3)), torch.zeros((2, 3)))

    with pytest.raises(NotImplementedError):  # not supported dim
        getattr(metric, metric_impl)(torch.zeros((2, 4)), torch.zeros((2, 4)))


@pytest.mark.parametrize(
    "tp,fp,expct",
    [
        (0, 0, float("nan")),
        (1, 0, 1.0),
        (0, 1, 0.0),
    ],
)
def test_precision(tp, fp, expct):
    out = metric.precision(tp, fp)
    assert out == expct if not math.isnan(expct) else math.isnan(out)


@pytest.mark.parametrize(
    "tp,fn,expct",
    [
        (0, 0, float("nan")),  # case 1
        (1, 0, 1.0),  # case 2
        (0, 0, float("nan")),
    ],
)
def test_recall(tp, fn, expct):
    out = metric.recall(tp, fn)
    assert out == expct if not math.isnan(expct) else math.isnan(out)


@pytest.mark.parametrize(
    "tp,fp,fn,expct",
    [
        (0, 0, 0, float("nan")),
        (1, 0, 0, 1.0),
        (0, 1, 0, 0),
    ],
)
def test_jaccard(tp, fp, fn, expct):
    out = metric.jaccard(tp, fp, fn)
    assert out == expct if not math.isnan(expct) else math.isnan(out)


@pytest.mark.parametrize("tp,fp,fn,expct", [
        (0, 0, 0, float("nan")),
        (1, 0, 0, 1.),
        (0, 1, 0, float("nan")),
    ])
def test_f1(tp, fp, fn, expct):
    out = metric.f1(tp, fp, fn)
    assert out == expct if not math.isnan(expct) else math.isnan(out)


def test_efficiency():
    out = metric.efficiency(0.91204, 32.077, 1.0)
    assert out == pytest.approx(0.66739, rel=1e-3)
