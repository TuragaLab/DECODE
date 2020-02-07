import pytest
import torch

import deepsmlm.generic.emitter as emitter
import deepsmlm.evaluation.metric_library as metr
import deepsmlm.generic.utils.test_utils as tutil


def test_rmse_mad():
    rmse_eval = metr.RMSEMAD()

    out = emitter.EmptyEmitterSet()
    tar = out.clone()
    eval_out = torch.tensor(rmse_eval.forward(out, tar))
    assert torch.isnan(eval_out).all()

    out = emitter.CoordinateOnlyEmitter(torch.tensor([[0., 50., 0.]]))
    tar = emitter.CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.]]))
    eval_out = torch.tensor(rmse_eval.forward(out, tar))
    assert tutil.tens_almeq(eval_out, torch.tensor([50., 50., 0., 50., 50., 0.]))


test_efficiency_data = [
    (0.7, 25., 1., 0.6095, 0.001),
    (0.248, 40., 0.42, 0.2295, 0.001),
]


@pytest.mark.parametrize("jac, rmse, alpha, expect, delta", test_efficiency_data)
def test_efficiency(jac, rmse, alpha, expect, delta):
    assert metr.efficiency(jac, rmse, alpha) == pytest.approx(expect, delta)
