import pytest
import torch

import deepsmlm.generic.emitter as emitter
import deepsmlm.evaluation.metric_library as metr
import deepsmlm.test.utils_ci as tutil


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