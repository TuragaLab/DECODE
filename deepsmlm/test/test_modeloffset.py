import torch
import pytest

from deepsmlm.neuralfitter.models.model_offset import OffsetUnet
import deepsmlm.test.utils_ci as tutil


class TestOffsetUnet:
    @pytest.fixture(scope='class')
    def net(self):
        return OffsetUnet(3)

    @pytest.fixture(scope='class')
    def loss(self):
        return torch.nn.MSELoss()

    def test_forward(self, net, loss):
        x = torch.rand((2, 3, 64, 64), requires_grad=True)
        gt = torch.rand((2, 5, 64, 64))

        out = net.forward(x)
        dummyloss = loss(out, gt)
        dummyloss.backward()
        assert x.grad is not None

    def test_eval(self, net, loss):
        """In Eval mode, a sigmoid should be applied"""
        x = torch.rand((2, 3, 64, 64))
        net.train()
        out_tr = net(x)

        net.eval()
        out_eval = net(x)

        assert (not tutil.tens_almeq(out_tr, out_eval, 0.1))
        assert (out_eval[:, 0] >= 0.).all() and (out_eval[:, 0] <= 1.).all()
