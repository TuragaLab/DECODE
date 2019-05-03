import torch
from unittest import TestCase

from deepsmlm.neuralfitter.models.model_offset import OffsetUnet


class TestOfsetUnet(TestCase):
    def setUp(self) -> None:
        self.net = OffsetUnet(3)
        self.loss = torch.nn.MSELoss()

    def test_forward(self):
        x = torch.rand((32, 3, 64, 64), requires_grad=True)
        gt = torch.rand((32, 5, 64, 64))

        out = self.net.forward(x)
        dummyloss = self.loss(out, gt)
        dummyloss.backward()
        self.assertTrue(x.grad is not None, "Backprop.")
