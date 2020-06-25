import pytest
import torch

from deepsmlm.neuralfitter.models import model_speced_impl as model_impl


class TestSigmaMUNet:

    @pytest.fixture()
    def model(self):
        return model_impl.SigmaMUNet(3, 2, 2, 48, 48)

    def test_forward(self, model):

        """Setup"""
        x = torch.rand((2, 3, 64, 64))

        """Run"""
        out = model.forward(x)

        """Setup"""
        assert out.dim() == 4
        assert out.size(1) == 10

    def test_backward(self, model):

        """Setup"""
        x = torch.rand((2, 3, 64, 64))
        loss = torch.nn.MSELoss()

        """Run"""
        out = model.forward(x)
        loss(out, torch.rand_like(out)).backward()
