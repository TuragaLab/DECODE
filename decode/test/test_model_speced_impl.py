import pytest
import torch
import copy

from decode.neuralfitter.models import model_speced_impl as model_impl


class TestSigmaMUNet:

    @pytest.fixture()
    def model(self):
        return model_impl.SigmaMUNet(3, depth_shared=2, depth_union=2, initial_features=48, inter_features=48)

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

    def test_custom_init(self, model):
        """Tests whether the custom weight init works, rudimentary test, which asserts that all weights were touched."""

        model_old = copy.deepcopy(model)
        model = model.apply(model.weight_init)

        """Assertions"""
        for mod, mod_old in zip(model.named_parameters(), model_old.named_parameters()):
            if mod[0][-6:] == 'weight':
                assert (mod[1] != mod_old[1]).all()
