import pytest
import torch
import copy

from decode.neuralfitter.models import model_speced_impl as model_impl
from decode.generic import test_utils


class TestSigmaMUNet:
    @pytest.fixture()
    def model(self):
        return model_impl.SigmaMUNet(
            ch_in_map=[[0], [1], [2]],
            depth_shared=1,
            depth_union=1,
            initial_features=8,
            inter_features=8
        )

    def test_forward(self, model):
        x = torch.rand((2, 3, 64, 64))

        out = model.forward(x)

        assert out.dim() == 4
        assert out.size(1) == 10

    def test_backward(self, model):
        x = torch.rand((2, 3, 64, 64))
        loss = torch.nn.MSELoss()

        out = model.forward(x)
        loss(out, torch.rand_like(out)).backward()

    def test_custom_init(self, model):
        # tests whether the custom weight init works, rudimentary test, which asserts
        # that all weights were touched

        model_old = copy.deepcopy(model)
        model = model.apply(model.weight_init)

        assert not test_utils.same_weights(model_old, model)
