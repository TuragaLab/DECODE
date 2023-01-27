from unittest import mock

import pytest
import torch
import copy

from decode.neuralfitter import spec
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
            inter_features=8,
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


def test_sigma_semantic_multi():
    m = model_impl.SigmaMUNet(
        ch_in_map=[[0, 3, 4], [1, 3, 4], [2, 3, 4]],
        ch_out_heads=(3, 4, 4, 1),
        depth_shared=1,
        depth_union=1,
        initial_features=8,
        inter_features=8,
    )

    x = torch.rand(2, 5, 64, 64)
    out = m.forward(x)

    assert out.size() == (2, 12, 64, 64)


@pytest.mark.parametrize("n_codes", [1, 2, 3])
@pytest.mark.parametrize(
    "ix_name,min,max,mean,mean_tol",
    [
        ("ix_prob", 0, 1, 0.5, 0.2),
        ("ix_phot", 0, 1, 0.5, 0.2),
        ("ix_xyz", -1, 1, 0, 0.5),
        ("ix_sig", 0, 3, 1.5, 0.6),
        ("ix_bg", 0, 1, 0.5, 0.2),
    ],
)
def test_sigma_output_ranges(ix_name, min, max, mean, mean_tol, n_codes):
    # this is to test the output ranges of SigmaMUNet

    ch_map = spec.ModelChannelMapGMM(n_codes=n_codes)
    inter_feat = 64
    m = model_impl.SigmaMUNet(
        ch_in_map=[[0], [1], [2]],
        ch_map=ch_map,
        ch_out_heads=(n_codes, 4, 4, n_codes),
        depth_shared=1,
        depth_union=1,
        initial_features=8,
        inter_features=inter_feat,
    )
    # mok core and heads
    m._forward_core = mock.MagicMock()
    m._forward_core.return_value = torch.rand(2, inter_feat, 64, 64)
    m.mt_heads = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])

    out = m.forward(torch.rand(2, 3, 64, 64))
    out = out.detach()

    assert (
        min
        <= out[:, getattr(ch_map, ix_name)].min()
        < out[:, getattr(ch_map, ix_name)].max()
        <= max
    )
    assert out[:, getattr(ch_map, ix_name)].mean() == pytest.approx(mean, abs=mean_tol)
