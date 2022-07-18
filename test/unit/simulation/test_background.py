import pytest
import torch

import decode.simulation.background as background
from decode.emitter import emitter
from decode.generic import test_utils


@pytest.fixture
def bg_uniform():
    return background.UniformBackground((1., 100.), size=(43, 74))


@pytest.mark.parametrize("bg", [
    "bg_uniform",
])
@pytest.mark.parametrize("size,size_expct", [
    ((1, 2), (1, 2)),
    (None, (43, 74))
])
@pytest.mark.parametrize("device,device_expct", [
    ("cpu", "cpu"),
    ("cuda", "cuda"),
    (None, "cpu")
])
def test_bg_overwrites(size, size_expct, device, device_expct, bg, request):
    bg = request.getfixturevalue(bg)

    size_out, device_out = bg._arg_defaults(size, device)
    assert size_out == size_expct
    assert device_out == device_expct


@pytest.mark.parametrize("size,size_expct", [
    ((31, 32), (31, 32)),
    (None, (43, 74))
])
@pytest.mark.parametrize("bg", [
    "bg_uniform",
])
def test_sample_size(size, size_expct, bg, request):
    bg = request.getfixturevalue(bg)
    sample = bg.sample(size=size)

    assert sample.size() == torch.Size(size_expct)


@pytest.mark.parametrize("bg", [
    "bg_uniform",
])
def test_sample_like(bg, request):
    bg = request.getfixturevalue(bg)

    x = torch.rand(42, 43)
    sample = bg.sample_like(torch.rand(42, 43))

    assert sample.size() == x.size()


def test_bg_uniform(bg_uniform):
    out = bg_uniform.sample((5, 32, 32))

    assert (
        len(out.unique()) == 5
    ), "Should have as many unique values as we have batch size."

    for out_c in out:
        assert len(out_c.unique()) == 1, "Background should constant per batch element"
    assert ((out >= 0) * (out <= 100)).all(), "Wrong output values."


class TestBgPerEmitterFromBgFrame:
    @pytest.fixture(scope="class")
    def extractor(self):
        return background.BgPerEmitterFromBgFrame(
            17, (-0.5, 63.5), (-0.5, 63.5), (64, 64)
        )

    def test_mean_filter(self, extractor):
        x_in = []
        x_in.append(torch.randn((1, 1, 64, 64)))
        x_in.append(torch.zeros((1, 1, 64, 64)))
        x_in.append(
            torch.meshgrid(torch.arange(64), torch.arange(64))[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        # excpt outcome
        expect = []
        expect.append(torch.zeros_like(x_in[0]))
        expect.append(torch.zeros_like(x_in[0]))
        expect.append(8)

        out = []
        for x in x_in:
            out.append(extractor._mean_filter(x))

        assert test_utils.tens_almeq(out[0], expect[0], 1)  # 10 sigma
        assert test_utils.tens_almeq(out[1], expect[1])
        assert test_utils.tens_almeq(
            out[2][0, 0, 8, :], 8 * torch.ones_like(out[2][0, 0, 0, :]), 1e-4
        )

    test_data = [
        (torch.zeros((1, 1, 64, 64)), emitter.factory(100), torch.zeros((100,))),
        (
            torch.meshgrid(torch.arange(64), torch.arange(64))[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            emitter.CoordinateOnlyEmitter(torch.tensor([[8.0, 0.0, 0.0]])),
            torch.tensor([8.0]),
        ),
        (
            torch.rand((1, 1, 64, 64)),
            emitter.CoordinateOnlyEmitter(torch.tensor([[70.0, 32.0, 0.0]])),
            torch.tensor([float("nan")]),
        ),
    ]

    @pytest.mark.parametrize("bg,em,expect_bg", test_data)
    def test_forward(self, extractor, bg, em, expect_bg):
        out = extractor.forward(em, bg)

        assert test_utils.tens_almeq(out.bg, expect_bg, 1e-4, nan=True)
