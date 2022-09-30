import pytest
import torch

import decode.simulation.background as background


@pytest.fixture
def bg_uniform():
    return background.BackgroundUniform((1., 100.), size=(43, 74))


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
