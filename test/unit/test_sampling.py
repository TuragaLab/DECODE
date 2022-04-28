import torch

from decode.neuralfitter import sampling


def test_sample_tensor():
    x = torch.meshgrid(torch.arange(5), torch.arange(5))[0]

    out = sampling.sample_crop(x, (2, 3, 4))

    assert isinstance(out, torch.Tensor)
    assert out.size() == torch.Size([2, 3, 4])
