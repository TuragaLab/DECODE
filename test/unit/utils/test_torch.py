import pytest
import torch

from decode.utils import torch as torchutils


def test_itemized_dist():
    dist = torchutils.ItemizedDist(torch.distributions.Uniform(-5., 5.))
    s = dist.sample()

    assert isinstance(s, float)
