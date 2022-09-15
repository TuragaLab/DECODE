import torch
import pytest

from decode.emitter import emitter
from decode.neuralfitter import post_processing


class ModelSpatial(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.dim() == 4

        return x / x.max()


class Loss(torch.nn.MSELoss):
    def forward(self, y_out: torch.Tensor, y_tar: torch.Tensor) -> torch.Tensor:
        assert isinstance(y_out, torch.Tensor)
        assert isinstance(y_tar, torch.Tensor)
        return super().forward(y_out, y_out)


class PostProcessing(post_processing.PostProcessing):
    # fake post processing that outputs random emitters. for a quick proof we output
    # them on the frame indices 0 ... N (N batch-dim of x), and in numbers of N * H * W
    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
        assert isinstance(x, torch.Tensor)

        n = len(x) * x.size(-1) * x.size(-2)
        em = emitter.factory(frame_ix=torch.randint(len(x), size=(n,)))

        return em
