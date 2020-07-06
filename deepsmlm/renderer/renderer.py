from abc import ABC

import torch

from ..generic import emitter


class Renderer(ABC):

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:
        raise NotImplementedError
