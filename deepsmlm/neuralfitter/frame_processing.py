from abc import ABC, abstractmethod
import torch

from typing import Tuple, List


class FrameProcessing(ABC):

    @abstractmethod
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Forward frame through processing implementation.

        Args:
            frame:

        """
        raise NotImplementedError


class Mirror2D(FrameProcessing):

    def __init__(self, dims: Tuple):
        """
        Mirror the specified dimensions. Providing dim index in negative format is recommended.
        Given format N x C x H x W and you want to mirror H and W set dims=(-2, -1).

        Args:
            dims: dimensions

        """
        super().__init__()

        self.dims = dims

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.flip(self.dims)
