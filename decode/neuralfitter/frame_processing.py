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


class AutoCenterCrop(FrameProcessing):

    def __init__(self, px_fold: int):
        """
        Automatic cropping in centre. Specify pixel_fold which the target frame size must satistfy
        and the frame will be center-cropped to this size.

        Args:
            px_fold: integer in which multiple the frame must dimensioned (H, W dimension)

        """
        super().__init__()
        self.px_fold = px_fold

        if not isinstance(self.px_fold, int):
            raise ValueError

    def forward(self, frame: torch.Tensor) -> torch.Tensor:

        # actual size
        size_is = torch.tensor(frame.size())[-2:]
        size_tar = (size_is // self.px_fold) * self.px_fold

        if (size_tar <= 0).any():
            raise ValueError

        """Crop"""
        ix_front = ((size_is - size_tar).float() / 2).ceil().long()
        ix_back = ix_front + size_tar

        return frame[..., ix_front[0]:ix_back[0], ix_front[1]:ix_back[1]]


def get_frame_extent(size, func) -> torch.Size:
    """
    Get frame extent after processing pipeline

    Args:
        size:
        func:

    Returns:

    """

    if len(size) == 4:  # avoid to forward large batches just to get the output extent
        n_batch = size[0]
        size_out = func(torch.zeros(2, *size[1:])).size()
        return torch.Size([n_batch, *size_out[1:]])

    else:
        return func(torch.zeros(*size)).size()
