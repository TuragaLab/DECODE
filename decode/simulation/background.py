from abc import ABC, abstractmethod  # abstract class
from typing import Callable, Iterable, Optional, Union, Sequence

import torch


class Background(ABC):
    def __init__(self, size: Union[tuple[int, ...], torch.Size], device: str = "cpu"):
        """
        Background
        """
        super().__init__()

        self._size = size
        self._device = device

    @abstractmethod
    def sample(
        self, size: Union[tuple[int, ...], torch.Size], device: str = "cpu"
    ) -> torch.Tensor:
        """
        Samples from background implementation in the specified size.

        Args:
            size: size of the sample
            device: from which device to sample from

        """
        raise NotImplementedError

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples background in the shape and on the device as the input.

        Args:
            x: input

        Returns:
            background sample

        """
        return self.sample(size=x.size(), device=x.device)

    def _arg_defaults(self, size: Optional[Union[tuple[int, ...], torch.Size]], device: str):
        """Overwrite optional args with instance defaults."""
        size = self._size if size is None else size
        device = self._device if device is None else device

        return size, device


class BackgroundUniform(Background):
    def __init__(
        self,
        bg: Union[float, tuple, Callable],
        size: Optional[Union[tuple[int, ...], torch.Size]] = None,
        device: str = "cpu",
    ):
        """
        Spatially constant background (i.e. a constant offset).

        Args:
            bg: background value, range or callable to sample from
            size:
            device:

        """
        super().__init__(size=size, device=device)

        if callable(bg):
            self._bg_dist = bg
        if isinstance(bg, Iterable):
            self._bg_dist = torch.distributions.uniform.Uniform(
                *bg,
            ).sample
        else:
            self._bg_dist = _get_delta_sampler(bg)

    def sample(
        self,
        size: Optional[Union[tuple[int, ...], torch.Size]] = None,
        device: str = "cpu",
    ):
        size = size if size is not None else self._size
        device = device if device is not None else self._device

        if len(size) not in (2, 3, 4):
            raise NotImplementedError("Not implemented size spec.")

        # create as many sample as there are batch-dims
        bg = self._bg_dist(
            sample_shape=[size[0]] if len(size) >= 3 else torch.Size([]),
        )

        # unsqueeze until we have enough dimensions
        if len(size) >= 3:
            bg = bg.view(-1, *((1,) * (len(size) - 1)))

        return bg.to(device) * torch.ones(size, device=device)


class Merger:
    @classmethod
    def forward(
            cls,
            frame: Union[torch.Tensor, Sequence[torch.Tensor]],
            bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    ) -> torch.Tensor:
        """
        Combine frame and background

        Args:
            frame:
            bg:

        """
        if bg is None:
            return frame

        if isinstance(frame, Sequence) != isinstance(bg, Sequence):
            raise NotImplementedError("Either none or both frame and bg must be "
                                      "Sequence.")
        elif isinstance(frame, Sequence):
            if len(frame) != len(bg):
                raise ValueError("Sequence of unequal length.")
            frame = [cls._kernel(f, bg) for f, bg in zip(frame, bg)]
        else:
            frame = cls._kernel(frame, bg)

        return frame

    @classmethod
    def _kernel(cls, frame: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        return frame + bg


def _get_delta_sampler(val: float):
    def delta_sampler(sample_shape) -> float:
        return val * torch.ones(sample_shape)

    return delta_sampler
