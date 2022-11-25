from abc import abstractmethod
from typing import Any, Callable, Union, Sequence, Optional, Protocol

import torch
from itertools import zip_longest

from ... import emitter
from ...simulation import camera, microscope


class ModelInput(Protocol):
    @abstractmethod
    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class ModelInputINeedToFindAName(ModelInput):
    def __init__(
        self,
        noise: Optional[Union[camera.Camera, Sequence[camera.Camera]]],
        cat_frame: Optional[Callable],
        cat_input: Optional[Callable],
    ):
        self._noise = (
            [noise] if noise is not None and not isinstance(noise, Sequence) else noise
        )
        self._cat_frame = cat_frame
        self._cat_input = cat_input

    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:

        if bg is not None:
            bg = [bg] if not isinstance(bg, Sequence) else bg
            frames = [f + bg_ for f, bg_ in zip_longest()]

        if self._noise is not None:
            frame = [n.forward(f) for n, f in zip_longest(self._noise, frame)]

        frame = self._cat_frame(frame) if self._cat_frame is not None else frame

        frame = self._cat_input(frame, aux)


class MergerFrameBg:
    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    ):
        x = frame

        if bg is not None:
            x = frame + bg

        return x
