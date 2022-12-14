from abc import abstractmethod
from itertools import zip_longest
from typing import Any, Callable, Union, Sequence, Optional, Protocol

import torch

from ... import emitter
from ...simulation import camera, background


class ModelInput(Protocol):
    @abstractmethod
    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class ModelInputPostponed(ModelInput):
    def __init__(
        self,
        cam: Optional[Union[camera.Camera, Sequence[camera.Camera]]],
        cat_input: Optional[Callable] = None,
        merger_bg: Optional[background.Merger] = None
    ):
        """
        Prepares model's input data by combining with background, applying noise
        and merge everything together.

        Args:
            cam: camera module
            cat_input: optional callable that combines frames and auxiliary
            merger_bg: optional background merger
        """

        # make it a list because this allows for zipping later on
        self._noise: Optional[list[cam.Camera]] = (
            [cam] if cam is not None and not isinstance(cam, Sequence) else cam
        )
        self._merger_bg = background.Merger() if merger_bg is None else merger_bg
        self._cat_input_impl = cat_input

    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:

        if bg is not None:
            frame = self._merger_bg.forward(frame=frame, bg=bg)

        if self._noise is not None:
            frame = [n.forward(f) for n, f in zip_longest(self._noise, frame)]

        # list of channels and auxiliary to frame tensor
        frame = self._cat_input(frame, aux) if self._cat_input is not None else frame

        return frame

    def _cat_input(
        self,
        frame: Optional[Union[torch.Tensor, Sequence]],
        aux: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._cat_input_impl is not None:
            return self._cat_input_impl(frame, aux)

        frame = torch.stack(frame, -3) if isinstance(frame, Sequence) else frame
        if aux is not None:
            frame = torch.cat(
                [frame, torch.stack(aux) if isinstance(aux, Sequence) else aux], -3
            )

        return frame
