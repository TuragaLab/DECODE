from abc import abstractmethod
from itertools import zip_longest
from typing import Any, Callable, Union, Sequence, Optional, Protocol

import torch

from ... import emitter
from ...simulation import camera, background
from ...utils import future


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
        merger_bg: Optional[background.Merger] = None,
        scaler_frame: Optional[Callable[..., torch.Tensor]] = None,
        scaler_aux: Optional[Callable[..., torch.Tensor]] = None,
    ):
        """
        Prepares model's input data by combining with background, applying camera noise
        and merge everything together.

        Args:
            cam: camera module
            merger_bg: optional background merger
        """

        # make it a list because this allows for zipping later on
        self._noise: Optional[list[cam.Camera]] = (
            [cam] if cam is not None and not isinstance(cam, Sequence) else cam
        )
        self._merger_bg = background.Merger() if merger_bg is None else merger_bg
        self._scaler_frame = scaler_frame
        self._scaler_aux = scaler_aux

    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """

        Args:
            frame: of size T x H x W (T being the temporal window)
            em: not used
            bg: background tensor or sequence of background tensors
            aux: tensor of auxiliary model input or sequence of tensors

        """

        if bg is not None:
            frame = self._merger_bg.forward(frame=frame, bg=bg)

        if self._noise is not None:
            frame = [
                n.forward(f)
                for n, f in future.zip(  # raises err for unequal
                    self._noise,
                    frame if isinstance(frame, Sequence) else (frame,),
                    strict=True,
                )
            ]

        frame = torch.cat(frame, -3) if isinstance(frame, Sequence) else frame
        aux = torch.stack(aux) if isinstance(aux, Sequence) else aux

        frame = self._scaler_frame(frame) if self._scaler_frame is not None else frame
        aux = self._scaler_aux(aux) if self._scaler_aux is not None else aux

        # list of channels and auxiliary to frame tensor
        frame = torch.cat([frame, aux]) if aux is not None else frame
        return frame
