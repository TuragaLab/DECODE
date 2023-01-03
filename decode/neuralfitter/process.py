from typing import Any, Optional, Protocol, Sequence, Union

import torch
from deprecated import deprecated

from ..emitter import emitter


@deprecated(version="0.11.0", reason="No purpose, too abstract")
class Processing:
    pass


class _Forwardable(Protocol):
    def forward(self, x: Any) -> Any:
        ...


class ProcessingSupervised:
    def __init__(
        self,
        m_input: Optional[_Forwardable] = None,
        tar: Optional[_Forwardable] = None,
        tar_em: Optional[_Forwardable] = None,
        post_model: Optional[_Forwardable] = None,
        post: Optional[_Forwardable] = None,
        mode: str = "train",
    ):
        """

        Args:
            m_input: input processing, forward has frame, emitter and auxiliary
             arguments; must return model compatible input
            tar: compute target
            tar_em: compute target emitters without actually computing the target,
             useful for validation
            post_model:
            post:
            mode:
        """
        super().__init__()
        self.mode = mode
        self._m_input = m_input
        self._tar = tar
        self._tar_em = tar_em
        self._post_model = post_model
        self._post = post

    def pre_train(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Preprocessing for training

        Args:
            frame:
            em:
            bg:
            aux:

        """
        return self._m_input.forward(frame=frame, em=em, bg=bg, aux=aux)

    def pre_inference(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Preprocessing for inference

        Args:
            frame:
            aux:

        """
        raise NotImplementedError

    def tar(self, em: emitter.EmitterSet, aux: dict[str, Any]) -> torch.Tensor:
        return self._tar.forward(em, aux)

    def tar_em(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        return self._tar_em.forward(em)

    def post(self, x: torch.Tensor) -> emitter.EmitterSet:
        """
        Process model output through whole post-processing pipeline to get EmitterSet.

        Args:
            x: model output

        """
        return self._post.forward(x)

    def post_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process model output only to post-processing necessary to compute the loss

        Args:
            x: model output

        """
        return self._post_model.forward(x)
