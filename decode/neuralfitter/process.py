from typing import Any, Optional, Protocol

import torch

from ..emitter import emitter


class Processing:
    def __init__(self, mode: str = "train"):
        """Signature illustration of processing wrapping class"""
        self._mode = mode

    def pre(self, *args, **kwargs):
        """
        Returns a processed training sample in `train` mode and an inference sample in
        `eval` mode.

        Args:
            *args:
            **kwargs:

        """
        if self._mode == "train":
            return self.pre_train(*args, **kwargs)
        elif self._mode == "eval":
            return self.pre_inference(*args, **kwargs)
        else:
            raise NotImplementedError

    def pre_train(self, x: torch.Tensor, y: Any) -> tuple[torch.Tensor, Any]:
        """
        Preprocessing at training time

        Args:
            x: input to the model
            y: target
        """
        return self._pre_input(x), self._pre_tar(y)

    def pre_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing at inference time

        Args:
            x: input to the model
        """
        return self._pre_input(x)

    def input(
        self, frame: torch.Tensor, em: emitter.EmitterSet, aux: Any
    ) -> torch.Tensor:
        # default: frame preparation only depends on frame
        return self._pre_input(frame)

    def post(self, x: Any) -> emitter.EmitterSet:
        """
        Postprocessing takes intermediate transformed model output and returns final
        endpoint data.

        Args:
            x:
        """
        raise NotImplementedError

    def post_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes immediate model output
        """

    def tar(self, em: emitter.EmitterSet, aux: Any) -> torch.Tensor:
        """
        Transforms to input for loss

        Args:
            em: set of emitters
            aux:
        """
        raise NotImplementedError

    def _pre_input(self, x: torch.Tensor) -> torch.Tensor:
        """Independent input transformation"""
        raise NotImplementedError

    def _pre_tar(self, y: Any) -> Any:
        """Independent target transformation"""
        raise NotImplementedError


class _Forwardable(Protocol):
    def forward(self, x: Any) -> Any:
        ...


class ProcessingSupervised(Processing):
    def __init__(
        self,
        shared_input: Optional[_Forwardable] = None,
        pre_input: Optional[_Forwardable] = None,
        pre_tar: Optional[_Forwardable] = None,
        tar: Optional[_Forwardable] = None,
        post_model: Optional[_Forwardable] = None,
        post: Optional[_Forwardable] = None,
        mode: str = "train",
    ):
        """

        Args:
            shared_input:
            pre_input: input processing (must return model input), forward depends on frame only
            pre_tar:
            tar:
            post_model:
            post:
            mode:
        """
        super().__init__(mode=mode)

        self._shared_input = shared_input
        self._pre_input_impl = pre_input
        self._pre_tar_impl = pre_tar
        self._tar = tar
        self._post_model = post_model
        self._post = post

    def input(
        self, frame: torch.Tensor, em: emitter.EmitterSet, aux: Any
    ) -> torch.Tensor:
        if self._shared_input is not None:
            frame = self._shared_input.forward(frame, em, aux)
        if self._pre_input_impl is not None:
            frame = self._pre_input_impl.forward(frame)

        return frame

    def tar(self, em: emitter.EmitterSet, aux: Any) -> torch.Tensor:
        em = self._pre_tar(em)
        return self._tar.forward(em, aux)

    def post(self, x: torch.Tensor) -> emitter.EmitterSet:
        x = self.post_model(x)
        return self._post.forward(x)

    def post_model(self, x: torch.Tensor) -> torch.Tensor:
        return self._post_model.forward(x)

    def _pre_tar(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        return self._pre_tar_impl.forward(em)
