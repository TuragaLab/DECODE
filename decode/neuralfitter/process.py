from typing import Any, Optional

import torch

from ..emitter import emitter


class IxWindow:
    def __init__(self, win: int, n: Optional[int]):
        """
        'Window' an index, that make convolution like.

        Args:
            win: window size
            n: data size

        Examples:
            >>> IxWindow(3)(0)
            [0, 0, 1]

            >>> IxWindow(3)(5)
            [4, 5, 6]

            >>> IxWindow(3, n=5)(4)
            [3, 4, 4]
        """
        self._win = win
        self._n = n

    def __call__(self, ix: int) -> list[int]:
        hw = (self._win - 1) // 2  # half window without centre
        ix = torch.arange(ix - hw, ix + hw + 1).clamp(0)

        if self._n is not None:
            ix = ix.clamp(max=self._n - 1)

        return ix.tolist()


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
        pass

    def _pre_input(self, x: torch.Tensor) -> torch.Tensor:
        """Independent input transformation"""
        pass

    def _pre_tar(self, y: Any) -> Any:
        """Independent target transformation"""
        pass


class ProcessingSupervised(Processing):
    def __init__(
        self,
        pre_input: Optional = None,
        pre_tar: Optional = None,
        tar: Optional = None,
        post_model: Optional = None,
        post: Optional = None,
        mode: str = "train",
    ):
        super().__init__(mode=mode)

        self._pre_input_impl = pre_input
        self._pre_tar_impl = pre_tar
        self._tar = tar
        self._post_model = post_model
        self._post = post

    def tar(self, em: emitter.EmitterSet, aux: Any) -> torch.Tensor:
        return self._tar.forward(em, aux)

    def post(self, x: torch.Tensor) -> emitter.EmitterSet:
        return self._post.forward(x)

    def post_model(self, x: torch.Tensor) -> torch.Tensor:
        return self._post_model.forward(x)

    def _pre_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._pre_input_impl.forward(x)

    def _pre_tar(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        return self._pre_tar_impl.forward(em)
