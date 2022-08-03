from typing import Any, Optional, Protocol, Union

import torch

from ..emitter import emitter


class IxWindow:
    def __init__(self, win: int, n: Optional[int]):
        """
        'Window' an index, that make convolution like.

        Args:
            win: window size
            n: data size to clamp upper index values

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

    def __call__(self, ix: Union[int, slice]) -> Union[list[int], list[list[int]]]:
        return self._compute(ix)

    def __getitem__(self, ix: Union[int, slice]) -> Union[list[int], list[list[int]]]:
        return self._compute(ix)

    def __len__(self) -> int:
        if self._n is None:
            raise ValueError

        return self._n

    def _compute(self, ix: Union[int, slice]) -> Union[list[int], list[list[int]]]:
        if isinstance(ix, slice):
            # for slice, recurse via ints
            ix = list(
                range(
                    ix.start if ix.start is not None else 0,
                    ix.stop if ix.stop is not None else self._n,
                    ix.step if ix.step is not None else 1,
                )
            )
            return [self._compute(i) for i in ix]

        if ix < 0:
            raise NotImplementedError("Negative indexing not supported.")
        if self._n is not None and ix >= self._n:
            raise IndexError("index out of range.")

        hw = (self._win - 1) // 2  # half window without centre
        ix = torch.arange(ix - hw, ix + hw + 1).clamp(0)

        if self._n is not None:
            ix = ix.clamp(max=self._n - 1)

        return ix.tolist()

    def attach(self, x: Any):
        class _WrappedSliceable:
            def __init__(self, obj):
                self._obj = obj

            def __getitem__(self_inner, item: int):
                return self_inner._obj[self(item)]

        self._n = len(x)
        return _WrappedSliceable(x)


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
        pre_input: Optional[_Forwardable] = None,
        pre_tar: Optional[_Forwardable] = None,
        tar: Optional[_Forwardable] = None,
        post_model: Optional[_Forwardable] = None,
        post: Optional[_Forwardable] = None,
        mode: str = "train",
    ):
        """

        Args:
            pre_input: input processing (must return model input), forward depends on frame only
            pre_tar:
            tar:
            post_model:
            post:
            mode:
        """
        super().__init__(mode=mode)

        self._pre_input_impl = pre_input
        self._pre_tar_impl = pre_tar
        self._tar = tar
        self._post_model = post_model
        self._post = post

    def tar(self, em: emitter.EmitterSet, aux: Any) -> torch.Tensor:
        em = self._pre_tar(em)
        return self._tar.forward(em, aux)

    def post(self, x: torch.Tensor) -> emitter.EmitterSet:
        x = self.post_model(x)
        return self._post.forward(x)

    def post_model(self, x: torch.Tensor) -> torch.Tensor:
        return self._post_model.forward(x)

    def _pre_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._pre_input_impl.forward(x)

    def _pre_tar(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        return self._pre_tar_impl.forward(em)
