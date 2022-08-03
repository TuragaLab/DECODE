from typing import Optional, Union, Any

import torch


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
