from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import torch


class TensorMemoryMapped(ABC, Sequence):
    """
    Minimal pseudo torch tensor, must be a sequence, implements `size` and
    returns proper torch.Tensor on __getitem__.
    """
    @abstractmethod
    def size(self, dim: Optional[int]) -> Union[int, torch.Size]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item) -> torch.Tensor:
        pass
