import collections
import pathlib
import warnings
from typing import Union, Optional

import numpy as np
import tifffile
import torch
from deprecated import deprecated
from pydantic import validate_arguments


@validate_arguments
def load_tif(
    path: Union[list[pathlib.Path], pathlib.Path],
    multifile: bool = True,
    memmap: bool = False,
    dtype: str = "float32",
) -> torch.Tensor:
    """
    Reads the tif(f) files. When a list of paths is specified, multiple tiffs are
    loaded and stacked.

    Args:
        path: path to the tiff / or list of paths
        multifile: auto-load multi-file tiff (for large files)
        memmap: load as memory mapped tensor
    """

    # if dir, load multiple files and stack them if more than one found
    if isinstance(path, collections.abc.Sequence):
        if memmap:
            raise NotImplementedError("Memory map not supported for sequence of paths.")
        return torch.stack(
            [load_tif(p, multifile=multifile, memmap=False) for p in path], dim=0
        )

    if memmap:
        mm = tifffile.memmap(path)
        frames = TensorMemMap(mm)
    else:
        im = tifffile.imread(path, multifile=multifile)
        frames = torch.from_numpy(im.astype(dtype))

    if frames.dim() <= 2:
        warnings.warn(
            f"Frames seem to be of wrong dimension ({frames.size()}), "
            f"or could only find a single frame.",
            ValueError,
        )

    return frames


class TensorMemMap:
    def __init__(self, np_memmap: np.memmap):
        """
        Memory-mapped tensor from numpy memmap.
        Note that data is loaded only to the extent to  which the object is accessed
        through brackets '[ ]' Therefore, this tensor has no value and no state until it
        is sliced and then returns a torch tensor.
        You can of course enforce loading the whole tiff by tiff_tensor[:]

        Args:
            np_memmap: numpy memory map
        """
        self._memmap = np_memmap

    def __getitem__(self, pos: Union[int, tuple, slice]) -> torch.Tensor:
        return torch.from_numpy(self._memmap[pos])

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._memmap)

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        size = torch.Size(self._memmap.shape)

        if dim is not None:
            return size[dim]

        return size

    def dim(self) -> int:
        return self._memmap.ndim


@deprecated(version="0.11", reason="nonsense")
class BatchFileLoader:
    pass
