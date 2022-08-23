import collections
import pathlib
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Union, Optional, Any

import tifffile
import torch
from deprecated import deprecated

from ..generic import slicing


def load_tif(
    path: Union[str, pathlib.Path],
    auto_ome: bool = True,
    memmap: bool = False,
    dtype: str = torch.float32,
    dtype_inter: torch.dtype = "float32"
) -> torch.Tensor:
    """
    Reads the tif(f) files. When a list of paths is specified, multiple tiffs are
    loaded and stacked.

    Args:
        path: path to the tiff / or list of paths
        auto_ome: autoload ome files
        memmap: load as memory mapped tensor
        dtype: output dtype
        dtype_inter: intermediate read dtype
    """
    path = Path(path) if not isinstance(path, Path) else path

    # if dir, load multiple files and stack them if more than one found
    if isinstance(path, collections.abc.Sequence):
        if memmap:
            raise NotImplementedError("Memory map not supported for sequence of paths.")

    if memmap:
        frames = TiffTensor(path, auto_ome=auto_ome, dtype=dtype, dtype_inter=dtype_inter)
    else:
        if not auto_ome:
            raise ValueError(f"Auto OME is always on when not using memmap.")
        im = tifffile.imread(path)
        frames = torch.from_numpy(im.astype(dtype_inter)).type(dtype)

    if frames.dim() <= 2:
        warnings.warn(f"Frames are of dimension {frames.size()}.")

    return frames


class TiffTensor:
    def __init__(
        self,
        path: Path,
        auto_ome: bool = True,
        pattern_ome: str = ".ome",
        pattern_ome_suffix: str = "_[0-9]{1,2}",
    ):
        """
        Construct memory mapped tensor from path to tiff file

        Args:
            path:
            auto_ome: autodiscover ome connected files
            pattern_ome: pattern of ome, e.g. `.ome`
            pattern_suffix: regex compatible pattern of suffixes,
             default searches for _1, _2, ..., _99
        """
        if auto_ome:
            path = auto_discover_ome(
                path,
                pattern_ome=pattern_ome,
                pattern_suffix=pattern_ome_suffix,
            )
        else:
            path = [path] if not isinstance(path, Sequence) else path

        self._tifffile_raw = [tifffile.TiffFile(p, mode="rb") for p in path]
        self._data = TiffFilesTensor(self._tifffile_raw)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close files on exit
        [t.close() for t in self._tifffile_raw]

    def __getitem__(self, item) -> torch.Tensor:
        return self._data[item]

    def __len__(self) -> int:
        return len(self._data)

    def size(self, dim: Optional[int] = None) -> torch.Size:
        return self._data.size()


class TiffFilesTensor(slicing._LinearGetitemMixin, slicing._SizebyFirstMixin):
    def __init__(
        self,
        container: Union[tifffile.TiffFile, Sequence[tifffile.TiffFile]],
        dtype_inter: str = "int32",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Wraps tifffile.TiffFile's into a quasi-memory mapped tensor.

        Args:
            container: list of tifffile.TiffFiles
            dtype_inter: numpy dtype to convert to before converting to tensor
            dtype: torch dtype to convert to
        """
        container = [container] if not isinstance(container, Sequence) else container

        self._seq = slicing.ChainedSequence([c.pages for c in container])
        self._dtype_inter = dtype_inter
        self._dtype = dtype

    def __len__(self):
        return len(self._seq)

    def _collect_batch(self, batch: list) -> Any:
        return torch.stack(batch, 0)

    def _get_element(self, item: int) -> torch.Tensor:
        return torch.from_numpy(
            self._seq[item].asarray().astype(self._dtype_inter)
        ).type(self._dtype)


def auto_discover_ome(
    p: Path, pattern_ome: str = ".ome", pattern_suffix: str = "_[0-9]{1,2}"
) -> list[Path]:
    """
    Auto discovers adjacent .ome files in directory based on main .ome files path.
    Note that this is entirley based on naming conventions, no metadata or similiar is
    touched.

    Args:
        p: path of main ome file
        pattern_ome: pattern of ome, e.g. `.ome`
        pattern_suffix: regex compatible pattern of suffixes,
         default searches for _1, _2, ..., _99

    Returns:
        list of paths
    """
    base = p.stem.strip(pattern_ome)

    files = sorted(p.parent.glob(f"{base}*{p.suffix}"))
    files = [files[0]] + [f for f in files[1:] if re.search(pattern_suffix, f.stem)]

    return files


@deprecated(version="0.11", reason="nonsense")
class BatchFileLoader:
    pass
