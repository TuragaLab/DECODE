import warnings

import torch
import pathlib
import tifffile
from typing import Union, Tuple

from tqdm import tqdm


def load_tif(file: (str, pathlib.Path)) -> torch.Tensor:
    """
    Reads the tif(f) files. When a folder is specified, potentially multiple files are loaded.
    Which are stacked into a new first axis.
    Make sure that if you provide multiple files (i.e. a folder) sorting gives the correct order. Otherwise we can
    not guarantee anything.

    Args:
        file: path to the tiff / or folder

    Returns:
        torch.Tensor: frames

    """

    p = pathlib.Path(file)

    """If dir, load multiple files and stack them if more than one found"""
    if p.is_dir():

        file_list = sorted(p.glob('*.tif*'))  # load .tif or .tiff
        frames = []
        for f in tqdm(file_list, desc="Tiff loading"):
            frames.append(torch.from_numpy(tifffile.imread(str(f)).astype('float32')))

        if frames.__len__() >= 2:
            frames = torch.stack(frames, 0)
        else:
            frames = frames[0]

    else:
        im = tifffile.imread(str(p))
        frames = torch.from_numpy(im.astype('float32'))

    if frames.squeeze().ndim <= 2:
        warnings.warn(f"Frames seem to be of wrong dimension ({frames.size()}), "
                      f"or could only find a single frame.", ValueError)

    return frames


class BatchTiffLoader:

    def __init__(self, par_folder: Union[str, pathlib.Path], file_suffix: str = 'tiff'):
        """
        Iterates through parent folder and returns the loaded frames as well as the filename in their iterator

        Example:
            >>> batch_loader = BatchTiffLoader('dummy_folder')
            >>> for frame, file in batch_loader:
            >>>     out = model.forward(frame)

        Args:
            par_folder:
            file_suffix:

        """

        self.par_folder = par_folder if isinstance(par_folder, pathlib.Path) else pathlib.Path(par_folder)
        self.file_suffix = file_suffix
        self.tiffs = self.get_all_files_rec(self.par_folder, self.file_suffix)

        self._n = -1

    @staticmethod
    def get_all_files_rec(par_folder: pathlib.Path, suffix):

        assert par_folder.is_dir()
        return list(par_folder.rglob('*.' + suffix))

    def __len__(self):
        return len(self.tiffs)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, pathlib.Path]:
        """

        Returns:
            torch.Tensor: frames
            Path: path to file

        """
        if self._n >= len(self) - 1:
            raise StopIteration

        self._n += 1
        return load_tif(self.tiffs[self._n]), self.tiffs[self._n]
