import warnings

import torch
import pathlib
import tifffile

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
