import torch
from skimage.util.shape import view_as_windows

from typing import Sequence


def sample_crop(x_in: torch.Tensor, sample_size: Sequence[int]) -> torch.Tensor:

    """
    Takes a 2D tensor and returns random crops

    Args:
        x_in: input tensor
        sample_size: size of sample, size specification (N, H, W)

    Returns:
        random crops with size sample_size
    """

    assert x_in.dim() == 2, "Not implemented dimensionality"
    assert len(sample_size) == 3, "Wrong sequence dimension."

    windows = view_as_windows(x_in.numpy(), sample_size[-2:])  # converts array via sliding window into smaller ones

    n = sample_size[0]
    ix_max = (x_in.size(-2) - sample_size[-2], x_in.size(-1) - sample_size[-1])
    x_ix = torch.randint(0, ix_max[0] + 1, size=(n,))
    y_ix = torch.randint(0, ix_max[1] + 1, size=(n,))

    return torch.from_numpy(windows[x_ix, y_ix])
