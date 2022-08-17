from pathlib import Path
from unittest import mock

import numpy as np
import torch
import pytest

from decode.io import frames


@pytest.mark.parametrize("path,memmap", [
    (["", ""], False),  # simulate list of tiffs
    ("", False),
    ("", True)
])
@mock.patch.object(frames.tifffile, "imread")
@mock.patch.object(frames.tifffile, "memmap")
def test_load_tif(mock_mem, mock_imread, path, memmap, tmpdir):
    frames_tensor = torch.rand(5, 32, 34)

    p = Path(tmpdir) / "mm.dat"
    mm = np.memmap(p, shape=(5, 32, 34), mode="w+", dtype="float32")
    mm[:] = frames_tensor.numpy()
    mm.flush()

    mock_imread.return_value = frames_tensor.numpy()
    mock_mem.return_value = mm

    f = frames.load_tif(path, memmap=memmap)
    if isinstance(path, list):
        assert f.size() == torch.Size([2, *frames_tensor.size()])
        assert (f[0] == frames_tensor).all()
    else:
        assert f.size() == frames_tensor.size()
        assert (f[:] == frames_tensor).all()


def test_tensor_memmap(tmpdir):
    x = torch.rand(32, 34, 36)

    p = Path(tmpdir) / "memmap.dat"
    x_mm = np.memmap(p, shape=(32, 34, 36), mode="w+", dtype="float32")
    x_mm[:] = x[:]
    x_mm.flush()

    x_tensor_mm = frames.TensorMemMap(x_mm)

    assert (x_tensor_mm[:] == x[:]).all()
    assert (x_tensor_mm[27:49, 29:] == x[27:49, 29:]).all()
    assert len(x_tensor_mm) == len(x)
    assert x_tensor_mm.size() == x.size()
    assert x_tensor_mm.dim() == x.dim()
