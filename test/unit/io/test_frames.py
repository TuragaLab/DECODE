from pathlib import Path
from unittest import mock

import numpy as np
import torch
import pytest

from decode.io import frames


@pytest.fixture()
def tifffile_pages():
    # mocked tifffile pages
    pages_raw = torch.unbind(torch.randint(256, size=(200, 32, 34), dtype=torch.uint8))
    pages = [None] * len(pages_raw)
    for i, pr in enumerate(pages_raw):
        m = mock.MagicMock()
        m.asarray.return_value = pr.numpy().astype("uint16")
        pages[i] = m
    return pages


@pytest.fixture
def tifffile_tifffile(tifffile_pages):  # aka tifffile.TiffFile
    t = mock.MagicMock()
    t.pages = tifffile_pages
    t.__len__.return_value = len(tifffile_pages)
    return t


def test_tiff_tensor(tifffile_tifffile):
    tiff = frames.TiffTensor([], auto_ome=False)
    tiff._data = torch.rand(200, 2, 4)

    assert len(tiff) == len(tiff._data)
    assert (tiff[0] == tiff._data[0]).all()
    assert tiff.size() == tiff._data.size()


def test_tiff_file_tensor(tifffile_tifffile):
    tiff = frames.TiffFilesTensor([tifffile_tifffile, tifffile_tifffile])

    assert len(tiff) == 2 * len(tifffile_tifffile)
    assert tiff.size() == torch.Size([400, 32, 34])
    assert isinstance(tiff._get_element(0), torch.Tensor)
    assert tiff[5:10].size() == torch.Size([5, 32, 34])


@pytest.mark.parametrize("paths", [
    ["a.ome.tif", "a_1.ome.tif"],
    ["b.ome.tif"],
])
def test_auto_discover_ome(paths):
    paths = [Path(pp) for pp in paths]

    p = mock.MagicMock()
    p.stem.strip.return_value = "a"
    p.parent.glob.return_value = paths

    path_out = frames.auto_discover_ome(p)

    assert path_out == paths
