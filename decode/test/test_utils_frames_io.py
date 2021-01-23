import torch
import tifffile
import pytest
import time
import threading

from decode.utils import frames_io


def online_tiff_writer(path):

    img = torch.randint(255, (100, 64, 64), dtype=torch.short)
    tifffile.imwrite(path, data=img.numpy(), ome=True)

    for _ in range(10):
        time.sleep(1)

        new = torch.randint(255, (100, 64, 64), dtype=torch.short)
        tifffile.imwrite(path, data=new.numpy(), append=True)


def test_tiff_tensor(tmpdir):

    fname = tmpdir / 'contin.tiff'

    # start a thread that continuously write to a tiff file
    thread = threading.Thread(target=online_tiff_writer, args=[str(fname)])
    thread.start()

    while not fname.isfile():
        time.sleep(0.5)

    # check file length check
    n = []
    for i in range(15):
        n.append(len(frames_io.TiffTensor(fname)))
        time.sleep(1)

    assert len(torch.Tensor(n).unique()) >= 5

    # check loading
    assert isinstance(frames_io.TiffTensor(fname)[:500], torch.Tensor)
