import multiprocessing
import time

import tifffile
import torch

from decode.utils import frames_io


def online_tiff_writer(path, n: int, sleep: float):
    """Creates a tiff file and writes for n iterations to it with at least 1s in between."""

    img = torch.randint(255, (100, 64, 64), dtype=torch.short)
    tifffile.imwrite(path, data=img.numpy(), ome=True)

    for _ in range(n):
        time.sleep(sleep)

        new = torch.randint(255, (100, 64, 64), dtype=torch.short)
        tifffile.imwrite(path, data=new.numpy(), append=True)


def test_tiff_tensor(tmpdir):
    fname = tmpdir / 'contin.tiff'

    # start a thread that continuously write to a tiff file
    thread = multiprocessing.Process(target=online_tiff_writer, args=[str(fname), 10, 1])
    thread.start()

    while not fname.isfile():
        time.sleep(0.5)

    tiff = frames_io.TiffTensor(fname)

    # check file length check
    n = []
    for i in range(15):
        n.append(len(tiff))
        time.sleep(0.5)

    # wait for last write
    thread.join()
    n.append(len(tiff))

    assert len(torch.Tensor(n).unique()) >= 5  # kind of stochastic, would fail for ultra slow write
    assert n[-1] == 1100

    # check loading
    assert isinstance(frames_io.TiffTensor(fname)[:500], torch.Tensor)
