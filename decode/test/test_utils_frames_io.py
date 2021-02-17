import threading
import queue
import time

import tifffile
import torch

from decode.generic import test_utils
from decode.utils import frames_io


def online_tiff_writer(path, iterations: int, sleep: float, out_queue=None):
    """Creates a tiff file and writes for n iterations to it with at least 1s in between."""
    assert iterations >= 2

    batch_size = 100

    img = torch.randint(255, (iterations * batch_size, 64, 64), dtype=torch.short)
    if out_queue is not None:
        out_queue.put(img)  # put image to queue already here so that the other side can take it already

    img_chunked = torch.chunk(img, iterations, dim=0)

    # write batches of img
    for chunk in img_chunked:
        time.sleep(sleep)
        tifffile.imwrite(path, data=chunk.numpy(), ome=True, append=True)


def test_tiff_tensor(tmpdir):
    fname = tmpdir / 'contin.tiff'

    # start a thread that continuously write to a tiff file
    q = queue.Queue()
    thread = threading.Thread(target=online_tiff_writer, args=[str(fname), 10, 1, q])
    thread.start()

    # get ground through tensor already to be able to check against it
    tiff_gt = q.get()

    # wait until file is there
    while not test_utils.file_loadable(fname, tifffile.TiffFile, mode='rb',
                                       exceptions=(KeyError, tifffile.TiffFileError)):
        time.sleep(0.5)

    tiff = frames_io.TiffTensor(fname)

    lengths = []
    for i in range(15):
        n = len(tiff)

        # check that what is loadable is correct
        assert(tiff[:n] == tiff_gt[:n]).all()

        lengths.append(n)
        time.sleep(0.5)

    # wait for last write
    thread.join()
    lengths.append(len(tiff))

    assert len(torch.Tensor(n).unique()) >= 5  # kind of stochastic, would fail for ultra slow write
    assert lengths[-1] == 1000
