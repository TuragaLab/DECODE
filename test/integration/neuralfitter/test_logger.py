from pathlib import Path

import matplotlib.pyplot as plt
import torch

from decode.emitter import emitter
from decode.neuralfitter import logger


def test_logger(tmpdir):
    # Note: it's instructive to look at the actual tensorboard instance
    tb = logger.TensorboardLogger(save_dir=tmpdir)

    for i in range(10):
        tb.log_group({"a": 42 * i}, step=i)

    # get events file
    p = next(Path(tmpdir).rglob("event*"))
    assert p.stat().st_size > 50, "TB event file too small."

    # plotting
    tb.log_figure("dummy", plt.figure(), close=True, step=0)
    assert p.stat().st_size > 400, "TB event file too small after plotting"

    # plot tensor
    tb.log_tensor(torch.rand(5, 32, 32), "tensor", unbind=0)

    # plot frame and emitters
    em = emitter.factory(10, extent=32, xy_unit="px")
    em_tar = em.clone()
    em_tar.xyz_px += torch.randn_like(em_tar.xyz_px)
    tb.log_emitter(
        "dummy_frame", em=em, em_tar=em_tar, frame=torch.rand(32, 32), step=0
    )
