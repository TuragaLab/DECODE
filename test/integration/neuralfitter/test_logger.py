from pathlib import Path

import matplotlib.pyplot as plt
import torch

from decode.neuralfitter import logger


def test_logger(tmpdir):
    tb = logger.TensorboardLogger(save_dir=tmpdir)

    for i in range(10):
        tb.log_group({"a": 42 * i}, step=i)

    # get events file
    p = next(Path(tmpdir).rglob("event*"))
    assert p.stat().st_size > 50, "TB event file too small."

    # plotting
    tb.log_figure("dummy", plt.figure(), close=True, step=0)
    assert p.stat().st_size > 400, "TB event file too small after plotting"

    tb.log_tensor(torch.rand(5, 32, 32), "tensor", unbind=0)
    assert True
