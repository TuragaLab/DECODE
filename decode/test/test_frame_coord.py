import torch
import matplotlib.pyplot as plt

import pytest

from decode.plot import frame_coord


@pytest.mark.plot
@pytest.mark.parametrize("em_tar", [torch.zeros((0, 3)), torch.rand((50, 3))])
@pytest.mark.parametrize("em_out", [torch.zeros((0, 3)), torch.rand((50, 3))])
def test_plot3d(em_tar, em_out):

    f = plt.figure()
    frame_coord.PlotCoordinates3D(pos_tar=em_tar, pos_out=em_out).plot()
    plt.show()
