import pytest
import torch

from decode.generic import utils


@pytest.mark.parametrize("arr,expct", [((5108, 3239, 3892, 570, 4994, 3428, 800, 2025, 1206, 655, 1707, 3239),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)),
                                       ((5108, 3239, 1707, 3239), (0, 0, 0, 1))])
def test_cum_count_per_group(arr, expct):

    out = utils.cum_count_per_group(torch.Tensor(arr))
    assert isinstance(out, torch.LongTensor)
    assert (out == torch.LongTensor(expct)).all()


@pytest.mark.parametrize("xextent,yextent,img_size,expct_x,expct_y", [
    ((-0.5, 31.5), (-0.5, 31.5), (32, 32), torch.arange(32).float(), torch.arange(32).float()),
    ((-0.5, 31.5), (0.5, 32.5), (64, 64), torch.arange(64).float() / 2 - 0.25, torch.arange(64).float() / 2 - 0.25 + 1)
])
def test_frame_grid(xextent, yextent, img_size, expct_x, expct_y):

    _, _, ctr_x, ctr_y = utils.frame_grid(img_size, xextent, yextent)

    assert (ctr_x == expct_x).all()
    assert (ctr_y == expct_y).all()


@pytest.mark.parametrize("origin,px_size,xextent,yextent,img_size", [
    ((-0.5, -0.5), (1., 1.), (-0.5, 31.5), (-0.5, 31.5), (32, 32)),
    ((-0.5, 0.5), (0.5, 0.5), (-0.5, 31.5), (0.5, 32.5), (64, 64))
])
def test_frame_grid_argument_equivalence(origin, px_size, xextent, yextent, img_size):

    bin_x_expct, bin_y_expct, ctr_x_expct, ctr_y_expct = utils.frame_grid(img_size, xextent, yextent)
    bin_x, bin_y, ctr_x, ctr_y = utils.frame_grid(img_size=img_size, origin=origin, px_size=px_size)

    assert (bin_x == bin_x_expct).all()
    assert (bin_y == bin_y_expct).all()
    assert (ctr_x == ctr_x_expct).all()
    assert (ctr_y == ctr_y_expct).all()
