import torch

from decode.evaluation import utils


class TestKDESorted:

    def test_nan(self):
        x = torch.randn(25, )
        x[1] = float('nan')
        y = torch.rand_like(x)
        y[5:] = float('nan') * torch.ones_like(y[5:])

        z, x, y = utils.kde_sorted(x, y, plot=False, nan_inf_ignore=True, sub_sample=False)
