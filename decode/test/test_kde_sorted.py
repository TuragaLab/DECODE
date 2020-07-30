import torch
import pytest
import matplotlib.pyplot as plt

from decode.evaluation import utils


class TestKDESorted:

    def test_nan(self):

        x = torch.randn(25, )
        y = float('nan') * torch.ones_like(x)

        z, x, y = utils.kde_sorted(x, y, True, nan_inf_ignore=True)
        plt.show()
