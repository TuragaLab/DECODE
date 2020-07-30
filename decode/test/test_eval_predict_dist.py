import matplotlib.pyplot as plt
import pytest
import torch

from decode.evaluation import predict_dist


class TestPredictDist:

    @pytest.mark.plot
    def test_z_pred_gt(self):
        """Setup"""
        z_gt = torch.rand(10000)
        z = z_gt + torch.randn_like(z_gt) * 0.05

        """Run"""
        f, ax = plt.subplots()
        predict_dist.deviation_dist(z, z_gt, residuals=True, kde=True, ax=ax)
        plt.show()
