import matplotlib.pyplot as plt
import pytest
import torch

import deepsmlm.evaluation.evaluation as evaluation


class TestCumulantMeter:

    def test_cumulation(self):
        m = evaluation.CumulantMeter()
        m.update(torch.rand(10000))
        m.update(torch.rand(10000))
        assert 0.5 == pytest.approx(m.avg, 1e-2)

    def test_histogram(self):
        m = evaluation.CumulantMeter()
        m.update(torch.rand(10000))
        m.update(torch.rand(10000))
        f = m.hist
        assert True
