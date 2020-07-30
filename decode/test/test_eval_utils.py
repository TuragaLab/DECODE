import pytest
import torch
from matplotlib import pyplot as plt

import decode.evaluation
from decode.generic import test_utils as tutil


class TestMetricMeter:

    @pytest.fixture(scope='class')
    def mm0(self):
        x = torch.ones(32)
        m = decode.evaluation.utils.MetricMeter()
        m.vals = x

        return m

    @pytest.fixture(scope='class')
    def mm1(self):
        x = torch.ones(32) * 2
        m = decode.evaluation.utils.MetricMeter()
        m.vals = x
        return m

    def test_operators(self, mm0, mm1):
        assert tutil.tens_almeq((mm0 + mm1).vals, mm0.vals + mm1.vals)
        assert tutil.tens_almeq((mm0 + 42.).vals, mm0.vals + 42.)
        assert tutil.tens_almeq((42. + mm0).vals, mm0.vals + 42.)
        assert tutil.tens_almeq((mm0 - mm1).vals, mm0.vals - mm1.vals)
        assert tutil.tens_almeq((mm0 - 42.).vals, mm0.vals - 42.)
        assert tutil.tens_almeq((42. - mm0).vals, 42. - mm0.vals)
        assert tutil.tens_almeq((mm0 * mm1).vals, mm0.vals * mm1.vals)
        assert tutil.tens_almeq((mm0 * 42.).vals, mm0.vals * 42.)
        assert tutil.tens_almeq((42. * mm0).vals, mm0.vals * 42.)
        assert tutil.tens_almeq((mm0 / mm1).vals, mm0.vals / mm1.vals)
        assert tutil.tens_almeq((mm0 / 42.).vals, mm0.vals / 42.)
        assert tutil.tens_almeq((mm0 ** 2).vals, mm0.vals**2)


class TestCumulantMeter:

    def test_cumulation(self):
        m = decode.evaluation.utils.CumulantMeter()
        m.update(torch.rand(10000))
        m.update(torch.rand(10000))
        assert 0.5 == pytest.approx(m.avg, 1e-2)

    @pytest.mark.plot
    def test_histogram(self):
        m = decode.evaluation.utils.CumulantMeter()
        m.update(torch.rand(10000))
        m.update(torch.rand(10000))
        f = m.hist()
        plt.show()
        assert True