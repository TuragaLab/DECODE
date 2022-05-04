import pytest
import torch

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
        assert tutil.tens_almeq((mm0 ** 2).vals, mm0.vals ** 2)
