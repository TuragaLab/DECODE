import pytest
import torch

from deepsmlm.generic.utils import processing


class TestTransformSequence:

    @pytest.fixture()
    def trafo(self):

        class MockCom0:
            def forward(self, a, b):
                return a * b, a, b

        class MockCom1:
            def forward(self, a):
                return a + 2, a

        class MockCom2:
            def forward(self, a):
                return a, a * 2

        return processing.TransformSequence([MockCom0(), MockCom1(), MockCom2()],
                                            input_slice=[[0], [0]])

    def test_len(self, trafo):

        assert len(trafo) == len(trafo.com)

    def test_forward(self, trafo):

        trafo.forward(torch.rand((32, 32)), torch.rand((32, 32)))



