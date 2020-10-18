import pytest
import torch
import unittest.mock

from decode.neuralfitter.utils import processing


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
                                            input_slice=[[0, 1], [0], [0]])

    def test_len(self, trafo):

        assert len(trafo) == len(trafo.com)

    def test_forward(self, trafo):

        trafo.forward(torch.rand((32, 32)), torch.rand((32, 32)))


class TestParallelTransformSequence(TestTransformSequence):

    @pytest.fixture()
    def trafo(self):

        class MockCom0:
            def forward(self, a, b):
                return a * b

        class MockCom1:
            def forward(self, a, b, c):
                return a * b ** c

        def mock_combiner(out_cache):
            return torch.cat(out_cache, dim=1).squeeze(0)

        return processing.ParallelTransformSequence([MockCom0(), MockCom1()],
                                                    input_slice=[[0, 2], [0, 1, 2]],
                                                    merger=mock_combiner)

    def test_forward(self, trafo):

        """Run"""
        out = trafo.forward(torch.zeros((1, 5, 32, 32)), torch.rand((1, 5, 32, 32)), torch.ones((1, 1, 32, 32)) * 2)

        """Assert"""
        assert out.dim() == 3
        assert len(out) == 10
        assert (out[:5] == 0.).all()


class TestWrapCallable:

    @staticmethod
    def test_wrap_callable():

        def dummy(x):
            return x.clone()

        assert hasattr(processing.wrap_callable(dummy), 'forward')

        x = torch.rand(5, 5)
        assert (dummy(x) == processing.wrap_callable(dummy).forward(x)).all()

    @staticmethod
    def test_pickleability():
        """Test whether camera.backward is pickleable"""
        import pickle
        from decode.simulation.camera import Photon2Camera

        camera = Photon2Camera(qe=1.0, spur_noise=0., em_gain=100., e_per_adu=10., baseline=100., read_sigma=25, photon_units=True, device='cpu')
        wrapped = processing.wrap_callable(camera.backward)

        """Tests"""
        s = pickle.dumps(wrapped)
        w = pickle.loads(s)
