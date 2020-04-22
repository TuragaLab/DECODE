import torch
import pytest

import deepsmlm.simulation.camera as pc
import deepsmlm.generic.utils.test_utils as tutil


class TestPhotons2Camera:

    @pytest.fixture(scope='class')
    def m2_spec(self):
        return pc.Photon2Camera(qe=1.0, spur_noise=0.002, em_gain=300., e_per_adu=45.,
                                baseline=100, read_sigma=74.4, photon_units=False)

    def test_shape(self, m2_spec):
        x = torch.ones((32, 3, 64, 64))
        assert tutil.tens_eqshape(x, m2_spec.forward(x))

    def test_photon_units(self, m2_spec):
        m2_spec.photon_units = True

        x = torch.rand((32, 3, 64, 64)) * 2000
        out = m2_spec.forward(x)

        tol = 0.01
        assert abs((x.mean() - out.mean()) / x.mean()) <= tol

    def test_warning(self, m2_spec):
        m2_spec.photon_units = True
        x = torch.rand((32, 3, 64, 64))
        out = m2_spec.backward(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Shipping to CUDA makes only sense if CUDA is available.")
    @pytest.mark.parametrize("input_device", ["cpu", "cuda"])
    @pytest.mark.parametrize("forward_device", ["cpu", "cuda"])
    def test_forward_backward_device(self, m2_spec, input_device, forward_device):

        exp_device = torch.rand(1).to(forward_device).device

        assert m2_spec.forward(torch.rand((1, 32, 32)).to(input_device), forward_device).device \
               == exp_device

        assert m2_spec.backward(torch.rand((1, 32, 32)).to(input_device), forward_device).device \
               == exp_device