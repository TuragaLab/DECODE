import torch
import pytest

import deepsmlm.generic.phot_camera as pc
import deepsmlm.test.utils_ci as tutil


class TestPhotons2Camera:

    @pytest.fixture(scope='class')
    def mt1_spec(self):
        return pc.Photon2Camera(qe=0.9, spur_noise=, em_gain=300., e_per_adu=45., baseline=100, read_sigma=74.4)

    def test_shape(self, mt1_spec):
        x = torch.ones((32, 3, 64, 64))
        assert tutil.tens_eqshape(x, mt1_spec.forward(x))
