import pytest
import torch

from decode.simulation import camera
from decode.generic import test_utils


@pytest.fixture
def cam_emccd():
    return camera.CameraEMCCD(
        qe=1.0,
        spur_noise=0.002,
        em_gain=300.0,
        e_per_adu=45.0,
        baseline=100,
        read_sigma=74.4,
        photon_units=False,
    )


@pytest.fixture
def cam_scmos():
    return camera.CameraSCMOS(
        qe=1.0,
        spur_noise=0.002,
        e_per_adu=45.0,
        baseline=100,
        read_sigma=74.4,
        photon_units=False,
    )


@pytest.fixture
def cam_perfect():
    return camera.CameraPerfect()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Shipping to CUDA makes only sense if CUDA is available.",
)
@pytest.mark.parametrize("cam", ["cam_emccd", "cam_scmos", "cam_perfect"])
@pytest.mark.parametrize("device_tar", ["cpu", "cuda"])
@pytest.mark.parametrize("device_input", ["cpu", "cuda"])
def test_device(cam, device_tar, device_input, request):
    cam = request.getfixturevalue(cam)
    # easier like this because index implicitly included
    exp_device = torch.rand(1, device=device_tar).device

    x = torch.rand((1, 32, 32), device=device_input)
    assert cam.forward(x, device_tar).device == exp_device
    assert cam.backward(x, device_tar).device == exp_device


@pytest.mark.parametrize("cam", ["cam_emccd", "cam_scmos", "cam_perfect"])
def test_shape(cam, request):
    cam = request.getfixturevalue(cam)

    x = torch.rand((32, 3, 64, 64))
    assert x.size() == cam.forward(x).size()


@pytest.mark.parametrize("cam", ["cam_emccd", "cam_scmos", "cam_perfect"])
def test_photon_units(cam, request):
    cam = request.getfixturevalue(cam)
    cam.photon_units = True

    x = torch.rand((32, 3, 64, 64)) * 20000
    out = cam.forward(x)
    assert x.mean() == pytest.approx(out.mean(), rel=0.01), (
        "Mean should not change much in " "photon units"
    )


def test_perfect_forward_backward(cam_perfect):
    # for a perfect camera forward and backward is just statistical noise and should
    # be equivalent
    x = torch.rand((32, 3, 64, 64)) * 10000

    x_out_cnt = cam_perfect.forward(x)
    x_out_phot = cam_perfect.backward(x_out_cnt)

    assert test_utils.tens_almeq(x_out_phot, x_out_cnt)
