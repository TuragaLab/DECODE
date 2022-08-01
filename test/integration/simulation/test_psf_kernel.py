import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import decode.plot.plot as plf
from decode import io
from decode.generic import asset_handler


@pytest.mark.parametrize(
    "calib",
    [
        "bead_cal",
        "bead_cal_mat73",
    ],
)
def test_psf_mat_versions(calib):
    path = asset_handler.load_asset(calib)
    io.psf.load_spline(path, (-0.5, 31.5), (-0.5, 38.5), (32, 39))


@pytest.fixture()
def psf():
    xextent = (-0.5, 38.5)
    yextent = (-0.5, 63.5)
    img_shape = (39, 64)

    path = asset_handler.load_asset("bead_cal")

    return io.psf.load_spline(
        path=path,
        xextent=xextent,
        yextent=yextent,
        img_shape=img_shape,
        roi_size=(32, 32),
    )


def _random_positions(n: int, xextent=64, yextent=64, zextent=(-500, 500), weight=1.0):

    xyz = torch.rand(n, 3)
    xyz[:, :2] *= torch.tensor([xextent, yextent])
    xyz[:, 2] = xyz[:, 2] * (zextent[1] - zextent[0]) - 500
    weight = torch.ones(n) * weight

    return xyz, weight


def test_spline_crlb_range(psf):
    # many photons to make the derivatives not explode
    xyz, weight = _random_positions(100, weight=10000)
    bg = torch.ones_like(weight) * 50

    crlb, _ = psf.crlb(xyz, weight, bg)

    assert (
        torch.Tensor([0.01, 0.01, 0.02]) ** 2 <= crlb[:, :3]
    ).all(), "CRLB not in reasonable range (lower bound)"
    assert (
        torch.Tensor([0.1, 0.1, 100]) ** 2 >= crlb[:, :3]
    ).all(), "CRLB not in reasonable range (upper bound)"


def test_crlb_inversion(psf):
    # give a few photons to make the derivatives not explode
    xyz, weight = _random_positions(100, weight=10000)
    bg = torch.ones_like(weight) * 50

    crlb, _ = psf.crlb(xyz, weight, bg)
    crlb_alt, _ = psf.crlb(xyz, weight, bg, inversion=torch.pinverse)

    diff_inv = (crlb_alt - crlb).abs()

    np.testing.assert_allclose(crlb[:, :2], crlb_alt[:, :2], atol=1e-4)
    np.testing.assert_allclose(crlb[:, 2], crlb_alt[:, 2], rtol=0.1)
    np.testing.assert_allclose(crlb[:, 3], crlb_alt[:, 3], atol=1e2)
    np.testing.assert_allclose(crlb[:, 4], crlb_alt[:, 4], atol=1e-2)


@pytest.mark.plot
def test_frame_visual(psf):
    xyz, weight = _random_positions(10)

    frames_cpu = psf.forward(
        xyz=xyz,
        weight=torch.ones(len(xyz)),
        frame_ix=torch.zeros(len(xyz), dtype=torch.long),
    )

    plt.figure()
    plf.PlotFrameCoord(frames_cpu[0], pos_tar=xyz).plot()
    plt.title(
        "Random Frame sample.\n"
        "Should show a couple of emitters at\n"
        "random positions distributed over a frame."
    )
    plt.show()


@pytest.mark.plot
def test_roi_visual(psf):
    xyz, weight = _random_positions(1, 32, 32)

    roi = psf.forward_rois(xyz, weight)
    frame = psf.forward(xyz, weight)

    plt.figure()

    plt.subplot(121)
    plf.PlotFrameCoord(roi[0], pos_ini=psf.ref0.unsqueeze(0), labels=[None, None,
                                                                      "ROI ctr."]).plot()
    plt.title(
        f"Random ROI sample.\n"
        f"Should show a single emitter close to the reference point of the psf\n"
    )
    plt.legend()

    plt.subplot(122)
    plf.PlotFrameCoord(frame.squeeze(), pos_tar=xyz, labels=["Emitter pos."]).plot()

    plt.legend()
    plt.show()


@pytest.mark.plot
def test_roi_drv_visual(psf):
    xyz, weight = _random_positions(10, 32, 32)

    drv_rois, rois = psf.derivative(xyz, weight, torch.ones_like(weight))

    rix = random.randint(0, len(xyz))  # pick random sample
    dr = drv_rois[rix]
    r = rois[rix]
    xyzr = xyz[[rix]]

    plt.figure(figsize=(20, 12))

    plt.subplot(231)
    plf.PlotFrameCoord(r, pos_ini=psf.ref0.unsqueeze(0)).plot()
    plt.title(
        f"Random ROI sample.\n"
        f"Should show a single emitter close to the reference point of the psf.\n"
        f"Reference: {psf.ref0}"
    )

    plt.subplot(232)
    plf.PlotFrame(dr[0], plot_colorbar=True).plot()
    plt.title("d/dx")

    plt.subplot(233)
    plf.PlotFrame(dr[1], plot_colorbar=True).plot()
    plt.title("d/dy")

    plt.subplot(234)
    plf.PlotFrame(dr[2], plot_colorbar=True).plot()
    plt.title("d/dz")

    plt.subplot(235)
    plf.PlotFrame(dr[3], plot_colorbar=True).plot()
    plt.title("d/dphot")

    plt.subplot(236)
    plf.PlotFrame(dr[4], plot_colorbar=True).plot()
    plt.title("d/dbg")

    plt.show()
