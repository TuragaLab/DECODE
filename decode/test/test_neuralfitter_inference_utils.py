from unittest import mock

import torch
import pytest

from decode.neuralfitter import inference


@pytest.mark.parametrize("camera", [None, mock.MagicMock])
@pytest.mark.parametrize("auto_crop", [None, 8])
@pytest.mark.parametrize("mirror_dim", [None, -1])
def test_get_preprocessing(camera, auto_crop, mirror_dim):

    if camera is not None:
        camera = camera()

    with mock.patch('decode.simulation.camera.Photon2Camera.backward') as mock_cam:
        with mock.patch('decode.neuralfitter.frame_processing.AutoCenterCrop.forward') as mock_crop:
            with mock.patch('decode.neuralfitter.frame_processing.Mirror2D.forward') as mock_mirror:
                with mock.patch('decode.neuralfitter.scale_transform.AmplitudeRescale.forward') as mock_ampl:
                    pipeline = inference.utils.get_preprocessing((100., 5.), camera, auto_crop, mirror_dim)

                    x = torch.rand(2, 63, 64)
                    _ = pipeline.forward(x)

    if camera is not None:
        camera.backward.assert_called_once()  # here it is the variable because
    else:
        mock_cam.assert_not_called()

    if auto_crop is not None:
        mock_crop.assert_called_once()
    else:
        mock_crop.assert_not_called()

    if mirror_dim is not None:
        mock_mirror.assert_called_once()
    else:
        mock_mirror.assert_not_called()

    mock_ampl.assert_called_once()