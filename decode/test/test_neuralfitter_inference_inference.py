import threading
import time
from unittest import mock

import pytest
import tifffile
import torch

from decode.generic import emitter
from decode.generic import test_utils
from decode.generic.process import Identity
from decode.neuralfitter import post_processing
from decode.neuralfitter.inference import inference
from decode.utils import frames_io

from .test_utils_frames_io import online_tiff_writer


class TestInfer:

    @pytest.fixture()
    def model(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.parameter.Parameter(torch.rand(1, 1, 1, 1))

            def forward(self, x):
                assert x.dim() == 4
                assert x.size(1) == 3

                return self.dummy * torch.rand_like(x[:, [0]])

        return DummyModel()

    @pytest.fixture()
    def infer(self, model):
        return inference.Infer(model=model, ch_in=3, frame_proc=None, post_proc=post_processing.NoPostProcessing(),
                               device='cuda' if torch.cuda.is_available() else 'cpu')

    def test_forward_em(self, infer):
        """
        Tests the inference wrapper

        Args:
            infer:

        """

        frames = torch.rand((100, 64, 64))

        em = infer.forward(frames)

        assert isinstance(em, emitter.EmitterSet)

    @pytest.mark.parametrize("batch_size", [16, 'auto'])
    def test_forward_frames(self, infer, batch_size):
        # reinit because now we output frames
        infer = inference.Infer(model=infer.model, batch_size=batch_size, ch_in=3,
                                frame_proc=None, post_proc=Identity(), forward_cat='frames',
                                device='cuda' if torch.cuda.is_available() else 'cpu')

        out = infer.forward(torch.rand((100, 64, 64)))

        assert isinstance(out, torch.Tensor)
        assert out.size() == torch.Size((100, 1, 64, 64))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA.")
    def test_get_max_batch_size(self, infer):
        infer.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                     in_channels=3, out_channels=1, init_features=32, pretrained=False)

        bs = infer.get_max_batch_size(infer.model.cuda(), (3, 256, 256), 1, 5024)

        assert 16 <= bs <= 1024


class TestLiveInfer(TestInfer):

    @pytest.fixture()
    def infer(self, model):
        return inference.LiveInfer(
            model, ch_in=3, stream=None, time_wait=1,
            safety_buffer=30,
            frame_proc=None, post_proc=post_processing.NoPostProcessing(),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def test_forward_em(self, infer):
        infer._stream = mock.MagicMock()
        infer.forward(torch.rand(100, 64, 64))

        args, _ = infer._stream.call_args
        assert isinstance(args[0], emitter.EmitterSet)

    def test_forward_frames(self):
        return

    @pytest.mark.skip(reason="Currently not stable because of simultaneous read/write. "
                             "Revisit when buffer is implemented.")
    @pytest.mark.skipif(int(tifffile.__version__[:4]) <= 2020, reason="Online writer does not work with this version.")
    def test_forward_online(self, infer, tmpdir):
        path = tmpdir / 'online.tiff'
        tiff_writer = threading.Thread(target=online_tiff_writer, args=[path, 10, 0.5])
        tiff_writer.start()

        frames = frames_io.TiffTensor(path)
        while not test_utils.file_loadable(path, tifffile.TiffFile, mode='rb',
                                           exceptions=(KeyError, tifffile.TiffFileError)):
            time.sleep(1)

        infer._stream = mock.MagicMock()
        with mock.patch.object(inference.Infer, 'forward') as mock_forward:
            infer.forward(frames)

        tiff_writer.join()  # wait for last frame writing

        # check that first call of inference starts with 0 index for frames
        args, _ = infer._stream.call_args_list[0]
        assert args[1] == 0

        # check that last call of inference ends with last index of frames
        args, _ = infer._stream.call_args_list[-1]
        assert args[2] == 1000
