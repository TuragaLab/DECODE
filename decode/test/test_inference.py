import torch
import pytest

from decode.generic import emitter
from decode.generic.process import Identity
from decode.neuralfitter.inference import inference
from decode.neuralfitter import post_processing


class TestInfer:

    @pytest.fixture()
    def infer(self):
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=False)

        return inference.Infer(model=model, ch_in=3, frame_proc=None, post_proc=post_processing.NoPostProcessing(),
                               device='cuda' if torch.cuda.is_available() else 'cpu')

    def test_forward_em(self, infer):
        """
        Tests the inference wrapper

        Args:
            infer:

        """

        """Setup"""
        frames = torch.rand((100, 64, 64))

        """Run"""
        em = infer.forward(frames)

        """Assertions"""
        assert isinstance(em, emitter.EmitterSet)

    def test_forward_frames(self, infer):
        infer.post_proc = Identity()
        infer.forward_cat = infer._setup_forward_cat('frames')

        out = infer.forward(torch.rand((100, 64, 64)))

        """Assertions"""
        assert isinstance(out, torch.Tensor)
        assert out.size() == torch.Size((100, 1, 64, 64))

