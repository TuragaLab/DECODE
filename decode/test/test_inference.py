import torch
import pytest

from decode.generic import emitter
from decode.neuralfitter.inference import inference
from decode.neuralfitter import post_processing


class TestInfer:

    @pytest.fixture()
    def infer(self):
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=False)

        return inference.Infer(model=model, ch_in=3, frame_proc=None, post_proc=post_processing.NoPostProcessing(),
                               device='cuda' if torch.cuda.is_available() else 'cpu')

    def test_forward(self, infer):
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

