import pytest
import torch

import deepsmlm.generic.emitter
import deepsmlm.neuralfitter.target_generator
import deepsmlm.neuralfitter.dataset as can  # candidate


class TestDataset:

    @pytest.fixture(scope='class', params=[None, 1, 3, 5])
    def ds(self, request):
        class DummyFrameProc:
            def forward(x: torch.Tensor):
                return x.clamp(0., 0.5)

        class DummyEmProc:
            def forward(em: deepsmlm.generic.emitter.EmitterSet):
                return em[em.xyz[:, 0] <= 16]

        n = 100

        em = deepsmlm.generic.emitter.RandomEmitterSet(n * 100)
        em.frame_ix = torch.randint_like(em.frame_ix, n + 1)

        dataset = can.SMLMStaticDataset(frames=torch.rand((n, 1, 32, 32)),
                                        em=em.split_in_frames(0, n - 1),
                                        tar_gen=deepsmlm.neuralfitter.target_generator.SinglePxEmbedding(
                                            (-0.5, 31.5), (-0.5, 31.5), (32, 32)),
                                        em_proc=DummyEmProc,
                                        frame_proc=DummyFrameProc,
                                        fwindow=request.param,
                                        return_em=True)

        return dataset

    def test_len(self, ds):
        """
        Test length

        Args:
            ds: fixture

        """
        assert len(ds) == 100

    @pytest.mark.parametrize("ix", [0, 10, 15, 99])  # border cases and a few in the middle
    def test_frames(self, ds, ix):
        """
        Most important get_item test

        Args:
            ds: fixture

        """
        """Prepare"""

        """Run"""
        sample = ds[ix]
        if isinstance(sample, tuple):  # potentially multiple return arguments
            frs = ds[ix][0]
        else:
            frs = sample

        """Tests"""
        assert frs.dim() == 3, "Wrong dimensionality."
        assert frs.max() <= 0.5

        if ds.multi_frame:
            assert frs.size(0) == ds.multi_frame
        else:
            assert frs.size(0) == 1


class TestInferenceDataset(TestDataset):

    @pytest.fixture(scope='class', params=[None, 1, 3, 5])
    def ds(self, request):
        class DummyFrameProc:
            def forward(x: torch.Tensor):
                return x.clamp(0., 0.5)

        n = 100

        dataset = can.InferenceDataset(frames=torch.rand((n, 1, 32, 32)),
                                        frame_proc=DummyFrameProc,
                                        fwindow=request.param)

        return dataset
