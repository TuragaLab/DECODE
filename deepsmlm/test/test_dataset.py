import pytest
import torch
import pathlib
import pickle

import deepsmlm
import deepsmlm.generic.utils.test_utils as test_utils
import deepsmlm.generic.emitter
import deepsmlm.neuralfitter.target_generator
import deepsmlm.neuralfitter.dataset as can  # candidate
from deepsmlm.neuralfitter.engine import SMLMTrainingEngine
from deepsmlm.neuralfitter.pre_processing import Identity

deepsmlm_root = pathlib.Path(deepsmlm.__file__).parent.parent  # 'repo' directory


class TestDataset:

    @pytest.fixture(scope='class', params=[1, 3, 5])
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

        dataset = can.SMLMStaticDataset(frames=torch.rand((n, 1, 32, 32)), em=em.split_in_frames(0, n - 1),
                                        frame_proc=DummyFrameProc, em_proc=DummyEmProc,
                                        tar_gen=deepsmlm.neuralfitter.target_generator.SinglePxEmbedding(
                                            (-0.5, 31.5), (-0.5, 31.5), (32, 32)), frame_window=request.param, return_em=True)

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

        assert frs.size(0) == ds.frame_window


class TestInferenceDataset(TestDataset):

    @pytest.fixture(scope='class', params=[1, 3, 5])
    def ds(self, request):
        class DummyFrameProc:
            def forward(x: torch.Tensor):
                return x.clamp(0., 0.5)

        n = 100

        dataset = can.InferenceDataset(frames=torch.rand((n, 1, 32, 32)), frame_proc=DummyFrameProc,
                                       frame_window=request.param)

        return dataset


class TestDatasetEngineDataset:

    @pytest.fixture(scope='class', params=[1, 3, 5])
    def ds(self, request):

        dataset = can.SMLMDatasetEngineDataset(engine=None, em_proc=Identity(), frame_proc=Identity(), tar_gen=None, weight_gen=None,
                                               frame_window=request.param, pad=None, return_em=True)
        dataset._x_in = torch.rand((2048, 64, 64))

        em = deepsmlm.RandomEmitterSet(20000)
        em.frame_ix = torch.randint_like(em.frame_ix, low=0, high=2048)
        dataset._tar_em = em

        dataset._aux = [torch.rand_like(dataset._x_in)]
        return dataset

    def test_len(self, ds):

        ds.pad = None
        assert len(ds) == ds._x_in.size(0) - ds.frame_window + 1

        ds.pad = 'same'
        assert len(ds) == ds._x_in.size(0)

    def test_get_samples(self, ds):

        x, y, w, em = ds.__getitem__(500)  # frames, target, weight, emitters

        assert x.size(0) == ds.frame_window
        assert em.frame_ix.unique().numel() == 1


