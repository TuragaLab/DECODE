import pathlib

import pytest
import torch

import deepsmlm
import deepsmlm.generic.emitter
import deepsmlm.neuralfitter.dataset as can  # candidate
import deepsmlm.neuralfitter.target_generator
from deepsmlm.generic.process import Identity
from deepsmlm.simulation.simulator import Simulation

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

        class DummyWeight:
            def forward(self, *args, **kwargs):
                return torch.rand((1, 1, 32, 32))

        n = 100

        em = deepsmlm.generic.emitter.RandomEmitterSet(n * 100)
        em.frame_ix = torch.randint_like(em.frame_ix, n + 1)

        dataset = can.SMLMStaticDataset(frames=torch.rand((n, 32, 32)), emitter=em.split_in_frames(0, n - 1),
                                        frame_proc=DummyFrameProc, bg_frame_proc=None, em_proc=DummyEmProc,
                                        tar_gen=deepsmlm.neuralfitter.target_generator.UnifiedEmbeddingTarget(
                                            (-0.5, 31.5), (-0.5, 31.5), (32, 32), roi_size=1, ix_low=0, ix_high=0),
                                        weight_gen=DummyWeight(),
                                        frame_window=request.param,
                                        pad='same',
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

        assert frs.size(0) == ds.frame_window


class TestInferenceDataset(TestDataset):

    @pytest.fixture(scope='class', params=[1, 3, 5])
    def ds(self, request):
        class DummyFrameProc:
            def forward(x: torch.Tensor):
                return x.clamp(0., 0.5)

        n = 100

        dataset = can.InferenceDataset(frames=torch.rand((n, 32, 32)), frame_proc=DummyFrameProc,
                                       frame_window=request.param)

        return dataset


class TestSMLMLiveDataset:

    @pytest.fixture()
    def ds(self):

        class DummySimulation(Simulation):
            def __init__(self):
                pass

            def sample(self):
                em = deepsmlm.RandomEmitterSet(1024)
                em.frame_ix = torch.randint_like(em.frame_ix, 0, 256)
                frames, bg_frames = self.forward(em)

                return em, frames, bg_frames

            def forward(self, em):
                return torch.rand((256, 64, 64)), torch.rand((256, 64, 64))

        class DummyTarAndWeightGen:
            def forward(self, *args):
                return torch.rand((6, 64, 64))

        dataset = can.SMLMLiveDataset(simulator=DummySimulation(), em_proc=None, frame_proc=None, bg_frame_proc=None,
                                      tar_gen=DummyTarAndWeightGen(), weight_gen=DummyTarAndWeightGen(), frame_window=1,
                                      pad=None)

        return dataset

    @pytest.mark.parametrize("window", [1, 3, 5])
    @pytest.mark.parametrize("pad", [None, "same"])
    def test_sample(self, ds, window, pad):

        """Setup"""
        ds.frame_window = window
        ds.pad = pad

        """Run"""
        ds.sample()

        """Assert"""
        if pad == 'same':
            assert len(ds) == 256

        elif pad is None:
            assert len(ds) == 256 - window + 1

    @pytest.mark.parametrize("window", [1, 3, 5])
    @pytest.mark.parametrize("ix", [0, 50, 200])
    @pytest.mark.parametrize("return_em", [False, True])
    def test_get_item(self, ds, window, ix, return_em):

        """Setup"""
        ds.frame_window = window
        ds.return_em = return_em

        """Run"""
        ds.sample()
        sample_out = ds[ix]

        """Assert"""
        assert len(sample_out) == 4 if return_em else 3
        if return_em:  # unpack
            x, y_tar, weight, emitter = sample_out
            assert emitter.frame_ix.unique().numel() <= 1
        else:
            x, y_tar, weight = sample_out

        assert x.dim() == 3
        assert y_tar.dim() == 3
        assert weight.dim() == 3


class TestLiveSampleDataset:
    @pytest.fixture()
    def ds(self):
        class DummySimulation(Simulation):
            def __init__(self):
                pass

            def sample(self):
                em = deepsmlm.RandomEmitterSet(150)
                em.frame_ix = torch.randint_like(em.frame_ix, -1, 2)
                frames, bg_frames = self.forward(em)

                return em, frames, bg_frames

            def forward(self, em):
                return torch.rand((3, 64, 64)), torch.rand((3, 64, 64))

        class DummyTarAndWeightGen:
            def forward(self, *args):
                return torch.rand((6, 64, 64))

        dataset = can.SMLMLiveSampleDataset(ds_len=1000, simulator=DummySimulation(), em_proc=None, frame_proc=None, bg_frame_proc=None,
                                            tar_gen=DummyTarAndWeightGen(), weight_gen=DummyTarAndWeightGen(), frame_window=3)

        return dataset

    def test_len(self, ds):
        assert len(ds) == 1000

    @pytest.mark.parametrize("window", [1, 3, 5])
    @pytest.mark.parametrize("return_em", [False, True])
    def test_getitem(self, ds, window, return_em):

        """Setup"""
        ds.return_em = return_em
        ds.frame_window = window

        r_ix = torch.randint(0, len(ds), size=(1, )).item()

        sample_out = ds[r_ix]

        """Assert"""
        assert len(sample_out) == 4 if return_em else 3
        if return_em:  # unpack
            x, y_tar, weight, emitter = sample_out
            assert emitter.frame_ix.unique().numel() <= 1
        else:
            x, y_tar, weight = sample_out

        assert x.dim() == 3
        assert y_tar.dim() == 3
        assert weight.dim() == 3
