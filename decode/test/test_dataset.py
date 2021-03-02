import pathlib

import pytest
import torch

import decode
import decode.generic.emitter
import decode.neuralfitter.dataset as can  # candidate
import decode.neuralfitter.target_generator
from decode.neuralfitter import em_filter
from decode.simulation.simulator import Simulation

decode_root = pathlib.Path(decode.__file__).parent.parent  # 'repo' directory


class TestDataset:

    @pytest.fixture(scope='class', params=[1, 3, 5])
    def ds(self, request):
        class DummyFrameProc:
            def forward(x: torch.Tensor):
                return x.clamp(0., 0.5)

        class DummyEmProc:
            def forward(em: decode.generic.emitter.EmitterSet):
                return em[em.xyz[:, 0] <= 16]

        class DummyWeight:
            def forward(self, *args, **kwargs):
                return torch.rand((1, 1, 32, 32))

        n = 100

        em = decode.generic.emitter.RandomEmitterSet(n * 100)
        em.frame_ix = torch.randint_like(em.frame_ix, n + 1)

        dataset = can.SMLMStaticDataset(frames=torch.rand((n, 32, 32)), emitter=em.split_in_frames(0, n - 1),
                                        frame_proc=DummyFrameProc, bg_frame_proc=None, em_proc=DummyEmProc,
                                        tar_gen=decode.neuralfitter.target_generator.UnifiedEmbeddingTarget(
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
                em = decode.RandomEmitterSet(1024)
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


class TestSMLMAPrioriDataset:

    @pytest.fixture()
    def ds(self):
        class DummySimulation(Simulation):
            def __init__(self):
                return

            def sample(self):
                em = decode.RandomEmitterSet(1000)
                em.phot = torch.rand_like(em.phot) * 10000
                em.frame_ix = torch.randint_like(em.frame_ix, -10, 5000)
                frames, bg_frames = self.forward(em)

                return em, frames, bg_frames

            def forward(self, em):
                return torch.rand((5000, 64, 64)), torch.rand((5000, 64, 64))

        class DummyFrameProc:
            def forward(self, x):
                return x / 2

        class DummyEMProc:
            def forward(self, x):
                return x[:500]

        class DummyTargen:
            def forward(self, *args):
                return torch.rand((5000, 64, 64))

        dataset = can.SMLMAPrioriDataset(simulator=DummySimulation(), em_proc=DummyEMProc(),
                                         frame_proc=DummyFrameProc(), bg_frame_proc=DummyFrameProc(),
                                         tar_gen=DummyTargen(), weight_gen=None,
                                         frame_window=3, pad=None, return_em=False)

        return dataset

    def test_sample(self, ds):

        ds.sample()

        """Assertions"""
        assert isinstance(ds._emitter, decode.generic.emitter.EmitterSet)
        assert isinstance(ds._em_split, list)

    def test_len(self, ds):

        ds.sample()
        assert len(ds) == 5000 - (ds.frame_window - 1)


class TestLiveSampleDataset:
    @pytest.fixture()
    def ds(self):
        class DummySimulation(Simulation):
            def __init__(self):
                pass

            def sample(self):
                em = decode.RandomEmitterSet(150)
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
