import os

import pytest
import torch

import deepsmlm.simulation.engine
from deepsmlm import RandomEmitterSet

torch.multiprocessing.set_sharing_strategy('file_system')
import time
from torch.utils.data import Dataset
import pathlib

import deepsmlm.simulation.engine as engine

deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


class DummyDataset(Dataset):
    def __init__(self, n=10):
        self.frames = torch.rand((n, 32, 32))
        self.gt = torch.rand_like(self.frames)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        time.sleep(.1)
        return self.frames[item], self.gt[item]


class TestSampleStreamEngine:

    @staticmethod
    def fin():
        """Delete temp. folder after fixture is out of scope."""
        deepsmlm.simulation.engine.del_dir(deepsmlm_root + 'deepsmlm/test/assets/sim_engine/dummy_data')

    @pytest.fixture()
    def dummy_sim_engine(self, request):
        dummy_ds = DummyDataset(n=10)

        """Create folder for testing. In productive setting not okay, but for testing purpose it's fine."""
        temp_folder = pathlib.Path(deepsmlm_root + 'deepsmlm/test/assets/sim_engine')
        temp_folder.mkdir(exist_ok=True)

        can = engine.SampleStreamEngine(
            cache_dir=deepsmlm_root + 'deepsmlm/test/assets/sim_engine',
            exp_id='dummy_data',
            cpu_worker=4,
            buffer_size=3,
            ds_train=dummy_ds,
            ds_test=None
        )

        request.addfinalizer(self.fin)  # teardown
        return can

    @pytest.mark.slow
    def test_run(self, dummy_sim_engine):
        dummy_sim_engine.run(n_max=5)


class TestDatasetStreamEngine(TestSampleStreamEngine):

    @pytest.fixture()
    def dummy_sim_engine(self, request):
        class DummySimulation:
            def __init__(self):
                self.em_sampler = True

            @staticmethod
            def forward():
                em = RandomEmitterSet(1024)
                em.frame_ix = torch.randint_like(em.frame_ix, 0, 256)
                return torch.rand((256, 64, 64)), torch.rand((256, 64, 64)), em

        eng = engine.DatasetStreamEngine(
            cache_dir=deepsmlm_root + 'deepsmlm/test/assets/sim_engine',
            exp_id='dummy_data',
            buffer_size=3,
            sim_train=DummySimulation(),
            sim_test=DummySimulation()
        )

        # teardown
        request.addfinalizer(self.fin)
        return eng
