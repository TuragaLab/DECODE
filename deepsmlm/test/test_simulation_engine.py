import os

import pytest
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import time
from torch.utils.data import Dataset
import pathlib

import deepsmlm.simulation.engine as engine
import deepsmlm.generic.utils.data_utils as deepsmlm_utils

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
        time.sleep(.3)
        return self.frames[item], self.gt[item]


class TestSimulationEngine:

    @pytest.fixture()
    def dummy_sim_engine(self, request):
        def fin():
            """Delete temp. folder after fixture is out of scope."""
            deepsmlm_utils.del_dir(deepsmlm_root + 'deepsmlm/test/assets/sim_engine/dummy_data')

        dummy_ds = DummyDataset(n=10)

        """Create folder for testing. In productive setting not okay, but for testing purpose it's fine."""
        temp_folder = pathlib.Path(deepsmlm_root + 'deepsmlm/test/assets/sim_engine')
        temp_folder.mkdir(exist_ok=True)

        can = engine.SimulationEngine(
            cache_dir=deepsmlm_root + 'deepsmlm/test/assets/sim_engine',
            exp_id='dummy_data',
            cpu_worker=8,
            buffer_size=3,
            ds_train=dummy_ds,
            ds_test=None
        )

        request.addfinalizer(fin)  # teardown
        return can

    @pytest.mark.slow
    def test_run(self, dummy_sim_engine):
        dummy_sim_engine.run(n_max=10)
