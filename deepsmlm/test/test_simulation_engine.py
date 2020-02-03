import os
import pytest
import torch
import time
from torch.utils.data import Dataset

import deepsmlm.test.utils_ci as tutil
import deepsmlm.simulation.engine as engine
import deepsmlm.generic.utils.data_utils as deepsmlm_utils

deepsmlm_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir)) + '/'


class DummyDataset(Dataset):
    def __init__(self, n=10):

        self.frames = torch.rand((n, 32, 32))
        self.gt = torch.arange(n)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, item):

        time.sleep(1)
        return self.frames[item], self.gt[item]


class TestSimulationEngine:

    @pytest.fixture(scope='class')
    def dummy_ds(self):
        return DummyDataset(n=10)

    def test_setup(self, dummy_ds):

        can = engine.SimulationEngine(
            cache_dir=deepsmlm_root + 'deepsmlm/test/assets/sim_engine',
            exp_id='dummy_data',
            cpu_worker=1,
            buffer_size=3,
            ds_train=dummy_ds,
            ds_test=None
        )

        deepsmlm_utils.del_dir(deepsmlm_root + 'deepsmlm/test/assets/sim_engine/dummy_data')
        return True

    def test_run(self, dummy_ds):
        can = engine.SimulationEngine(
            cache_dir=deepsmlm_root + 'deepsmlm/test/assets/sim_engine',
            exp_id='dummy_data',
            cpu_worker=4,
            buffer_size=3,
            ds_train=dummy_ds,
            ds_test=None
        )

        can.run(10)
        deepsmlm_utils.del_dir(deepsmlm_root + 'deepsmlm/test/assets/sim_engine/dummy_data')
