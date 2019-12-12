import pytest
import torch
import torch.utils
import multiprocessing as mp
import time
import os

import deepsmlm.generic.background
import deepsmlm.generic.noise
import deepsmlm.neuralfitter.dataset as ds
import deepsmlm.generic.psf_kernel as psf
import deepsmlm.generic.noise as noise_bg
import deepsmlm.simulation.simulator as sim
import deepsmlm.simulation.structure_prior as structure_prior
import deepsmlm.generic.emitter as em
from deepsmlm.neuralfitter.pre_processing import N2C
from deepsmlm.neuralfitter.unifed_data_generator import UnifiedDataset

deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'


class TestDataset:

    @pytest.fixture(scope='class')
    def dataset(self):
        pointspread = psf.GaussianExpect((-0.5, 31.5), (-0.5, 31.5), None, (32, 32), 5.)
        noise = deepsmlm.generic.noise.Poisson(bg_uniform=10)
        simulator = sim.Simulation(None, ((-0.5, 31.5), (-0.5, 31.5), None), pointspread, noise, frame_range=(-1, 1),
                                   poolsize=0)
        target_generator = psf.DeltaPSF((-0.5, 31.5), (-0.5, 31.5), None, (32, 32))

        class Prior:
            def pop(self):
                return em.RandomEmitterSet(32, 32)

        prior = Prior()

        return ds.SMLMDatasetOnFly(None, prior, simulator, 32, N2C(), target_generator, None, return_em_tar=False)

    def test_lifetime(self, dataset):
        dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        epochs = 6

        for epoch in range(epochs):
            for idx, (sample, target) in enumerate(dl):
                print('Epoch {}, idx {}, data.shape {}'.format(epoch, idx, sample.shape))

            dl.dataset.step()