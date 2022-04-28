"""
Tests dataloader pipeline in multiprocessing and with cuda
"""
import time

import pytest
import torch
import torch.utils

import decode
import decode.neuralfitter
import decode.neuralfitter.utils.dataloader_customs


class Base:

    @pytest.fixture(params=[0, 4], ids=["single", "multi"])
    def dataloader(self, dataset, request):
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=32,
                                           num_workers=request.param,
                                           collate_fn=decode.neuralfitter.utils.dataloader_customs.smlm_collate)

    def test_iterate(self, dataloader):
        for _ in dataloader:
            time.sleep(0.1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Tests shipment to cuda")
    def test_iter_ship_cuda(self, dataloader):
        for batch in dataloader:
            for el in batch:
                if isinstance(el, torch.Tensor):
                    el = el.to('cuda')


class TestStaticDataset(Base):

    @pytest.fixture()
    def dataset(self):
        frames = torch.rand(1000, 32, 32)
        emitter = decode.RandomEmitterSet(10000)
        emitter.frame_ix = torch.randint_like(emitter.frame_ix, 0, 1000)

        ds = decode.neuralfitter.dataset.SMLMStaticDataset(frames=frames, emitter=emitter.split_in_frames(0, 999))

        return ds


class TestLiveDataset(Base):

    @pytest.fixture()
    def psf(self):
        extent = ((-0.5, 63.5), (-0.5, 63.5), (-500, 500))
        img_shape = (64, 64)

        return decode.simulation.psf_kernel.CubicSplinePSF(
            *extent[:2], img_shape, (15, 15, 50), coeff=torch.rand(33, 33, 100, 64), vx_size=(1., 1., 1.),
            device='cuda:0' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture()
    def em_sampler(self):
        prior = decode.simulation.structure_prior.RandomStructure((-0.5, 63.5), (-0.5, 63.5), (-500, 500))
        sampler = decode.simulation.emitter_generator.EmitterSamplerBlinking(
            structure=prior, intensity_mu_sig=(1000., 100.), lifetime=2., frame_range=(0, 1000), xy_unit='px',
            px_size=(100., 100.), em_avg=20.
        )
        return sampler

    @pytest.fixture()
    def background(self):
        return decode.simulation.background.UniformBackground((20., 100.))

    @pytest.fixture()
    def noise(self):
        return decode.simulation.camera.Photon2Camera(
            qe=0.9, spur_noise=0.02, em_gain=100., e_per_adu=45., baseline=100., read_sigma=74.4, photon_units=True,
            device='cpu')

    @pytest.fixture()
    def simulator(self, psf, em_sampler, background, noise):
        return decode.simulation.Simulation(psf=psf, em_sampler=em_sampler, background=background, noise=noise,
                                            frame_range=(0, 1000))

    @pytest.fixture()
    def emitter_processing(self):
        return decode.neuralfitter.em_filter.PhotonFilter(10.)

    @pytest.fixture()
    def frame_processing(self):
        return decode.neuralfitter.scale_transform.AmplitudeRescale(100., 5.)

    @pytest.fixture()
    def target_generator(self, psf):
        tar_gen = decode.neuralfitter.utils.processing.TransformSequence(
            [
                decode.neuralfitter.target_generator.ParameterListTarget(n_max=250,
                                                                         xextent=psf.xextent,
                                                                         yextent=psf.yextent,
                                                                         ix_low=0,
                                                                         ix_high=0,
                                                                         squeeze_batch_dim=True),

                decode.neuralfitter.target_generator.DisableAttributes(None),

                decode.neuralfitter.scale_transform.ParameterListRescale(phot_max=20000,
                                                                         z_max=1000.,
                                                                         bg_max=120.)
            ])

        return tar_gen

    @pytest.fixture()
    def dataset(self, simulator, emitter_processing, frame_processing, target_generator):
        ds = decode.neuralfitter.dataset.SMLMLiveDataset(simulator=simulator,
                                                         em_proc=emitter_processing, frame_proc=frame_processing,
                                                         bg_frame_proc=None, tar_gen=target_generator, weight_gen=None,
                                                         frame_window=3, pad='same', return_em=False)
        ds.sample(True)
        return ds


class TestAprioriDataset(TestLiveDataset):

    @pytest.fixture()
    def simulator(self, psf, em_sampler, background, noise):
        return decode.simulation.Simulation(psf=psf, em_sampler=em_sampler, background=background, noise=noise,
                                            frame_range=(0, 1000))

    @pytest.fixture()
    def dataset(self, simulator, emitter_processing, frame_processing, target_generator):

        target_generator.com[0].ix_low = 0
        target_generator.com[0].ix_high = 1000
        target_generator.com[0].squeeze_batch_dim = False
        target_generator.com[0].sanity_check()

        ds = decode.neuralfitter.dataset.SMLMAPrioriDataset(simulator=simulator,
                                                            em_proc=emitter_processing, frame_proc=frame_processing,
                                                            bg_frame_proc=None, tar_gen=target_generator,
                                                            weight_gen=None,
                                                            frame_window=3, pad='same', return_em=False)

        ds.sample(True)
        return ds
