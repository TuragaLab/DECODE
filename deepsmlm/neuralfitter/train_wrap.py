import datetime
import os
import sys
import time

from comet_ml import Experiment, OfflineExperiment
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepsmlm.generic.inout.load_save_emitter import NumpyInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.noise import Poisson
from deepsmlm.generic.psf_kernel import ListPseudoPSFInSize
from deepsmlm.neuralfitter.arguments import InOutParameter, HyperParamter, SimulationParam
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.neuralfitter.dataset import SMLMDatasetOnFly
from deepsmlm.neuralfitter.losscollection import MultiScaleLaplaceLoss
from deepsmlm.neuralfitter.model import DenseLoco
from deepsmlm.neuralfitter.pre_processing import N2C, SingleEmitterOnlyZ
from deepsmlm.neuralfitter.train_test import train, test
from deepsmlm.simulator.emittergenerator import EmitterPopperMultiFrame
from deepsmlm.simulator.simulator import Simulation

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


if __name__ == '__main__':

    """Set ur basic parameters"""
    io_par = InOutParameter(root=deepsmlm_root,
                            log_comment='dummy',
                            data_mode='online',
                            data_set=None,
                            model_out=deepsmlm_root + 'network/2019-03-26/model_0.pt',
                            model_init=None)

    hy_par = HyperParamter(dimensions=3,
                           channels=3,
                           max_emitters=256,
                           batch_size=128,
                           test_size=32,
                           num_epochs=10000,
                           lr=1E-4,
                           device=torch.device('cuda'))

    sim_par = SimulationParam(pseudo_data_size=(256*16*16 + 1024),
                              emitter_extent=((-0.5, 31.5), (-0.5, 31.5), (-500, 500)),
                              psf_extent=((-0.5, 31.5), (-0.5, 31.5), (-750., 750.)),
                              img_size=(32, 32),
                              density=0.01,
                              photon_range=(4000, 8000),
                              bg_pois=15,
                              calibration=deepsmlm_root +
                                          'data/Cubic Spline Coefficients/2019-02-20/60xOil_sampleHolderInv__CC0.140_1_MMStack.ome_3dcal.mat')

    """Log System"""
    log_dir = deepsmlm_root + 'log/' + str(datetime.datetime.now())[:16]

    experiment = Experiment(project_name='deepsmlm', workspace='haydnspass', auto_metric_logging=False)
    # experiment = OfflineExperiment(project_name='deepsmlm',
    #                                workspace='haydnspass',
    #                                offline_directory=deepsmlm_root + 'log/')

    experiment.log_parameters(hy_par._asdict(), prefix='Hyper Parameters')
    experiment.log_parameters(sim_par._asdict(), prefix='Simulation Parameters')
    experiment.log_parameters(io_par._asdict(), prefix='I/O Parameters')
    logger = SummaryWriter(log_dir, comment=io_par.log_comment)

    if io_par.data_mode == 'precomputed':
        """Load Data from binary."""
        emitter, extent, frames = NumpyInterface().load_binary(io_par.data_set)

        target_generator = SingleEmitterOnlyZ()
        # target_generator = ZasSimpleRegression()
        # target_generator = ListPseudoPSF(zero_fill_to_size=64, dim=3)

        data_smlm = SMLMDataset(emitter, extent, frames, target_generator,
                                multi_frame_output=False,
                                dimensionality=None)

        train_size = data_smlm.__len__() - hy_par.test_size
        train_data_smlm, test_data_smlm = torch.utils.data.random_split(data_smlm, [train_size, hy_par.test_size])

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=hy_par.batch_size, shuffle=True, num_workers=24, pin_memory=True)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=hy_par.test_size, shuffle=False, num_workers=12, pin_memory=True)

    elif io_par.data_mode == 'online':
        """Load 'Dataset' which is generated on the fly."""
        smap_psf = SMAPSplineCoefficient(sim_par.calibration)
        psf = smap_psf.init_spline(sim_par.psf_extent[0], sim_par.psf_extent[1], sim_par.img_size)
        noise = Poisson(bg_uniform=sim_par.bg_pois)

        # psf = GaussianExpect(xextent=extent[0], yextent=extent[1], zextent=None, img_shape=hyper['image_size'],
        #                      sigma_0=(1.5, 1.5))
        # prior = EmitterPopper(sim_extent[0], sim_extent[1], sim_extent[2], density=0.003, photon_range=(4000, 10000))
        prior = EmitterPopperMultiFrame(sim_par.emitter_extent[0],
                                        sim_par.emitter_extent[1],
                                        sim_par.emitter_extent[2],
                                        density=sim_par.density,
                                        photon_range=sim_par.photon_range,
                                        lifetime=1,
                                        num_frames=3)
        simulator = Simulation(None,
                               sim_par.emitter_extent,
                               psf,
                               noise,
                               poolsize=0,
                               frame_range=(-1, 1))

        input_preparation = N2C()

        target_generator = ListPseudoPSFInSize(sim_par.emitter_extent[0],
                                               sim_par.emitter_extent[1],
                                               sim_par.emitter_extent[2], zts=hy_par.max_emitters, dim=3)

        train_size = sim_par.pseudo_data_size - hy_par.test_size

        train_data_smlm = SMLMDatasetOnFly(None, prior, simulator, train_size,
                                           input_preparation, target_generator, None, reuse=False)

        test_data_smlm = SMLMDatasetOnFly(None, prior, simulator, hy_par.test_size,
                                          input_preparation, target_generator, None, reuse=True)

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=hy_par.batch_size, shuffle=False, num_workers=12, pin_memory=True)
        test_loader = DataLoader(test_data_smlm,
                                 batch_size=hy_par.test_size, shuffle=False, num_workers=8, pin_memory=True)

    else:
        raise NameError("You used the wrong switch of how to get the training data.")

    """Model load and save interface."""
    model = DenseLoco(extent=sim_par.psf_extent,
                      ch_in=hy_par.channels,
                      dim_out=hy_par.dimensions,
                      max_num_emitter=hy_par.max_emitters)

    """Log the model"""
    dummy = torch.rand((32, 3, *sim_par.img_size), requires_grad=True)
    logger.add_graph(model, dummy, False)

    model_ls = LoadSaveModel(model, output_file=io_par.model_out, input_file=io_par.model_init)
    model = model_ls.load_init()

    model = model.to(hy_par.device)

    optimiser = Adam(model.parameters(), lr=hy_par.lr)

    """Loss function."""
    criterion = MultiScaleLaplaceLoss(kernel_sigmas=(0.64, 3.20, 6.4, 19.2)).return_criterion()

    """Learning Rate Scheduling"""
    scheduler = ReduceLROnPlateau(optimiser,
                                  mode='min',
                                  factor=0.1,
                                  patience=25,
                                  threshold=0.0001,
                                  cooldown=10,
                                  verbose=True)

    """Ask if everything is correct before we start."""
    for i in range(hy_par.num_epochs):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        train(train_loader, model, optimiser, criterion, i, hy_par, logger, experiment)
        val_loss = test(test_loader, model, criterion, i, hy_par, logger, experiment)
        scheduler.step(val_loss)

        model_ls.save(model)

    experiment.end()
