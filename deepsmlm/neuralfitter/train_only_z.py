import datetime
import os
import time
from comet_ml import Experiment, OfflineExperiment

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.inout.load_save_emitter import NumpyInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.generic.noise import Poisson
from deepsmlm.generic.psf_kernel import ListPseudoPSFInSize, DeltaPSF
from deepsmlm.generic.utils.scheduler import ScheduleSimulation
from deepsmlm.neuralfitter.arguments import InOutParameter, HyperParamter, SimulationParam, LoggerParameter, \
    SchedulerParameter
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.generic.utils.data_utils import smlm_collate
from deepsmlm.neuralfitter.dataset import SMLMDatasetOnFly
from deepsmlm.neuralfitter.losscollection import MultiScaleLaplaceLoss, BumpMSELoss, MaskedOnlyZLoss
from deepsmlm.neuralfitter.models.model import DenseLoco, USMLM, USMLMLoco
from deepsmlm.neuralfitter.models.model_beta import DenseNetZPrediction, SMNET
from deepsmlm.neuralfitter.pre_processing import N2C, SingleEmitterOnlyZ, ZPrediction, ZasOneHot, EasyZ
from deepsmlm.simulation.emittergenerator import EmitterPopper, EmitterPopperMultiFrame
from deepsmlm.simulation.structure_prior import DiscreteZStructure, RandomStructure
from deepsmlm.simulation.simulator import Simulation
from deepsmlm.generic.plotting.frame_coord import PlotFrameCoord, PlotCoordinates3D
from deepsmlm.evaluation.evaluation import AverageMeter

DEBUG = True
LOG = True
TENSORBOARD = True
LOG_FIGURES = True

LOG = True if DEBUG else LOG
WRITE_TO_LOG = False

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


def train(train_loader, model, optimizer, criterion, epoch, hy_par, logger, experiment, calc_new):
    last_print_time = 0
    loss_values = []
    step_batch = epoch * train_loader.__len__()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target, em_tar, _) in enumerate(train_loader):

        if (epoch == 0) and (i == 0) and LOG:
            """Save a batch to see what we input into the network."""
            debug_file = deepsmlm_root + 'data/debug.pt'
            torch.save((input, target), debug_file)
            print("LOG: I saved a batch for you. Look what the network sees for verification purpose.")

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(hy_par.device)
        if type(target) is torch.Tensor:
            target = target.to(hy_par.device)
        elif type(target) in (tuple, list):
            target = (target[0].to(hy_par.device), target[1].to(hy_par.device))
        else:
            raise TypeError("Not supported type to push to cuda.")

        # compute output
        output = model(input)
        loss = criterion(output, target, input[:, 1, :, :])

        # record loss
        losses.update(loss.item(), input.size(0))
        loss_values.append(loss.item())

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i in [0, 1, 2, 10]) or (time.time() > last_print_time + 5):  # print the first few batches plus after 5s
            last_print_time = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4e} ({loss.avg:.4e})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        """Log Learning Rate, Benchmarks etc."""
        if i % 10 == 0:
            experiment.log_metric('learning/train_10_batch_loss', np.mean(loss_values), step=step_batch)

            logger.add_scalar('learning/train_loss', np.mean(loss_values), step_batch)
            logger.add_scalar('data/eval_time', batch_time.val, step_batch)
            logger.add_scalar('data/data_time', data_time.val, step_batch)
            loss_values.clear()

        if i == 0:
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    logger.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        logger.add_histogram(name + '.grad', param.grad, epoch)

        step_batch += 1


def test(val_loader, model, criterion, epoch, hy_par, logger, experiment, post_processor):
    """
        Taken from: https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
        """
    experiment.set_step(epoch)

    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, em_tar, _) in enumerate(val_loader):

            input = input.to(hy_par.device)
            if type(target) is torch.Tensor:
                target = target.to(hy_par.device)
            elif type(target) in (tuple, list):
                target = (target[0].to(hy_par.device), target[1].to(hy_par.device))
            else:
                raise TypeError("Not supported type to push to cuda.")

            # compute output
            output = model(input)
            loss = criterion(output, target, input[:, 1, :, :])

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # do evaluation
            # pred = post_processor(output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i in [0, 1, 2, 10]) or (i % 200 == 0):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4e} ({loss.avg:.4e})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))
                experiment.log_metric('learning/test_loss', losses.val, epoch)
    logger.add_scalar('learning/test_loss', losses.val, epoch)
    return losses.avg


if __name__ == '__main__':

    """Set ur basic parameters"""
    io_par = InOutParameter(
        root=deepsmlm_root,
        log_comment='',
        data_mode='online',
        data_set=None,
        model_out=deepsmlm_root + 'network/2019-04-29/z_only_xymask.pt',
        model_init=None)

    log_par = LoggerParameter(
        tags=['Z', 'Coord', 'DenseNet'])

    sched_par = SchedulerParameter(
        lr_factor=0.999999999999,
        lr_patience=10,
        lr_threshold=0.0025,
        lr_cooldown=10,
        lr_verbose=True,
        sim_factor=1,
        sim_patience=1,
        sim_threshold=0,
        sim_cooldown=10,
        sim_verbose=True,
        sim_disabled=True,
        sim_max_value=50,
    )

    hy_par = HyperParamter(
        dimensions=3,
        channels=3,
        max_emitters=128,
        min_phot=600.,
        data_lifetime=10,
        upscaling=1,
        upscaling_mode='nearest',
        batch_size=16,
        test_size=256,
        num_epochs=10000,
        lr=1E-5,
        device=torch.device('cuda'))

    sim_par = SimulationParam(
        pseudo_data_size=(256*16 + 256),  # (256*256 + 512),
        emitter_extent=((2.5, 28.5), (2.5, 28.5), (200., 750.)),
        psf_extent=((-0.5, 31.5), (-0.5, 31.5), (-750., 750.)),
        img_size=(32, 32),
        density=0,
        emitter_av=5,
        photon_range=(4000, 10000),
        bg_pois=10,
        calibration=deepsmlm_root +
                    'data/Cubic Spline Coefficients/2019-02-20/60xOil_sampleHolderInv__CC0.140_1_MMStack.ome_3dcal.mat')

    """Log System"""
    log_dir = deepsmlm_root + 'log/' + str(datetime.datetime.now())[:16]

    experiment = Experiment(project_name='deepsmlm', workspace='haydnspass',
                            auto_metric_logging=False, disabled=(not WRITE_TO_LOG))
    # experiment = OfflineExperiment(project_name='deepsmlm',
    #                                workspace='haydnspass',
    #                                auto_metric_logging=False,
    #                                offline_directory=deepsmlm_root + 'log/')
    experiment.log_parameters(hy_par._asdict(), prefix='Hyp')
    experiment.log_parameters(sim_par._asdict(), prefix='Sim')
    experiment.log_parameters(sched_par._asdict(), prefix='Sched')
    experiment.log_parameters(io_par._asdict(), prefix='IO')
    experiment.log_parameters(log_par._asdict(), prefix='Log')

    logger = SummaryWriter(log_dir, comment=io_par.log_comment, write_to_disk=WRITE_TO_LOG)
    logger.add_text('comet_ml_key', experiment.get_key())

    z_delta = DeltaPSF(xextent=sim_par.psf_extent[0],
                       yextent=sim_par.psf_extent[1],
                       zextent=None,
                       img_shape=(sim_par.img_size[0] * 8,
                                  sim_par.img_size[1] * 8),
                       photon_threshold=0,
                       photon_normalise=True,
                       dark_value=0.)
    target_generator = ZasOneHot(z_delta)

    smap_psf = SMAPSplineCoefficient(sim_par.calibration)
    psf = smap_psf.init_spline(sim_par.psf_extent[0], sim_par.psf_extent[1], sim_par.img_size)
    noise = Poisson(bg_uniform=sim_par.bg_pois)

    # structure_prior = DiscreteZStructure(torch.tensor([16., 16.]), 500., 10.)
    structure_prior = RandomStructure(sim_par.emitter_extent[0],
                                      sim_par.emitter_extent[1],
                                      sim_par.emitter_extent[2])

    prior = EmitterPopper(structure_prior,
                          density=sim_par.density,
                          photon_range=sim_par.photon_range,
                          emitter_av=sim_par.emitter_av)

    frame_range = (0, 0)
    simulator = Simulation(None,
                           sim_par.emitter_extent,
                           psf,
                           noise,
                           poolsize=0,
                           frame_range=frame_range)


    xy_helper_and_mask = DeltaPSF(xextent=sim_par.psf_extent[0],
                                  yextent=sim_par.psf_extent[1],
                                  zextent=None,
                                  img_shape=(sim_par.img_size[0] * 8,
                                             sim_par.img_size[1] * 8),
                                  photon_threshold=0,
                                  photon_normalise=True,
                                  dark_value=1 / 20)
    input_preparation = EasyZ(xy_helper_and_mask)

    train_size = sim_par.pseudo_data_size - hy_par.test_size

    train_data_smlm = SMLMDatasetOnFly(None, prior, simulator, train_size, input_preparation, target_generator, None,
                                       static=False, lifetime=hy_par.data_lifetime)

    test_data_smlm = SMLMDatasetOnFly(None, prior, simulator, hy_par.test_size, input_preparation, target_generator,
                                      None, static=True)

    train_loader = DataLoader(train_data_smlm,
                              batch_size=hy_par.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              collate_fn=smlm_collate)

    test_loader = DataLoader(test_data_smlm,
                             batch_size=hy_par.batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=False,
                             collate_fn=smlm_collate)

    model = USMLM(2, 1, 'nearest')
    model_ls = LoadSaveModel(model, output_file=io_par.model_out, input_file=io_par.model_init)
    model = model_ls.load_init()
    model = model.to(hy_par.device)

    optimiser = Adam(model.parameters(), lr=hy_par.lr)

    criterion = MaskedOnlyZLoss().return_criterion()

    lr_scheduler = ReduceLROnPlateau(optimiser,
                                     mode='min',
                                     factor=sched_par.lr_factor,
                                     patience=sched_par.lr_patience,
                                     threshold=sched_par.lr_threshold,
                                     cooldown=sched_par.lr_cooldown,
                                     verbose=sched_par.lr_verbose)

    last_new_model_name_time = time.time()

    for i in range(hy_par.num_epochs):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)
        experiment.log_metric('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        train(train_loader, model, optimiser, criterion, i, hy_par, logger, experiment, train_data_smlm.calc_new_flag)
        test(test_loader, model, criterion, i, hy_par, logger, experiment, None)

        """Give the output file a new suffix every hour (i.e. _0, _1, _2 ...)"""
        if time.time() > last_new_model_name_time + 60 * 60:
            trigger_new_name = True
            last_new_model_name_time = time.time()
        else:
            trigger_new_name = False

        model_ls.save(model, trigger_new_name)

