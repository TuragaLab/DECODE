from comet_ml import Experiment, OfflineExperiment

import datetime
import time
import os

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import deepsmlm.neuralfitter.post_processing as post
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.inout.load_save_emitter import NumpyInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
import deepsmlm.generic.background as background
import deepsmlm.generic.noise as noise_bg
import deepsmlm.generic.psf_kernel as psf_kernel
import deepsmlm.evaluation.evaluation as evaluation
from deepsmlm.neuralfitter.pre_processing import OffsetRep
import deepsmlm.generic.utils.logging as log_utils
from deepsmlm.generic.utils.data_utils import smlm_collate
import deepsmlm.generic.utils.processing as processing
from deepsmlm.generic.utils.scheduler import ScheduleSimulation
from deepsmlm.neuralfitter.arguments import InOutParameter, HyperParamter, SimulationParam, LoggerParameter, \
    SchedulerParameter, ScalingParam, EvaluationParam, PostProcessingParam
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.neuralfitter.dataset import SMLMDatasetOnFly
from deepsmlm.neuralfitter.losscollection import MultiScaleLaplaceLoss, BumpMSELoss, SpeiserLoss
from deepsmlm.neuralfitter.models.model import DenseLoco, USMLM, USMLMLoco, UNet
from deepsmlm.neuralfitter.models.model_offset import OffsetUnet
from deepsmlm.neuralfitter.pre_processing import N2C, SingleEmitterOnlyZ
from deepsmlm.neuralfitter.scale_transform import InverseOffsetRescale, OffsetRescale
from deepsmlm.neuralfitter.train_test import train, test

from deepsmlm.simulation import structure_prior, emittergenerator, simulator

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'

WRITE_TO_LOG = True

if __name__ == '__main__':

    """Set ur basic parameters"""
    io_par = InOutParameter(
        root=deepsmlm_root,
        log_comment='',
        data_mode='online',
        data_set=None,  # deepsmlm_root + 'data/2019-03-26/complete_z_range.npz',
        model_out=deepsmlm_root + 'network/2019-06-17_debug/model_re.pt',
        model_init=deepsmlm_root + 'network/2019-0')

    log_par = LoggerParameter(
        tags=['3D', 'Offset', 'UNet'])

    sched_par = SchedulerParameter(
        lr_factor=0.1,
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
        max_emitters=64,
        min_phot=0.,
        data_lifetime=10,
        upscaling=1,
        upscaling_mode='nearest',
        batch_size=2,
        test_size=4,
        num_epochs=10000,
        lr=1E-4,
        device=torch.device('cuda'))

    sim_par = SimulationParam(
        pseudo_data_size=(128 + 4),  # (256*256 + 512),
        emitter_extent=((-0.5, 255.5), (-0.5, 255.5), (-750, 750)),
        psf_extent=((-0.5, 255.5), (-0.5, 255.5), (-750., 750.)),
        img_size=(256, 256),
        density=0,
        emitter_av=100,
        photon_range=(1000, 20000),
        bg_pois=90,
        calibration=deepsmlm_root +
                    'data/calibration/2019-06-13_Calibration/sequence-as-stack-Beads-AS-Exp_3dcal.mat')

    scale_par = ScalingParam(
        dx_max=0.6,
        dy_max=0.6,
        z_max=750.,
        phot_max=25000.,
        linearisation_buffer=1.2
    )

    post_par = PostProcessingParam(
        single_val_th=0.3,
        total_th=0.6
    )

    eval_par = EvaluationParam(
        dist_lat=1.5,
        dist_ax=300,
        match_dims=2
    )

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
    experiment.log_parameters(scale_par._asdict(), prefix='Scale')

    """Add some tags as specified above."""
    for tag in log_par.tags:
        experiment.add_tag(tag)

    logger = SummaryWriter(log_dir, comment=io_par.log_comment, write_to_disk=WRITE_TO_LOG)
    logger.add_text('comet_ml_key', experiment.get_key())

    """Set target for the Neural Network."""
    offsetRep = OffsetRep(xextent=sim_par.psf_extent[0],
                          yextent=sim_par.psf_extent[1],
                          zextent=None,
                          img_shape=(sim_par.img_size[0], sim_par.img_size[1]))
    tar_seq = []
    tar_seq.append(offsetRep)
    tar_seq.append(InverseOffsetRescale(scale_par.dx_max,
                                        scale_par.dy_max,
                                        scale_par.z_max,
                                        scale_par.phot_max,
                                        scale_par.linearisation_buffer))

    target_generator = processing.TransformSequence(tar_seq)

    if io_par.data_mode == 'precomputed':
        """Load Data from binary."""
        emitter, extent, frames = NumpyInterface().load_binary(io_par.data_set)

        data_smlm = SMLMDataset(emitter, extent, frames, target_generator,
                                multi_frame_output=False,
                                dimensionality=None)

        train_size = data_smlm.__len__() - hy_par.test_size
        train_data_smlm, test_data_smlm = torch.utils.data.random_split(data_smlm, [train_size, hy_par.test_size])

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=hy_par.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=hy_par.test_size, shuffle=False, num_workers=0, pin_memory=True)

    elif io_par.data_mode == 'online':
        """Load 'Dataset' which is generated on the fly."""
        smap_psf = SMAPSplineCoefficient(sim_par.calibration)
        psf = smap_psf.init_spline(sim_par.psf_extent[0], sim_par.psf_extent[1], sim_par.img_size)

        """Define our noise model."""
        # Out of focus emitters, homogeneous background noise, poisson noise
        noise = []
        # noise.append(background.OutOfFocusEmitters(
        #     sim_par.psf_extent[0],
        #     sim_par.psf_extent[1],
        #     sim_par.img_size,
        #     bg_range=(15., 15.),
        #     num_bg_emitter=3))

        noise.append(noise_bg.Poisson(bg_uniform=sim_par.bg_pois))
        noise = processing.TransformSequence(noise)

        structure_prior = structure_prior.RandomStructure(sim_par.emitter_extent[0],
                                                          sim_par.emitter_extent[1],
                                                          sim_par.emitter_extent[2])

        prior = emittergenerator.EmitterPopperMultiFrame(structure_prior,
                                                         density=sim_par.density,
                                                         photon_range=sim_par.photon_range,
                                                         lifetime=1,
                                                         num_frames=3,
                                                         emitter_av=sim_par.emitter_av)

        if hy_par.channels == 3:
            frame_range = (-1, 1)
        elif hy_par.channels == 1:
            frame_range = (0, 0)
        else:
            raise ValueError("Channels must be 1 (for only target frame) or 3 for one adjacent frame.")

        simulator = simulator.Simulation(None,
                                                    sim_par.emitter_extent,
                                                    psf,
                                                    noise,
                                                    poolsize=0,
                                                    frame_range=frame_range)

        input_preparation = N2C()

        train_size = sim_par.pseudo_data_size - hy_par.test_size

        train_data_smlm = SMLMDatasetOnFly(None, prior, simulator, train_size, input_preparation, target_generator,
                                           None, static=False, lifetime=hy_par.data_lifetime, return_em_tar=False)

        test_data_smlm = SMLMDatasetOnFly(None, prior, simulator, hy_par.test_size, input_preparation, target_generator,
                                          None, static=True, return_em_tar=True)

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=hy_par.batch_size,
                                  shuffle=True,
                                  num_workers=12,
                                  pin_memory=False,
                                  collate_fn=smlm_collate)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=hy_par.batch_size,
                                 shuffle=False,
                                 num_workers=6,
                                 pin_memory=False,
                                 collate_fn=smlm_collate)

    else:
        raise NameError("You used the wrong switch of how to get the training data.")

    """Set model and corresponding post-processing"""
    model = OffsetUnet(hy_par.channels)

    proc = processing.TransformSequence([
        OffsetRescale(scale_par.dx_max,
                      scale_par.dy_max,
                      scale_par.z_max,
                      scale_par.phot_max,
                      scale_par.linearisation_buffer),
        post.Offset2Coordinate(sim_par.psf_extent[0], sim_par.psf_extent[1], sim_par.img_size),
        post.SpeiserPost(0.3, 0.6, 'emitters')
    ])

    """Log the model"""
    try:
        dummy = torch.rand((2, hy_par.channels, *sim_par.img_size), requires_grad=True)
        logger.add_graph(model, dummy, False)
    except:
        print("Your dummy input is wrong. Please update it.")

    model_ls = LoadSaveModel(model, output_file=io_par.model_out, input_file=io_par.model_init)
    model = model_ls.load_init()
    model = model.to(hy_par.device)

    optimiser = Adam(model.parameters(), lr=hy_par.lr)

    """Loss function."""
    criterion = SpeiserLoss().return_criterion()

    """Set up post processor"""
    post_processor = processing.TransformSequence([
        OffsetRescale(1, 1, 750., 25000, 1.2),
        post.Offset2Coordinate(sim_par.psf_extent[0], sim_par.psf_extent[1], sim_par.img_size),
        post.SpeiserPost(post_par.single_val_th, post_par.total_th, 'emitters')
    ])

    """Learning Rate Scheduling"""
    lr_scheduler = ReduceLROnPlateau(optimiser,
                                     mode='min',
                                     factor=sched_par.lr_factor,
                                     patience=sched_par.lr_patience,
                                     threshold=sched_par.lr_threshold,
                                     cooldown=sched_par.lr_cooldown,
                                     verbose=sched_par.lr_verbose)

    sim_scheduler = ScheduleSimulation(prior=prior,
                                       datasets=[train_data_smlm, test_data_smlm],
                                       optimiser=optimiser,
                                       threshold=sched_par.sim_threshold,
                                       step_size=sched_par.sim_factor,
                                       max_emitter=sched_par.sim_max_value,
                                       patience=sched_par.sim_patience,
                                       cooldown=sched_par.sim_cooldown)

    last_new_model_name_time = time.time()

    """Evaluation Specification"""
    matcher = evaluation.NNMatching(eval_par.dist_lat, eval_par.dist_ax, eval_par.match_dims)
    segmentation_eval = evaluation.SegmentationEvaluation(False)
    distance_eval = evaluation.DistanceEvaluation(False)

    batch_ev = evaluation.BatchEvaluation(matcher, segmentation_eval, distance_eval)
    epoch_logger = log_utils.LogTestEpoch(logger, experiment)

    """Ask if everything is correct before we start."""
    for i in range(hy_par.num_epochs):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)
        experiment.log_metric('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        train(train_loader, model, optimiser, criterion, i, hy_par, logger, experiment, train_data_smlm.calc_new_flag)

        val_loss = test(test_loader, model, criterion, i, hy_par, logger, experiment, post_processor, batch_ev, epoch_logger)
        lr_scheduler.step(val_loss)
        sim_scheduler.step(val_loss)

        """When using online generated data, reduce lifetime."""
        if io_par.data_mode == 'online':
            train_data_smlm.step()

        """Save."""
        model_ls.save(model, val_loss)

    experiment.end()
