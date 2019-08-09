from comet_ml import Experiment, OfflineExperiment

import datetime
import time
import os
import getopt
import sys

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import deepsmlm.neuralfitter.post_processing as post
import deepsmlm.generic.inout.write_load_param as wlp
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.inout.load_save_emitter import NumpyInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
import deepsmlm.generic.background as background
import deepsmlm.generic.noise as noise_bg
import deepsmlm.generic.psf_kernel as psf_kernel
import deepsmlm.evaluation.evaluation as evaluation
from deepsmlm.generic.phot_camera import Photon2Camera
from deepsmlm.neuralfitter.pre_processing import OffsetRep, GlobalOffsetRep, ROIOffsetRep
import deepsmlm.generic.utils.logging as log_utils
from deepsmlm.generic.utils.data_utils import smlm_collate
import deepsmlm.generic.utils.processing as processing
from deepsmlm.generic.utils.scheduler import ScheduleSimulation
from deepsmlm.neuralfitter.arguments import InOutParameter, HyperParameter, SimulationParam, LoggerParameter, \
    SchedulerParameter, ScalingParam, EvaluationParam, PostProcessingParam, CameraParam
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.neuralfitter.dataset import SMLMDatasetOnFly
from deepsmlm.neuralfitter.losscollection import MultiScaleLaplaceLoss, BumpMSELoss, SpeiserLoss, OffsetROILoss
from deepsmlm.neuralfitter.models.model import DenseLoco, USMLM, USMLMLoco, UNet
from deepsmlm.neuralfitter.models.model_offset import OffsetUnet
from deepsmlm.neuralfitter.pre_processing import N2C, SingleEmitterOnlyZ
from deepsmlm.neuralfitter.scale_transform import InverseOffsetRescale, OffsetRescale
from deepsmlm.neuralfitter.train_test import train, test
from deepsmlm.generic.inout.util import add_root_relative

from deepsmlm.simulation import structure_prior, emittergenerator, simulator

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'

WRITE_TO_LOG = True

if __name__ == '__main__':

    """
    Parsing parameters from command line. 
    Valid options are p for specifying the parameter file and n for not logging the stuff.
    """
    param_file = None
    unixOptions = "p:n"
    gnuOptions = ["params=", "no_log"]

    # read commandline arguments, first
    argumentList = sys.argv[1:]

    try:
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-n", "--no_log"):
            WRITE_TO_LOG = False
            print("Not logging the current run.")

        elif currentArgument in ("-p", "--params"):
            print("Parameter file is in: {}".format(currentValue))
            param_file = currentValue

    param_file = add_root_relative(param_file, deepsmlm_root)

    """Load Parameters"""
    if param_file is None:
        raise ValueError("Parameters not specified. "
                         "Parse the parameter file via -p [Your parameeter.json]")
    param = wlp.load_params(param_file)

    """If path is relative add deepsmlm root."""
    param['InOut']['model_out'] = add_root_relative(param['InOut']['model_out'],
                                                    deepsmlm_root)
    param['InOut']['model_init'] = add_root_relative(param['InOut']['model_init'],
                                                     deepsmlm_root)
    param['Simulation']['calibration'] = add_root_relative(param['Simulation']['calibration'],
                                                           deepsmlm_root)

    # write params to folder where the network weights are
    param_file = param['InOut']['model_out'][:-3] + '_param.json'
    wlp.write_params(param_file, param)

    """Log System"""
    log_dir = deepsmlm_root + 'log/' + str(datetime.datetime.now())[:16]

    experiment = Experiment(project_name='deepsmlm', workspace='haydnspass',
                            auto_metric_logging=False, disabled=(not WRITE_TO_LOG))

    experiment.log_parameters(param['Hyper'], prefix='Hyp')
    experiment.log_parameters(param['Simulation'], prefix='Sim')
    experiment.log_parameters(param['Scheduler'], prefix='Sched')
    experiment.log_parameters(param['InOut'], prefix='IO')
    experiment.log_parameters(param['Logging'], prefix='Log')
    experiment.log_parameters(param['Scaling'], prefix='Scale')
    experiment.log_parameters(param['Camera'], prefix='Cam')
    experiment.log_parameters(param['PostProcessing'], prefix='Post')
    experiment.log_parameters(param['Evaluation'], prefix='Eval')

    """Add some tags as specified above."""
    for tag in param['Logging']['tags']:
        experiment.add_tag(tag)

    logger = SummaryWriter(log_dir,
                           comment=param['Logging']['log_comment'],
                           write_to_disk=WRITE_TO_LOG)

    logger.add_text('comet_ml_key', experiment.get_key())

    """Set target for the Neural Network."""
    # offsetRep = OffsetRep(xextent=param['Simulation']['psf_extent'][0],
    #                       yextent=param['Simulation']['psf_extent'][1],
    #                       zextent=None,
    #                       img_shape=(param['Simulation']['img_size'][0], param['Simulation']['img_size'][1]))
    offsetRep = ROIOffsetRep(xextent=param['Simulation']['psf_extent'][0],
                          yextent=param['Simulation']['psf_extent'][1],
                          zextent=None,
                          img_shape=(param['Simulation']['img_size'][0], param['Simulation']['img_size'][1]),
                             roi_size=3)

    tar_seq = []
    tar_seq.append(offsetRep)
    tar_seq.append(InverseOffsetRescale(param['Scaling']['dx_max'],
                                        param['Scaling']['dy_max'],
                                        param['Scaling']['z_max'],
                                        param['Scaling']['phot_max'],
                                        param['Scaling']['linearisation_buffer']))

    target_generator = processing.TransformSequence(tar_seq)

    if param['InOut']['data_mode'] == 'precomputed':
        """Load Data from binary."""
        emitter, extent, frames = NumpyInterface().load_binary(param['InOut']['data_set'])

        data_smlm = SMLMDataset(emitter, extent, frames, target_generator,
                                multi_frame_output=False,
                                dimensionality=None)

        train_size = data_smlm.__len__() - param['Hyper']['test_size']
        train_data_smlm, test_data_smlm = torch.utils.data.\
            random_split(data_smlm, [train_size, param['Hyper']['test_size']])

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=param['Hyper']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=param['Hyper']['test_size'], shuffle=False, num_workers=0, pin_memory=True)

    elif param['InOut']['data_mode'] == 'online':
        """Load 'Dataset' which is generated on the fly."""

        smap_psf = SMAPSplineCoefficient(param['Simulation']['calibration'])
        psf = smap_psf.init_spline(param['Simulation']['psf_extent'][0],
                                   param['Simulation']['psf_extent'][1],
                                   param['Simulation']['img_size'])

        """Define our noise model."""
        # Out of focus emitters, homogeneous background noise, poisson noise
        # noise = []
        # noise.append(background.OutOfFocusEmitters(
        #     sim_par.psf_extent[0],
        #     sim_par.psf_extent[1],
        #     sim_par.img_size,
        #     bg_range=(15., 15.),
        #     num_bg_emitter=3))

        # noise.append(noise_bg.Poisson(bg_uniform=sim_par.bg_pois))
        # noise = processing.TransformSequence(noise)
        noise = Photon2Camera(qe=param['Camera']['qe'],
                              spur_noise=param['Camera']['spur_noise'],
                              bg_uniform=param['Simulation']['bg_pois'],
                              em_gain=param['Camera']['em_gain'],
                              e_per_adu=param['Camera']['e_per_adu'],
                              baseline=param['Camera']['baseline'],
                              read_sigma=param['Camera']['read_sigma'])

        structure_prior = structure_prior.RandomStructure(param['Simulation']['emitter_extent'][0],
                                                          param['Simulation']['emitter_extent'][1],
                                                          param['Simulation']['emitter_extent'][2])

        prior = emittergenerator.EmitterPopperMultiFrame(structure_prior,
                                                         density=param['Simulation']['density'],
                                                         intensity_mu_sig=param['Simulation']['intensity_mu_sig'],
                                                         lifetime=param['Simulation']['lifetime_avg'],
                                                         num_frames=3,
                                                         emitter_av=param['Simulation']['emitter_av'])

        if param['Hyper']['channels'] == 3:
            frame_range = (-1, 1)
        elif param['Hyper']['channels'] == 1:
            frame_range = (0, 0)
        else:
            raise ValueError("Channels must be 1 (for only target frame) or 3 for one adjacent frame.")

        simulator = simulator.Simulation(None,
                                         param['Simulation']['emitter_extent'],
                                         psf,
                                         noise,
                                         poolsize=0,
                                         frame_range=frame_range)

        input_preparation = N2C()

        train_size = param['Simulation']['pseudo_data_size'] - param['Hyper']['test_size']

        train_data_smlm = SMLMDatasetOnFly(None, prior, simulator, train_size, input_preparation, target_generator,
                                           None, static=False, lifetime=param['Hyper']['data_lifetime'], return_em_tar=False)

        test_data_smlm = SMLMDatasetOnFly(None, prior, simulator, param['Hyper']['test_size'], input_preparation, target_generator,
                                          None, static=True, return_em_tar=True)

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=param['Hyper']['batch_size'],
                                  shuffle=True,
                                  num_workers=12,
                                  pin_memory=False,
                                  collate_fn=smlm_collate)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=param['Hyper']['batch_size'],
                                 shuffle=False,
                                 num_workers=6,
                                 pin_memory=False,
                                 collate_fn=smlm_collate)

    else:
        raise NameError("You used the wrong switch of how to get the training data.")

    """Set model and corresponding post-processing"""
    model = OffsetUnet(param['Hyper']['channels'])

    """Set up post processor"""
    post_processor = processing.TransformSequence([
        OffsetRescale(param['Scaling']['dx_max'],
                      param['Scaling']['dy_max'],
                      param['Scaling']['z_max'],
                      param['Scaling']['phot_max'],
                      param['Scaling']['linearisation_buffer']),
        post.Offset2Coordinate(param['Simulation']['psf_extent'][0],
                               param['Simulation']['psf_extent'][1],
                               param['Simulation']['img_size']),
        post.SpeiserPost(param['PostProcessing']['single_val_th'],
                         param['PostProcessing']['total_th'],
                         'emitters')
    ])

    """Log the model"""
    try:
        dummy = torch.rand((2, param['Hyper']['channels'],
                            *param['Simulation']['img_size']), requires_grad=True)
        logger.add_graph(model, dummy, False)
    except:
        print("Your dummy input is wrong. Please update it.")

    model_ls = LoadSaveModel(model,
                             output_file=param['InOut']['model_out'],
                             input_file=param['InOut']['model_init'])

    model = model_ls.load_init()
    model = model.to(torch.device(param['Hyper']['device']))

    optimiser = Adam(model.parameters(), lr=param['Hyper']['lr'])

    """Loss function."""
    # criterion = SpeiserLoss(weight_sqrt_phot=param['Hyper']['speiser_weight_sqrt_phot'],
    #                         class_freq_weight=param['Hyper']['class_freq_weight'],
    #                         pch_weight=param['Hyper']['pch_weight'], logger=logger)

    criterion = OffsetROILoss(roi_size=3, weight_sqrt_phot=param['Hyper']['speiser_weight_sqrt_phot'],
                              class_freq_weight=param['Hyper']['class_freq_weight'],
                              ch_weight=torch.tensor(param['Hyper']['ch_weight']), logger=logger)

    """Learning Rate Scheduling"""
    lr_scheduler = ReduceLROnPlateau(optimiser,
                                     mode='min',
                                     factor=param['Scheduler']['lr_factor'],
                                     patience=param['Scheduler']['lr_patience'],
                                     threshold=param['Scheduler']['lr_threshold'],
                                     cooldown=param['Scheduler']['lr_cooldown'],
                                     verbose=param['Scheduler']['lr_verbose'])

    sim_scheduler = ScheduleSimulation(prior=prior,
                                       datasets=[train_data_smlm, test_data_smlm],
                                       optimiser=optimiser,
                                       threshold=param['Scheduler']['sim_threshold'],
                                       step_size=param['Scheduler']['sim_factor'],
                                       max_emitter=param['Scheduler']['sim_max_value'],
                                       patience=param['Scheduler']['sim_patience'],
                                       cooldown=param['Scheduler']['sim_cooldown'])

    last_new_model_name_time = time.time()

    """Evaluation Specification"""
    matcher = evaluation.NNMatching(param['Evaluation']['dist_lat'],
                                    param['Evaluation']['dist_ax'],
                                    param['Evaluation']['match_dims'])
    segmentation_eval = evaluation.SegmentationEvaluation(False)
    distance_eval = evaluation.DistanceEvaluation(False)

    batch_ev = evaluation.BatchEvaluation(matcher, segmentation_eval, distance_eval)
    epoch_logger = log_utils.LogTestEpoch(logger, experiment)

    """Ask if everything is correct before we start."""
    for i in range(param['Hyper']['num_epochs']):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)
        experiment.log_metric('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        train(train_loader, model, optimiser, criterion, i, param, logger, experiment, train_data_smlm.calc_new_flag)

        val_loss = test(test_loader, model, criterion, i, param, experiment, post_processor, batch_ev, epoch_logger)
        lr_scheduler.step(val_loss)
        sim_scheduler.step(val_loss)

        """When using online generated data, reduce lifetime."""
        if param['InOut']['data_mode'] == 'online':
            train_data_smlm.step()

        """Save."""
        model_ls.save(model, val_loss)

    experiment.end()
