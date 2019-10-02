from comet_ml import Experiment, OfflineExperiment

import datetime
import time
import os
import getopt
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
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
from deepsmlm.neuralfitter.pre_processing import OffsetRep, GlobalOffsetRep, ROIOffsetRep, CombineTargetBackground, \
    DiscardBackground
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

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'

WRITE_TO_LOG = True

"""Load Parameters here, import torch afterwards"""

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

    """Some Server stuff"""
    if param['Hardware']['device'] == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = param['Hardware']['cuda_ix']
    else:
        print("Device is not CUDA. Cuda ix not set.")
    os.nice(param['Hardware']['unix_niceness'])

    assert torch.cuda.device_count() <= 1
    torch.set_num_threads(param['Hardware']['torch_threads'])

    """If path is relative add deepsmlm root."""
    param['InOut']['model_out'] = add_root_relative(param['InOut']['model_out'],
                                                    deepsmlm_root)
    param['InOut']['model_init'] = add_root_relative(param['InOut']['model_init'],
                                                     deepsmlm_root)
    param['InOut']['calibration_file'] = add_root_relative(param['InOut']['calibration_file'],
                                                           deepsmlm_root)

    # write params to folder where the network weights are
    param_file = param['InOut']['model_out'][:-3] + '_param.json'
    wlp.write_params(param_file, param)

    """Log System"""
    log_dir = deepsmlm_root + 'log/' + str(datetime.datetime.now())[:16]

    experiment = Experiment(project_name='deepsmlm', workspace='haydnspass',
                            auto_metric_logging=False, disabled=(not WRITE_TO_LOG))

    experiment.log_parameters(param['InOut'], prefix='IO')
    experiment.log_parameters(param['Hardware'], prefix='Hw')
    experiment.log_parameters(param['Logging'], prefix='Log')
    experiment.log_parameters(param['HyperParameter'], prefix='Hyp')
    experiment.log_parameters(param['LearningRateScheduler'], prefix='Sched')
    experiment.log_parameters(param['SimulationScheduler'], prefix='Sched')
    experiment.log_parameters(param['Simulation'], prefix='Sim')
    experiment.log_parameters(param['Scaling'], prefix='Scale')
    experiment.log_parameters(param['Camera'], prefix='Cam')
    experiment.log_parameters(param['PostProcessing'], prefix='Post')
    experiment.log_parameters(param['Evaluation'], prefix='Eval')

    """Add some tags as specified above."""
    for tag in param['Logging']['cometml_tags']:
        experiment.add_tag(tag)

    if not WRITE_TO_LOG:
        log_comment = 'debug_'
    else:
        log_comment = param['Logging']['log_comment']

    logger = SummaryWriter(comment=log_comment,
                           write_to_disk=WRITE_TO_LOG)

    logger.add_text('comet_ml_key', experiment.get_key())

    """Set target for the Neural Network."""
    if param['HyperParameter']['predict_bg']:
        target_generator = processing.TransformSequence([
            CombineTargetBackground(ROIOffsetRep.parse(param), num_input_frames=param['HyperParameter']['channels_in']),
            InverseOffsetRescale.parse(param)
        ])
    else:
        target_generator = processing.TransformSequence.parse([ROIOffsetRep, InverseOffsetRescale], param)

    if param['InOut']['data_set'] == 'precomputed':
        """Load Data from binary."""
        emitter, extent, frames = NumpyInterface().load_binary(param['InOut']['data_set'])

        data_smlm = SMLMDataset(emitter, extent, frames, target_generator,
                                multi_frame_output=False,
                                dimensionality=None)

        train_size = data_smlm.__len__() - param['Hyper']['test_size']
        train_data_smlm, test_data_smlm = torch.utils.data.\
            random_split(data_smlm, [train_size, param['Hyper']['test_size']])

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=param['Hyper']['batch_size'], shuffle=True,
                                  num_workers=param['Simulation']['num_workers'], pin_memory=True)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=param['Hyper']['test_size'], shuffle=False,
                                 num_workers=param['Simulation']['num_workers'], pin_memory=True)

    elif param['InOut']['data_set'] == 'online':
        """Load 'Dataset' which is generated on the fly."""

        smap_psf = SMAPSplineCoefficient(param['InOut']['calibration_file'])
        psf = smap_psf.init_spline(param['Simulation']['psf_extent'][0],
                                   param['Simulation']['psf_extent'][1],
                                   param['Simulation']['img_size'])

        """Define our noise model."""
        background = processing.TransformSequence.parse([background.UniformBackground,
                                                         background.PerlinBackground], param)
        noise = Photon2Camera.parse(param)

        structure_prior = structure_prior.RandomStructure(param['Simulation']['emitter_extent'][0],
                                                          param['Simulation']['emitter_extent'][1],
                                                          param['Simulation']['emitter_extent'][2])

        if param['HyperParameter']['channels_in'] == 1:
            frame_range = (0, 0)
        elif param['HyperParameter']['channels_in'] == 3:
            frame_range = (-1, 1)
        else:
            raise ValueError("Channels must be 1 (for only target frame) or 3 for one adjacent frame.")

        prior = emittergenerator.EmitterPopperMultiFrame(structure_prior,
                                                         density=param['Simulation']['density'],
                                                         intensity_mu_sig=param['Simulation']['intensity_mu_sig'],
                                                         lifetime=param['Simulation']['lifetime_avg'],
                                                         num_frames=3,
                                                         emitter_av=param['Simulation']['emitter_av'])

        simulator = simulator.Simulation(None, extent=param['Simulation']['emitter_extent'],
                                         psf=psf,
                                         background=background,
                                         noise=noise,
                                         frame_range=frame_range,
                                         poolsize=0,
                                         out_bg=param['HyperParameter']['predict_bg'])

        input_preparation = processing.TransformSequence([
            DiscardBackground(),
            N2C()])

        train_data_smlm = SMLMDatasetOnFly(None, prior, simulator, param['HyperParameter']['pseudo_ds_size'], input_preparation, target_generator,
                                           None, static=False, lifetime=param['HyperParameter']['ds_lifetime'], return_em_tar=False)

        test_data_smlm = SMLMDatasetOnFly(None, prior, simulator, param['HyperParameter']['test_size'], input_preparation, target_generator,
                                          None, static=True, return_em_tar=True)

        train_loader = DataLoader(train_data_smlm,
                                  batch_size=param['HyperParameter']['batch_size'],
                                  shuffle=True,
                                  num_workers=param['Hardware']['num_worker_sim'],
                                  pin_memory=False,
                                  collate_fn=smlm_collate)

        test_loader = DataLoader(test_data_smlm,
                                 batch_size=param['HyperParameter']['batch_size'],
                                 shuffle=False,
                                 num_workers=param['Hardware']['num_worker_sim'],
                                 pin_memory=False,
                                 collate_fn=smlm_collate)

    else:
        raise NameError("You used the wrong switch of how to get the training data.")

    """Set model and corresponding post-processing"""
    model = OffsetUnet(n_channels=param['HyperParameter']['channels_in'],
                       n_classes=param['HyperParameter']['channels_out'])

    """Set up post processor"""
    post_processor = processing.TransformSequence.parse([OffsetRescale,
                                                         post.Offset2Coordinate,
                                                         post.ConsistencyPostprocessing], param)

    """Log the model"""
    try:
        dummy = torch.rand((2, param['HyperParameter']['channels'],
                            *param['Simulation']['img_size']), requires_grad=True)
        logger.add_graph(model, dummy, False)
    except:
        print("Your dummy input is wrong. Please update it.")

    model_ls = LoadSaveModel(model,
                             output_file=param['InOut']['model_out'],
                             input_file=param['InOut']['model_init'])

    model = model_ls.load_init()
    model = model.to(torch.device(param['Hardware']['device']))

    optimiser = Adam(model.parameters(), lr=param['HyperParameter']['lr'])

    """Loss function."""
    criterion = OffsetROILoss.parse(param, logger=logger)

    """Learning Rate and Simulation Scheduling"""
    lr_scheduler = ReduceLROnPlateau(optimiser, **param['LearningRateScheduler'])
    sim_scheduler = ScheduleSimulation.parse(prior, [train_loader.dataset, test_loader.dataset], optimiser, param)

    last_new_model_name_time = time.time()

    """Evaluation Specification"""
    matcher = evaluation.NNMatching.parse(param)
    segmentation_eval = evaluation.SegmentationEvaluation(False)
    distance_eval = evaluation.DistanceEvaluation(print_mode=False)

    batch_ev = evaluation.BatchEvaluation(matcher, segmentation_eval, distance_eval,
                                          px_size=torch.tensor(param['Camera']['px_size']))
    epoch_logger = log_utils.LogTestEpoch(logger, experiment)

    """Ask if everything is correct before we start."""
    for i in range(param['HyperParameter']['epochs']):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)
        experiment.log_metric('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        _ = train(train_loader, model, optimiser, criterion, i, param, logger, experiment, train_data_smlm.calc_new_flag)

        val_loss = test(test_loader, model, criterion, i, param, logger, experiment, post_processor, batch_ev, epoch_logger)

        """
        When using online generated data and data is given a lifetime, 
        reduce the steps until a new dataset is to be created. This needs to happen before sim_scheduler (for reasons).
        """
        if param['InOut']['data_set'] == 'online':
            train_loader.dataset.step()

        lr_scheduler.step(val_loss)
        sim_scheduler.step(val_loss)

        """Save."""
        model_ls.save(model, val_loss)

    experiment.end()
