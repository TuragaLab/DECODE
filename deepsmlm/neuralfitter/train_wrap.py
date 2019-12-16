import comet_ml
from comet_ml import Experiment, OfflineExperiment

import click
import datetime
import time
import os
import getopt
import sys
import torch
import tqdm
import pathlib
import socket
from datetime import datetime

import deepsmlm.evaluation.match_emittersets

torch.multiprocessing.set_sharing_strategy('file_system')
from tensorboardX import SummaryWriter
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import deepsmlm.neuralfitter.post_processing as post
import deepsmlm.generic.inout.write_load_param as wlp
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.inout.load_save_emitter import NumpyInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
import deepsmlm.generic.background as background
import deepsmlm.neuralfitter.weight_generator as wgen
import deepsmlm.generic.noise as noise_bg
import deepsmlm.generic.psf_kernel as psf_kernel
import deepsmlm.neuralfitter.pre_processing as prepro
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
from deepsmlm.neuralfitter.dataset import SMLMStaticDataset, SMLMDatasetOnFly, SMLMDatasetOneTimer, SMLMDatasetOnFlyCached
from deepsmlm.neuralfitter.losscollection import MultiScaleLaplaceLoss, BumpMSELoss, SpeiserLoss, OffsetROILoss
import deepsmlm.neuralfitter.losscollection as ls
from deepsmlm.neuralfitter.models.model import DenseLoco, USMLM, USMLMLoco, UNet
from deepsmlm.neuralfitter.models.model_offset import OffsetUnet, DoubleOffsetUNet, DoubleOffsetUNetDivided, \
    OffSetUNetBGBranch
import deepsmlm.neuralfitter.models.model_param as model_zoo
from deepsmlm.neuralfitter.pre_processing import N2C, SingleEmitterOnlyZ
from deepsmlm.neuralfitter.scale_transform import InverseOffsetRescale, OffsetRescale, InputFrameRescale
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

@click.command()
@click.option('--no_log', '-n', default=False, is_flag=True, help='Set no log if you do not want to log the current run.')
@click.option('--param_file', '-p', required=True, help='Specify your parameter file (.yml or .json).')
@click.option('--debug_param', '-d', default=False, is_flag=True, help='Debug the specified parameter file. Will reduce ds size for example.')
@click.option('--log_folder', '-l', default='runs', help='Specify the folder you want to log to. If rel-path, relative to DeepSMLM root.')
def train_wrap(param_file, no_log, debug_param, log_folder):

    if no_log:
        WRITE_TO_LOG = False
    else:
        WRITE_TO_LOG = True

    """
    From SummaryWriter. Mimics the usual subfolder stuff.
    """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_folder = os.path.join(log_folder, current_time + '_' + socket.gethostname())
    if not pathlib.Path(log_folder).is_absolute():
        log_folder = deepsmlm_root + log_folder

    """Load Parameters"""
    param_file = add_root_relative(param_file, deepsmlm_root)
    if param_file is None:
        raise ValueError("Parameters not specified. "
                         "Parse the parameter file via -p [Your parameeter.json]")
    param = wlp.ParamHandling().load_params(param_file)

    if debug_param:
        wlp.ParamHandling.convert_param_debug(param)

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
    param_file_out = param['InOut']['model_out'][:-3] + '_param.json'
    wlp.ParamHandling().write_params(param_file_out, param)

    """Log System"""
    experiment = Experiment(project_name='deepsmlm', workspace='haydnspass',
                            auto_metric_logging=False, disabled=(not WRITE_TO_LOG), api_key="PaCYtLsZ40Apm5CNOHxBuuJvF")

    experiment.log_asset(param_file, file_name='config_in')
    experiment.log_asset(param_file_out, file_name='config_out')

    param_comet = param.toDict()
    experiment.log_parameters(param_comet['InOut'], prefix='IO')
    experiment.log_parameters(param_comet['Hardware'], prefix='Hw')
    experiment.log_parameters(param_comet['Logging'], prefix='Log')
    experiment.log_parameters(param_comet['HyperParameter'], prefix='Hyp')
    experiment.log_parameters(param_comet['LearningRateScheduler'], prefix='Sched')
    experiment.log_parameters(param_comet['SimulationScheduler'], prefix='Sched')
    experiment.log_parameters(param_comet['Simulation'], prefix='Sim')
    experiment.log_parameters(param_comet['Scaling'], prefix='Scale')
    experiment.log_parameters(param_comet['Camera'], prefix='Cam')
    experiment.log_parameters(param_comet['PostProcessing'], prefix='Post')
    experiment.log_parameters(param_comet['Evaluation'], prefix='Eval')

    """Add some tags as specified above."""
    for tag in param['Logging']['cometml_tags']:
        experiment.add_tag(tag)

    if not WRITE_TO_LOG:
        log_comment = 'debug_'
    else:
        assert log_folder is not None
        log_comment = param['Logging']['log_comment']
        log_folder = log_folder + log_comment

    logger = SummaryWriter(write_to_disk=WRITE_TO_LOG, log_dir=log_folder)

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

        data_smlm = SMLMStaticDataset(emitter, extent, frames, target_generator, multi_frame_output=False)

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
        if param.Simulation.bg_perlin_amplitude is None:
            bg = background.UniformBackground.parse(param)
        else:
            bg = processing.TransformSequence.parse([background.UniformBackground,
                                                     background.PerlinBackground], param)

        noise = Photon2Camera.parse(param)

        prior_struct = structure_prior.RandomStructure(param['Simulation']['emitter_extent'][0],
                                                          param['Simulation']['emitter_extent'][1],
                                                          param['Simulation']['emitter_extent'][2])

        if param['HyperParameter']['channels_in'] == 1:
            frame_range = (0, 0)
        elif param['HyperParameter']['channels_in'] == 3:
            frame_range = (-1, 1)
        else:
            raise ValueError("Channels must be 1 (for only target frame) or 3 for one adjacent frame.")

        prior = emittergenerator.EmitterPopperMultiFrame(prior_struct,
                                                         density=param['Simulation']['density'],
                                                         intensity_mu_sig=param['Simulation']['intensity_mu_sig'],
                                                         lifetime=param['Simulation']['lifetime_avg'],
                                                         num_frames=3,
                                                         emitter_av=param['Simulation']['emitter_av'])

        sim = simulator.Simulation(None, extent=param['Simulation']['emitter_extent'],
                                         psf=psf,
                                         background=bg,
                                         noise=noise,
                                         frame_range=frame_range,
                                         poolsize=0,
                                         out_bg=param['HyperParameter']['predict_bg'])

        input_preparation = processing.TransformSequence([
            DiscardBackground(),
            N2C(),
            InputFrameRescale.parse(param)
        ])

        if param.HyperParameter.weight_base == 'crlb':
            weight_mask_generator = processing.TransformSequence([
                wgen.DerivePseudobgFromBg(param['Simulation']['psf_extent'][0],
                                          param['Simulation']['psf_extent'][1],
                                          param['Simulation']['img_size'], psf.roi_size),
                wgen.CalcCRLB(psf),
                wgen.GenerateWeightMaskFromCRLB(param['Simulation']['psf_extent'][0],
                                          param['Simulation']['psf_extent'][1],
                                          param['Simulation']['img_size'], param['HyperParameter']['target_roi_size'])
            ])
        else:
            weight_mask_generator = wgen.SimpleWeight.parse(param)

        if param.HyperParameter.ds_lifetime >= 2:
            train_data_smlm = SMLMDatasetOnFlyCached(extent=None,
                                                     prior=prior,
                                                     simulator=sim,
                                                     ds_size=param['HyperParameter']['pseudo_ds_size'],
                                                     in_prep=input_preparation,
                                                     tar_gen=target_generator,
                                                     w_gen=weight_mask_generator,
                                                     lifetime=param['HyperParameter']['ds_lifetime'],
                                                     return_em_tar=False,
                                                     predict_bg=param['HyperParameter']['predict_bg'])

        else:
            train_data_smlm = SMLMDatasetOnFly(extent=None, prior=prior, simulator=sim,
                                               ds_size=param['HyperParameter']['pseudo_ds_size'], in_prep=input_preparation,
                                               tar_gen=target_generator, w_gen=weight_mask_generator, return_em_tar=False,
                                               predict_bg=param['HyperParameter']['predict_bg'])

        test_data_smlm = SMLMDatasetOneTimer(None, prior, sim, param['HyperParameter']['test_size'],
                                             input_preparation, target_generator, weight_mask_generator,
                                             return_em_tar=True, predict_bg=param['HyperParameter']['predict_bg'])

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
    models_ava = {
        'BGNet': model_zoo.BGNet,
        'DoubleMUnet': model_zoo.DoubleMUnet,
        'SimpleSMLMNet': model_zoo.SimpleSMLMNet,
        'SMLMNetBG': model_zoo.SMLMNetBG
    }
    model = models_ava[param.HyperParameter.architecture]
    model = model.parse(param)

    """Set up post processor"""
    post_processor = processing.TransformSequence.parse([OffsetRescale,
                                                         post.Offset2Coordinate,
                                                         post.ConsistencyPostprocessing], param)

    if param['HyperParameter']['suppress_post_processing']:
        post_processor = post.NoPostProcessing()

    """Log the model"""
    try:
        dummy = torch.rand((2, param['HyperParameter']['channels_in'],
                            *param['Simulation']['img_size']), requires_grad=True)
        logger.add_graph(model, dummy, False)
    except:
        print("Your dummy input is wrong. Please update it.")

    model_ls = LoadSaveModel(model,
                             output_file=param['InOut']['model_out'],
                             input_file=param['InOut']['model_init'])

    model = model_ls.load_init()
    model = model.to(torch.device(param['Hardware']['device']))

    optimiser = eval(param['HyperParameter']['optimizer'])
    optimiser = optimiser(model.parameters(), **param['HyperParameter']['opt_param'])

    """Loss function."""
    criterion = ls.MaskedPxyzLoss.parse(param, logger)

    """Learning Rate and Simulation Scheduling"""
    lr_scheduler = ReduceLROnPlateau(optimiser, **param['LearningRateScheduler'])
    sim_scheduler = ScheduleSimulation.parse(prior, [train_loader.dataset, test_loader.dataset], optimiser, param)

    """Evaluation Specification"""
    matcher = deepsmlm.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)
    segmentation_eval = evaluation.SegmentationEvaluation(False)
    distance_eval = evaluation.DistanceEvaluation(print_mode=False)

    batch_ev = evaluation.BatchEvaluation(matcher, segmentation_eval, distance_eval,
                                          batch_size=param['HyperParameter']['batch_size'],
                                          px_size=torch.tensor(param['Camera']['px_size']))
    epoch_logger = log_utils.LogTestEpoch(logger, experiment)

    """Ask if everything is correct before we start."""
    first_epoch = param['HyperParameter']['epoch_0'] if param['HyperParameter']['epoch_0'] is not None else 0
    for i in range(first_epoch, param['HyperParameter']['epochs']):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)
        experiment.log_metric('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        _ = train(train_loader, model, optimiser, criterion, i, param, logger, experiment)

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


if __name__ == '__main__':
    train_wrap()
