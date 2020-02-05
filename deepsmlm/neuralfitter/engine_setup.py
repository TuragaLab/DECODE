import comet_ml

import click
import datetime
import os

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import deepsmlm
import deepsmlm.generic.psf_kernel
import deepsmlm.generic.utils

import deepsmlm.generic.background
import deepsmlm.generic.phot_camera
import deepsmlm.generic.inout.write_load_param as dsmlm_par
import deepsmlm.generic.inout.util
import deepsmlm.simulation.engine
import deepsmlm.generic.inout.load_calibration
import deepsmlm.neuralfitter.dataset
import deepsmlm.neuralfitter.scale_transform


"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'

WRITE_TO_LOG = True


@click.command()
@click.option('--no_log', '-n', default=False, is_flag=True,
              help='Set no log if you do not want to log the current run.')
@click.option('--param_file', '-p', required=True,
              help='Specify your parameter file (.yml or .json).')
@click.option('--debug_param', '-d', default=False, is_flag=True,
              help='Debug the specified parameter file. Will reduce ds size for example.')
@click.option('--log_folder', '-l', default='runs',
              help='Specify the folder you want to log to. If rel-path, relative to DeepSMLM root.')
@click.option('--num_worker_override', '-w', default=None, type=int,
              help='Override the number of workers for the dataloaders.')
def setup_train_engine(param_file, no_log, debug_param, log_folder, num_worker_override):
    """
    Sets up a training engine that loads data from the simulation engine.

    Args:
        param_file:
        no_log:
        debug_param:
        log_folder:
        num_worker_override:

    Returns:

    """

    def setup_logging():
        """
        Setup COMET_ML Logging system

        Returns:

        """
        experiment = comet_ml.Experiment(project_name='deepsmlm', workspace='haydnspass',
                                auto_metric_logging=False, disabled=(not WRITE_TO_LOG),
                                api_key="PaCYtLsZ40Apm5CNOHxBuuJvF")

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
        for tag in param.Logging.cometml_tags:
            experiment.add_tag(tag)

        return experiment

    """Load Parameters"""
    param_file = deepsmlm.generic.inout.util.add_root_relative(param_file, deepsmlm_root)
    if param_file is None:
        raise ValueError("Parameters not specified. "
                         "Parse the parameter file via -p [Your parameeter.json]")
    param = dsmlm_par.ParamHandling().load_params(param_file)

    if no_log:
        WRITE_TO_LOG = False
    else:
        WRITE_TO_LOG = True

    if debug_param:
        dsmlm_par.ParamHandling.convert_param_debug(param)

    if num_worker_override is not None:
        param.Hardware.num_worker_sim = num_worker_override

    """Server stuff."""
    assert torch.cuda.device_count() <= 1
    torch.set_num_threads(param.Hardware.torch_threads)

    """If path is relative add deepsmlm root."""
    param.InOut.model_out = deepsmlm.generic.inout.util.add_root_relative(param.InOut.model_out,
                                                    deepsmlm_root)
    param.InOut.model_init = deepsmlm.generic.inout.util.add_root_relative(param.InOut.model_init,
                                                     deepsmlm_root)

    """Backup copy of the modified parameters."""
    param_file_out = param.InOut.model_out[:-3] + '_param.json'
    dsmlm_par.ParamHandling().write_params(param_file_out, param)

    """Set model, optimiser, loss and schedulers"""
    models_ava = {
        'BGNet': model_zoo.BGNet,
        'DoubleMUnet': model_zoo.DoubleMUnet,
        'SimpleSMLMNet': model_zoo.SimpleSMLMNet,
        'SMLMNetBG': model_zoo.SMLMNetBG
    }
    model = models_ava[param.HyperParameter.architecture]
    model = model.parse(param)

    model_ls = LoadSaveModel(model,
                             output_file=param['InOut']['model_out'],
                             input_file=param['InOut']['model_init'])

    model = model_ls.load_init()
    model = model.to(torch.device(param['Hardware']['device']))

    # Small collection of optimisers
    opt_ava = {
        'Adam': Adam,
        'AdamW': AdamW
    }

    optimizer = opt_ava[param.HyperParameter.optimizer]
    optimizer = optimizer(model.parameters(), **param.HyperParameter.opt_param)

    """Loss function."""
    criterion = ls.MaskedPxyzLoss.parse(param, logger)

    """Learning Rate and Simulation Scheduling"""
    lr_scheduler = ReduceLROnPlateau(optimizer, **param.LearningRateScheduler)

    """Log the model"""
    try:
        dummy = torch.rand((2, param.HyperParameter.channels_in,
                            *param.Simulation.img_size), requires_grad=True)
        logger.add_graph(model, dummy, False)
    except:
        print("Your dummy input is wrong. Please update it.")

    # ToDo: Some logging

    """Transform input data, compute weight mask and target data"""
    in_prep = deepsmlm.neuralfitter.scale_transform.InputFrameRescale.parse(param)

    tar_gen = deepsmlm.generic.utils.processing.TransformSequence([
        ROIOffsetRep.parse(param),
        InverseOffsetRescale.parse(param)
    ])

    weight_gen = SimpleWeight.parse(param)

    train_ds = SMLMEngineDataset()
    test_ds = SMLMEngineDataset()

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameter.batch_size,
        shuffle=True,
        num_workers=param.Hardware.num_worker_train,
        pin_memory=False,
        collate_fn=smlm_collate)

    test_dl = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=param.HyperParameter.batch_size,
        shuffle=False,
        num_workers=param.Hardware.num_worker_train,
        pin_memory=False,
        collate_fn=smlm_collate)

    """Set up post processor"""
    if not param.HyperParameter.suppress_post_processing:
        post_processor = deepsmlm.generic.utils.processing.TransformSequence.parse([OffsetRescale,
                                                         post.Offset2Coordinate,
                                                         post.ConsistencyPostprocessing], param)
    else:
        post_processor = post.NoPostProcessing()

    """Evaluation Specification"""
    matcher = deepsmlm.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)
    segmentation_eval = evaluation.SegmentationEvaluation(False)
    distance_eval = evaluation.DistanceEvaluation(print_mode=False)

    batch_ev = evaluation.BatchEvaluation(matcher, segmentation_eval, distance_eval,
                                          batch_size=param.HyperParameter.batch_size,
                                          px_size=torch.tensor(param.Camera.px_size), weight='photons')

    epoch_logger = log_utils.LogTestEpoch(logger, experiment)

    # this is useful if we restart a training
    first_epoch = param['HyperParameter']['epoch_0'] if param['HyperParameter']['epoch_0'] is not None else 0
    for i in range(first_epoch, param.HyperParameter.epochs):
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)
        experiment.log_metric('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        _ = train(train_dl, model, optimizer, criterion, i, param, logger, experiment)

        val_loss = test(test_dl, model, criterion, i, param, logger, experiment, post_processor, batch_ev, epoch_logger)

        """
        When using online generated data and data is given a lifetime, 
        reduce the steps until a new dataset is to be created. This needs to happen before sim_scheduler (for reasons).
        """
        if param.InOut.data_set == 'online':
            train_loader.dataset.step()

        lr_scheduler.step(val_loss)
        sim_scheduler.step(val_loss)

        """Save."""
        model_ls.save(model, val_loss)

    experiment.end()


if __name__ == '__main__':
    setup_train_engine()