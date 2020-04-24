import os
import pathlib
import shutil
import socket
import datetime
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import click
from deepsmlm.neuralfitter.utils import log_train_val_progress

import deepsmlm
import deepsmlm.evaluation
import deepsmlm.evaluation.utils
import deepsmlm.generic.inout.load_calibration
import deepsmlm.generic.inout.load_save_model
import deepsmlm.generic.inout.util
import deepsmlm.generic.inout.write_load_param as dsmlm_par
import deepsmlm.generic.utils
import deepsmlm.generic.utils.logging
import deepsmlm.neuralfitter.engine
import deepsmlm.neuralfitter.filter
import deepsmlm.neuralfitter.utils
import deepsmlm.neuralfitter.models.model_param
import deepsmlm.neuralfitter.target_generator
import deepsmlm.neuralfitter.train_val_impl
import deepsmlm.neuralfitter.utils.pytorch_customs
import deepsmlm.simulation

"""Root folder"""
deepsmlm_root = pathlib.Path(deepsmlm.__file__).parent.parent  # 'repo' directory


@click.command()
@click.option('--param_file', '-p', required=True,
              help='Specify your parameter file (.yml or .json).')
@click.option('--exp_id', '-e', required=True,
              help='Specify the experiments id under which the engine stores the results.')
@click.option('--cache_dir', '-c', default=deepsmlm_root / pathlib.Path('cachedir/simulation_engine'),
              help='Overwrite the cache folder in which the simulation engine stores the results')
@click.option('--no_log', '-n', default=False, is_flag=True,
              help='Set no log if you do not want to log the current run.')
@click.option('--debug_param', '-d', default=False, is_flag=True,
              help='Debug the specified parameter file. Will reduce ds size for example.')
@click.option('--log_folder', '-l', default='runs',
              help='Specify the (parent) folder you want to log to. If rel-path, relative to DeepSMLM root.')
@click.option('--num_worker_override', '-w', default=None, type=int,
              help='Override the number of workers for the dataloaders.')
def setup_train_engine(param_file, exp_id, cache_dir, no_log, debug_param, log_folder, num_worker_override):
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

    """Load Parameters"""
    param_file = pathlib.Path(param_file)
    param = dsmlm_par.ParamHandling().load_params(param_file)

    if debug_param:
        dsmlm_par.ParamHandling.convert_param_debug(param)

    if num_worker_override is not None:
        param.Hardware.num_worker_sim = num_worker_override

    """Hardware / Server stuff."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param.Hardware.device_sim_ix)
    os.nice(param.Hardware.unix_niceness_train)  # set niceness of process

    import torch.utils.tensorboard
    assert torch.cuda.device_count() <= param.Hardware.max_cuda_devices
    torch.set_num_threads(param.Hardware.torch_threads_train)
    # torch.multiprocessing.set_sharing_strategy('file_system')  # does not seem to work with spawn method together

    """If path is relative add deepsmlm root."""
    param.InOut.model_out = deepsmlm.generic.inout.util.add_root_relative(param.InOut.model_out,
                                                                          deepsmlm_root)
    if param.InOut.model_init is not None:
        param.InOut.model_init = deepsmlm.generic.inout.util.add_root_relative(param.InOut.model_init,
                                                                               deepsmlm_root)

    """Backup copy of the parameters."""
    param_backup = param.InOut.model_out.with_suffix(param_file.suffix)
    shutil.copy(param_file, param_backup)

    """Setup Log System"""
    if no_log:
        logger = deepsmlm.neuralfitter.utils.logger.NoLog()
    else:
        log_folder = log_folder + '/' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '_' + socket.gethostname()
        logger = deepsmlm.neuralfitter.utils.logger.SummaryWriterSoph(log_dir=log_folder)

    """Setup the engines."""
    engine_train = deepsmlm.neuralfitter.engine.SMLMTrainingEngine(
        cache_dir=cache_dir,
        sim_id=exp_id
    )

    engine_test = deepsmlm.neuralfitter.engine.SMLMTrainingEngine(
        cache_dir=cache_dir,
        sim_id=exp_id,
        test_set=True
    )

    """Set model, optimiser, loss and schedulers"""
    models_ava = {
        'BGNet': deepsmlm.neuralfitter.models.model_param.BGNet,
        'DoubleMUnet': deepsmlm.neuralfitter.models.model_param.DoubleMUnet,
        'SimpleSMLMNet': deepsmlm.neuralfitter.models.model_param.SimpleSMLMNet,
        'SMLMNetBG': deepsmlm.neuralfitter.models.model_param.SMLMNetBG
    }
    model = models_ava[param.HyperParameter.architecture]
    model = model.parse(param)

    model_ls = deepsmlm.generic.inout.load_save_model.LoadSaveModel(model,
                                                                    output_file=param.InOut.model_out,
                                                                    input_file=param.InOut.model_init)

    model = model_ls.load_init()
    model = model.to(torch.device(param.Hardware.device_train))

    # Small collection of optimisers
    opt_ava = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }

    optimizer = opt_ava[param.HyperParameter.optimizer]
    optimizer = optimizer(model.parameters(), **param.HyperParameter.opt_param)

    """Loss function."""
    criterion = deepsmlm.neuralfitter.losscollection.PPXYZBLoss(chweight_stat=param.HyperParameter.ch_static_scale,
                                                                device=param.Hardware.device_train)

    """Learning Rate and Simulation Scheduling"""
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **param.LearningRateScheduler)

    """Log the model"""
    try:
        dummy = torch.rand((2, param.HyperParameter.channels_in,
                            *param.Simulation.img_size), requires_grad=True).to(next(model.parameters()).device)
        logger.add_graph(model, dummy, False)

    except:
        raise RuntimeError("Your dummy input is wrong. Please update it.")

    """Transform input data, compute weight mask and target data"""
    in_prep = deepsmlm.generic.utils.processing.TransformSequence.parse(
        [
            deepsmlm.neuralfitter.filter.FrameFilter,
            deepsmlm.neuralfitter.scale_transform.AmplitudeRescale
        ], param=param)

    # em_filter = deepsmlm.neuralfitter.filter.TarEmitterFilter()
    em_filter = deepsmlm.neuralfitter.filter.NoEmitterFilter()

    # Target generator is a sequence of multiple modules
    tar_gen = deepsmlm.generic.utils.processing.TransformSequence(
        [
            deepsmlm.generic.utils.processing.ParallelTransformSequence.parse(
                [
                    deepsmlm.neuralfitter.target_generator.KernelEmbedding,
                    deepsmlm.generic.utils.processing.Identity
                ],
                param, input_slice=[[0], [1]], merger=lambda x: torch.cat((x[0], x[1].unsqueeze(0)), 1).squeeze(0)),
            deepsmlm.neuralfitter.scale_transform.InverseOffsetRescale.parse(param)
        ],
        input_slice=None)

    # Set frame range for target generator  # this is somewhat dirty ...
    tar_gen.com[0].com[0].ix_low = 0
    tar_gen.com[0].com[0].ix_high = 0

    weight_gen = deepsmlm.neuralfitter.weight_generator.SimpleWeight.parse(param)

    """Setup training and test dataset / data loader."""

    train_ds = deepsmlm.neuralfitter.dataset.SMLMDatasetEngineDataset(
        engine=engine_train,
        em_proc=em_filter,
        frame_proc=in_prep,
        tar_gen=tar_gen,
        weight_gen=weight_gen,
        frame_window=param.HyperParameter.channels_in,
        return_em=False
    )

    test_ds = deepsmlm.neuralfitter.dataset.SMLMDatasetEngineDataset(
        engine=engine_test,
        em_proc=em_filter,
        frame_proc=in_prep,
        tar_gen=tar_gen,
        weight_gen=weight_gen,
        frame_window=param.HyperParameter.channels_in,
        return_em=True
    )

    test_ds.load_from_engine()
    train_ds.load_from_engine()

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameter.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_worker_train,
        pin_memory=False,
        collate_fn=deepsmlm.neuralfitter.utils.pytorch_customs.smlm_collate)

    test_dl = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=param.HyperParameter.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=param.Hardware.num_worker_train,
        pin_memory=False,
        collate_fn=deepsmlm.neuralfitter.utils.pytorch_customs.smlm_collate)

    """Set up post processor"""
    if not param.HyperParameter.suppress_post_processing:
        post_processor = deepsmlm.generic.utils.processing.TransformSequence.parse(
            [
                deepsmlm.neuralfitter.scale_transform.OffsetRescale,
                deepsmlm.neuralfitter.post_processing.Offset2Coordinate,
                deepsmlm.neuralfitter.post_processing.ConsistencyPostprocessing
            ],
            param)
    else:
        post_processor = deepsmlm.neuralfitter.post_processing.NoPostProcessing()

    """Evaluation Specification"""
    matcher = deepsmlm.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)
    # segmentation_eval = deepsmlm.evaluation.SegmentationEvaluation()
    # distance_eval = deepsmlm.evaluation.DistanceEvaluation()

    # useful if we restart a training
    first_epoch = param.HyperParameter.epoch_0 if param.HyperParameter.epoch_0 is not None else 0

    for i in range(first_epoch, param.HyperParameter.epochs):
        logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)

        _ = deepsmlm.neuralfitter.train_val_impl.train(
            model=model,
            optimizer=optimizer,
            loss=criterion,
            dataloader=train_dl,
            grad_rescale=param.HyperParameter.moeller_gradient_rescale,
            epoch=i,
            device=torch.device(param.Hardware.device_train),
            logger=logger
        )

        val_loss, test_out = deepsmlm.neuralfitter.train_val_impl.test(
            model=model,
            optimizer=optimizer,
            loss=criterion,
            dataloader=test_dl,
            grad_rescale=param.HyperParameter.moeller_gradient_rescale,
            post_processor=post_processor,
            epoch=i,
            device=torch.device(param.Hardware.device_train),
            logger=logger
        )

        """Post-Process and Evaluate"""
        log_train_val_progress.post_process_log_test(test_out.loss, loss_scalar=val_loss,
                                                     x=test_out.x, y_out=test_out.y_out, y_tar=test_out.y_tar,
                                                     weight=test_out.weight, em_tar=test_out.em_tar,
                                                     post_processor=post_processor, matcher=matcher, logger=logger,
                                                     step=i)

        lr_scheduler.step(val_loss)

        """Save."""
        model_ls.save(model, val_loss)

        """Load a new dataset"""
        train_ds.load_from_engine()


if __name__ == '__main__':
    setup_train_engine()
