import datetime
import os
import shutil
import socket
from pathlib import Path

import click
import torch

import deepsmlm.evaluation
import deepsmlm.neuralfitter
import deepsmlm.neuralfitter.utils
import deepsmlm.simulation
import deepsmlm.utils
from deepsmlm.neuralfitter.utils import log_train_val_progress

import resource



def setup_random_simulation(param):
    """
        Setup the actual simulation

        0. Define PSF function (load the calibration)
        1. Define our struture from which we sample (random prior in 3D) and its photophysics
        2. Define background and noise
        3. Setup simulation and datasets
        """

    native_psf_img_size = (int(param.Simulation.psf_extent[0][1] - param.Simulation.psf_extent[0][0]),
                           int(param.Simulation.psf_extent[1][1] - param.Simulation.psf_extent[1][0]))

    psf = deepsmlm.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file).init_spline(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=native_psf_img_size,
        cuda_kernel=True if param.Hardware.device_simulation[:4] == 'cuda' else False,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    )

    """Structure Prior"""
    prior_struct = deepsmlm.simulation.structure_prior.RandomStructure(
        xextent=param.Simulation.emitter_extent[0],
        yextent=param.Simulation.emitter_extent[1],
        zextent=param.Simulation.emitter_extent[2])

    if param.Simulation.mode == 'acquisition':
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size)

    elif param.Simulation.mode == 'samples':
        frame_range_train = (-((param.HyperParameter.channels_in - 1) // 2),
                             (param.HyperParameter.channels_in - 1) // 2)
    else:
        raise ValueError

    prior_train = deepsmlm.simulation.emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=prior_struct, frames=frame_range_train)

    """Define our background and noise model."""
    bg = deepsmlm.simulation.background.UniformBackground.parse(param)

    noise = deepsmlm.simulation.camera.Photon2Camera.parse(param, device=param.Hardware.device_simulation)

    simulation_train = deepsmlm.simulation.simulator.Simulation(
        psf=psf, em_sampler=prior_train, background=bg, noise=noise, frame_range=frame_range_train)

    frame_range_test = (0, param.TestSet.test_size)

    prior_test = deepsmlm.simulation.emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=prior_struct, frames=frame_range_test)

    simulation_test = deepsmlm.simulation.simulator.Simulation(
        psf=psf, em_sampler=prior_test, background=bg, noise=noise, frame_range=frame_range_test)

    return simulation_train, simulation_test


def setup_trainer(simulator_train, simulator_test, logger, model_out, param):
    """Set model, optimiser, loss and schedulers"""
    models_available = {
        'SigmaMUNet': deepsmlm.neuralfitter.models.SigmaMUNet,
        'DoubleMUnet': deepsmlm.neuralfitter.models.model_param.DoubleMUnet,
        'SimpleSMLMNet': deepsmlm.neuralfitter.models.model_param.SimpleSMLMNet,
    }

    model = models_available[param.HyperParameter.architecture]
    model = model.parse(param)

    model_ls = deepsmlm.utils.model_io.LoadSaveModel(model,
                                                     output_file=model_out,
                                                     input_file=param.InOut.model_init)

    model = model_ls.load_init()
    model = model.to(torch.device(param.Hardware.device))

    # Small collection of optimisers
    optimizer_available = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }

    optimizer = optimizer_available[param.HyperParameter.optimizer]
    optimizer = optimizer(model.parameters(), **param.HyperParameter.opt_param)

    """Loss function."""
    criterion = deepsmlm.neuralfitter.losscollection.GaussianMMLoss(xextent=param.Simulation.psf_extent[0],
                                                                    yextent=param.Simulation.psf_extent[1],
                                                                    img_shape=param.Simulation.img_size,
                                                                    device=param.Hardware.device,
                                                                    chweight_stat=param.HyperParameter.chweight_stat)

    """Learning Rate and Simulation Scheduling"""
    lr_scheduler_available = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'StepLR': torch.optim.lr_scheduler.StepLR
    }
    lr_scheduler = lr_scheduler_available[param.HyperParameter.learning_rate_scheduler]
    lr_scheduler = lr_scheduler(optimizer, **param.HyperParameter.learning_rate_scheduler_param)

    """Setup gradient modification"""
    grad_mod = param.HyperParameter.grad_mod

    """Log the model"""
    try:
        dummy = torch.rand((2, param.HyperParameter.channels_in,
                            *param.Simulation.img_size), requires_grad=True).to(torch.device(param.Hardware.device))
        logger.add_graph(model, dummy)

    except:
        raise RuntimeError("Your dummy input is wrong. Please update it.")

    """Transform input data, compute weight mask and target data"""
    frame_proc = deepsmlm.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
    bg_frame_proc = None

    if param.HyperParameter.emitter_label_photon_min is not None:
        em_filter = deepsmlm.neuralfitter.em_filter.PhotonFilter(param.HyperParameter.emitter_label_photon_min)
    else:
        em_filter = deepsmlm.neuralfitter.em_filter.NoEmitterFilter()

    tar_gen = deepsmlm.neuralfitter.utils.processing.TransformSequence(
        [
            deepsmlm.neuralfitter.target_generator.ParameterListTarget(n_max=param.HyperParameter.max_number_targets,
                                                                       xextent=param.Simulation.psf_extent[0],
                                                                       yextent=param.Simulation.psf_extent[1],
                                                                       ix_low=0, ix_high=0,
                                                                       squeeze_batch_dim=True),

            deepsmlm.neuralfitter.scale_transform.ParameterListRescale(phot_max=param.Scaling.phot_max,
                                                                       z_max=param.Scaling.z_max,
                                                                       bg_max=param.Scaling.bg_max)
        ])

    weight_gen = None

    if param.Simulation.mode == 'acquisition':
        train_ds = deepsmlm.neuralfitter.dataset.SMLMLiveDataset(simulator=simulator_train, em_proc=None,
                                                                 frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                                                                 tar_gen=tar_gen, weight_gen=weight_gen,
                                                                 frame_window=param.HyperParameter.channels_in,
                                                                 pad=None, return_em=False)

        train_ds.sample(True)

    elif param.Simulation.mode == 'samples':
        train_ds = deepsmlm.neuralfitter.dataset.SMLMLiveSampleDataset(simulator=simulator_train, em_proc=None,
                                                                       frame_proc=frame_proc,
                                                                       bg_frame_proc=bg_frame_proc,
                                                                       tar_gen=tar_gen, weight_gen=weight_gen,
                                                                       frame_window=param.HyperParameter.channels_in,
                                                                       return_em=False,
                                                                       ds_len=param.HyperParameter.pseudo_ds_size)

    test_ds = deepsmlm.neuralfitter.dataset.SMLMLiveDataset(simulator=simulator_test, em_proc=em_filter,
                                                            frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                                                            tar_gen=tar_gen, weight_gen=weight_gen,
                                                            frame_window=param.HyperParameter.channels_in,
                                                            pad=None, return_em=True)

    test_ds.sample(True)

    """Set up post processor"""
    if param.PostProcessing is None:
        post_processor = deepsmlm.neuralfitter.post_processing.NoPostProcessing(xy_unit='px',
                                                                                px_size=param.Camera.px_size)

    elif param.PostProcessing == 'LookUp':
        post_processor = deepsmlm.neuralfitter.utils.processing.TransformSequence([

            deepsmlm.neuralfitter.scale_transform.InverseParamListRescale(phot_max=param.Scaling.phot_max,
                                                                          z_max=param.Scaling.z_max,
                                                                          bg_max=param.Scaling.bg_max),

            deepsmlm.neuralfitter.post_processing.Offset2Coordinate.parse(param),

            deepsmlm.neuralfitter.post_processing.LookUpPostProcessing(raw_th=param.PostProcessingParam.raw_th,
                                                                       pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],
                                                                       xy_unit='px',
                                                                       px_size=param.Camera.px_size)
        ])

    elif param.PostProcessing == 'Consistency':
        raise NotImplementedError

    else:
        raise ValueError


    """Evaluation Specification"""
    matcher = deepsmlm.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)

    return train_ds, test_ds, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher


def setup_dataloader(param, train_ds, test_ds=None):
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameter.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_worker_train,
        pin_memory=True,
        collate_fn=deepsmlm.neuralfitter.utils.collate.smlm_collate)

    if test_ds is not None:

        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=param.HyperParameter.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=param.Hardware.num_worker_train,
            pin_memory=False,
            collate_fn=deepsmlm.neuralfitter.utils.collate.smlm_collate)
    else:

        test_dl = None

    return train_dl, test_dl


@click.command()
@click.option('--cuda_ix', '-i', default=None, required=False, type=int,
              help='Specify the cuda device index or set it to false.')
@click.option('--param_file', '-p', required=True,
              help='Specify your parameter file (.yml or .json).')
@click.option('--debug', '-d', default=False, is_flag=True,
              help='Debug the specified parameter file. Will reduce ds size for example.')
@click.option('--num_worker_override', '-w', default=None, type=int,
              help='Override the number of workers for the dataloaders.')
@click.option('--no_log', '-n', default=False, is_flag=True,
              help='Set no log if you do not want to log the current run.')
@click.option('--log_folder', '-l', default='runs',
              help='Specify the (parent) folder you want to log to. If rel-path, relative to DeepSMLM root.')
@click.option('--log_comment', '-c', default=None,
              help='Add a log_comment to the run.')
def live_engine_setup(cuda_ix, param_file, debug, num_worker_override, no_log, log_folder, log_comment):
    """
    Sets up the engine for simulation the DeepSMLM training data

    Args:
        param_file:
        debug:
        num_worker_override:

    Returns:

    """

    """Load Parameters and back them up to the network output directory"""
    param_file = Path(param_file)
    param = deepsmlm.utils.param_io.ParamHandling().load_params(param_file)

    """Experiment ID"""
    if not debug:
        experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + socket.gethostname()

        if log_comment:
            experiment_id = experiment_id + '_' + log_comment

    else:
        experiment_id = 'debug'

    """Set up unique folder for experiment"""
    experiment_path = Path(param.InOut.experiment_out) / Path(experiment_id)
    if not experiment_path.parent.exists():
        experiment_path.parent.mkdir()

    if debug:
        experiment_path.mkdir(exist_ok=True)
    else:
        experiment_path.mkdir(exist_ok=False)

    model_out = experiment_path / Path('model.pt')

    # Backup the parameter file under the network output path with the experiments ID
    param_backup = experiment_path / Path('param_run').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_backup)

    if debug:
        deepsmlm.utils.param_io.ParamHandling.convert_param_debug(param)

    if num_worker_override is not None:
        param.Hardware.num_worker_train = num_worker_override

    """Hardware / Server stuff."""
    cuda_ix = int(param.Hardware.device_ix) if cuda_ix is None else cuda_ix
    torch.cuda.set_device(cuda_ix)  # do this instead of set env variable, because torch is inevitably already imported
    os.nice(param.Hardware.unix_niceness)

    if param.Hardware.torch_multiprocessing_sharing_strategy is None:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    else:
        torch.multiprocessing.set_sharing_strategy(param.Hardware.torch_multiprocessing_sharing_strategy)

    torch.set_num_threads(param.Hardware.torch_threads)

    """Setup Log System"""
    if no_log:
        logger = deepsmlm.neuralfitter.utils.logger.NoLog()

    else:
        log_folder = log_folder + '/' + experiment_id
        logger = deepsmlm.neuralfitter.utils.logger.SummaryWriter(log_dir=log_folder)

    sim_train, sim_test = setup_random_simulation(param)
    ds_train, ds_test, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher = \
        setup_trainer(sim_train, sim_test, logger, model_out, param)

    dl_train, dl_test = setup_dataloader(param, ds_train, ds_test)

    # useful if we restart a training
    first_epoch = param.HyperParameter.epoch_0 if param.HyperParameter.epoch_0 is not None else 0

    for i in range(first_epoch, param.HyperParameter.epochs):
        logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)

        if i >= 1:

            train_loss = deepsmlm.neuralfitter.train_val_impl.train(
                model=model,
                optimizer=optimizer,
                loss=criterion,
                dataloader=dl_train,
                grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                grad_mod=grad_mod,
                epoch=i,
                device=torch.device(param.Hardware.device),
                logger=logger
            )

        val_loss, test_out = deepsmlm.neuralfitter.train_val_impl.test(model=model, loss=criterion, dataloader=dl_test,
                                                                       epoch=i,
                                                                       device=torch.device(param.Hardware.device))

        """Post-Process and Evaluate"""
        log_train_val_progress.post_process_log_test(loss_cmp=test_out.loss, loss_scalar=val_loss,
                                                     x=test_out.x, y_out=test_out.y_out, y_tar=test_out.y_tar,
                                                     weight=test_out.weight, em_tar=test_out.em_tar,
                                                     px_border=-0.5, px_size=1.,
                                                     post_processor=post_processor, matcher=matcher, logger=logger,
                                                     step=i)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        model_ls.save(model, val_loss)

        """Draw new samples Samples"""
        if param.Simulation.mode == 'acquisition':
            ds_train.sample(True)
        elif param.Simulation.mode != 'samples':
            raise ValueError


if __name__ == '__main__':
    live_engine_setup()
