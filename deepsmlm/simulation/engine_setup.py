import click
import datetime
import socket
from pathlib import Path
import os

import deepsmlm.simulation.psf_kernel
import deepsmlm.generic.utils
import deepsmlm.simulation.background
import deepsmlm.simulation.camera
import deepsmlm.generic.inout.write_load_param as dsmlm_par
import deepsmlm.generic.inout.util
import deepsmlm.simulation.engine
import deepsmlm.generic.inout.load_calibration
import deepsmlm.neuralfitter.dataset

deepsmlm_root = Path(__file__).parent.parent.parent


@click.command()
@click.option('--param_file', '-p', required=True,
              help='Specify your parameter file (.yml or .json).')
@click.option('--cache_dir', '-c', default=deepsmlm_root / Path('cachedir/simulation_engine'),
              help='Overwrite the cache folder in which the simulation engine stores the results')
@click.option('--exp_id', '-e', required=True,
              help='Specify the experiments id under which the engine stores the results.')
@click.option('--debug_param', '-d', default=False, is_flag=True,
              help='Debug the specified parameter file. Will reduce ds size for example.')
@click.option('--num_worker_override', '-w', default=None, type=int,
              help='Override the number of workers for the dataloaders.')
def smlm_engine_setup(param_file, cache_dir, exp_id, debug_param=False, num_worker_override=None):
    """
    Sets up the engine for simulation the DeepSMLM training data

    Args:
        param_file:
        cache_dir:
        exp_id:
        debug_param:
        num_worker_override:

    Returns:

    """
    """
    This is mainly boilerplate code in which setup all the things for proper simulation.
    0. Boilerplate parameter loading and some server assertions (using only one GPU etc.)
    1. Setting up the actual simulation
    """

    """Load Parameters"""
    param = dsmlm_par.ParamHandling().load_params(param_file)

    if exp_id is None:
        exp_id = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()

    if debug_param:
        dsmlm_par.ParamHandling.convert_param_debug(param)

    if num_worker_override is not None:
        param.Hardware.num_worker_sim = num_worker_override

    """Hardware / Server stuff."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param.Hardware.device_sim_ix)
    os.nice(param.Hardware.unix_niceness_sim)  # set niceness of process

    import torch
    # torch.multiprocessing.set_sharing_strategy('file_system')  # does not seem to work with spawn method together

    """Set multiprocessing strategy to spawn, otherwise you get errors"""
    # if param.Hardware.device_sim[:4] == 'cuda':
    import multiprocessing as mp
    mp.set_start_method('forkserver')

    assert torch.cuda.device_count() <= param.Hardware.max_cuda_devices
    torch.set_num_threads(param.Hardware.torch_threads)

    """
    Setup the actual simulation
    
    0. Define PSF function (load the calibration)
    1. Define our struture from which we sample (random prior in 3D) and its photophysics
    2. Define background and noise
    3. Setup simulation and datasets
    """
    psf = deepsmlm.generic.inout.load_calibration.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file).init_spline(
            xextent=param.Simulation.psf_extent[0],
            yextent=param.Simulation.psf_extent[1],
            img_shape=param.Simulation.img_size,
        cuda_kernel=True if param.Hardware.device_sim[:4] == 'cuda' else False
    )

    """Structure Prior"""
    prior_struct = deepsmlm.simulation.structure_prior.RandomStructure(
        xextent=param.Simulation.emitter_extent[0],
        yextent=param.Simulation.emitter_extent[1],
        zextent=param.Simulation.emitter_extent[2])

    prior = deepsmlm.simulation.emitter_generator.EmitterPopperMultiFrame(
        structure=prior_struct,
        xy_unit=param.Simulation.xy_unit,
        px_size=param.Camera.px_size,
        density=param.Simulation._density,
        intensity_mu_sig=param.Simulation.intensity_mu_sig,
        lifetime=param.Simulation.lifetime_avg,
        frames=3,
        emitter_av=param.Simulation._emitter_av,
        intensity_th=param.Simulation.intensity_th)

    """Define our background and noise model."""
    if param.Simulation.bg_perlin_amplitude is None:
        bg = deepsmlm.simulation.background.UniformBackground.parse(param)
    else:
        bg = deepsmlm.generic.utils.processing.TransformSequence.parse(
            [deepsmlm.simulation.background.UniformBackground,
             deepsmlm.simulation.background.PerlinBackground], param, input_slice=[[0], [0]])

    noise = deepsmlm.simulation.camera.Photon2Camera.parse(param)

    """
    Here we define some constants to give all possibilities during training. 
    Unnecessary stuff can be still kicked out later
    """
    # use frame range of 3 all the time, maybe kick out unnecessary frames later
    frame_range = (-1, 1)

    simulation = deepsmlm.simulation.simulator.Simulation(psf=psf, em_sampler=prior, background=bg, noise=noise,
                                                          frame_range=frame_range)

    ds_train = deepsmlm.simulation.engine.SMLMSimulationDatasetOnFly(simulator=simulation,
                                                                     ds_size=param.HyperParameter.pseudo_ds_size)
    ds_test = deepsmlm.simulation.engine.SMLMSimulationDatasetOnFly(simulator=simulation,
                                                                    ds_size=param.HyperParameter.test_size)

    simulation_engine = deepsmlm.simulation.engine.SampleStreamEngine(cache_dir=cache_dir,
                                                                      exp_id=exp_id,
                                                                      cpu_worker=param.Hardware.num_worker_sim,
                                                                      buffer_size=param.HyperParameter.ds_buffer,
                                                                      ds_train=ds_train,
                                                                      ds_test=ds_test)

    simulation_engine.run()
    

if __name__ == '__main__':
    smlm_engine_setup()
