import datetime
import os
import socket
from pathlib import Path

import click

import deepsmlm.generic.inout.load_calibration
import deepsmlm.generic.inout.util
import deepsmlm.generic.inout.write_load_param as dsmlm_par
import deepsmlm.generic.utils
import deepsmlm.neuralfitter.dataset
import deepsmlm.simulation.background
import deepsmlm.simulation.camera
import deepsmlm.simulation.engine
import deepsmlm.simulation.psf_kernel

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
    assert torch.cuda.device_count() <= param.Hardware.max_cuda_devices
    torch.set_num_threads(param.Hardware.torch_threads_sim)

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

    frame_range_train = (0, param.HyperParameter.pseudo_ds_size)
    frame_range_test = (0, param.HyperParameter.test_size)

    prior_train = deepsmlm.simulation.emitter_generator.EmitterPopperMultiFrame.parse(
        param, structure=prior_struct, frames=frame_range_train)

    prior_test = deepsmlm.simulation.emitter_generator.EmitterPopperMultiFrame.parse(
        param, structure=prior_struct, frames=frame_range_test)

    """Define our background and noise model."""
    if param.Simulation.bg_perlin_amplitude is None:
        bg = deepsmlm.simulation.background.UniformBackground.parse(param)
    else:
        bg = deepsmlm.generic.utils.processing.TransformSequence.parse(
            [deepsmlm.simulation.background.UniformBackground,
             deepsmlm.simulation.background.PerlinBackground], param, input_slice=[[0], [0]])

    noise = deepsmlm.simulation.camera.Photon2Camera.parse(param, device=param.Hardware.device_sim)

    simulation_train = deepsmlm.simulation.simulator.Simulation(
        psf=psf, em_sampler=prior_train, background=bg, noise=noise, frame_range=frame_range_train)

    simulation_test = deepsmlm.simulation.simulator.Simulation(
        psf=psf, em_sampler=prior_test, background=bg, noise=noise, frame_range=frame_range_test)

    """Setup the simulation engine"""
    simulation_engine = deepsmlm.simulation.engine.DatasetStreamEngine(
        cache_dir=cache_dir, exp_id=exp_id, buffer_size=param.HyperParameter.ds_buffer,
        sim_train=simulation_train, sim_test=simulation_test)

    simulation_engine.run()


if __name__ == '__main__':
    smlm_engine_setup()
