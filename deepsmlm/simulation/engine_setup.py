import click
import datetime
import socket
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils

import deepsmlm.generic.psf_kernel
import deepsmlm.generic.utils

import deepsmlm.generic.background
import deepsmlm.generic.phot_camera
import deepsmlm.generic.inout.write_load_param as dsmlm_par
import deepsmlm.generic.inout.util
import deepsmlm.simulation.engine
import deepsmlm.generic.inout.load_calibration
import deepsmlm.neuralfitter.dataset

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


@click.command()
@click.option('--param_file', '-p', required=True,
              help='Specify your parameter file (.yml or .json).')
@click.option('--cache_dir', '-c', default=deepsmlm_root + 'cachedir/simulation_engine',
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
    param_file = deepsmlm.generic.inout.util.add_root_relative(param_file, deepsmlm_root)
    if param_file is None:
        raise ValueError("Parameters not specified. "
                         "Parse the parameter file via -p [Your parameeter.json]")
    param = dsmlm_par.ParamHandling().load_params(param_file)

    if exp_id is None:
        exp_id = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()

    if debug_param:
        dsmlm_par.ParamHandling.convert_param_debug(param)

    if num_worker_override is not None:
        param.Hardware.num_worker_sim = num_worker_override

    # modify some parameters here
    # ToDo: Remove this!
    param.InOut.calibration_file = deepsmlm_root + param.InOut.calibration_file
    # param.HyperParameter.pseudo_ds_size = 2048

    """Server stuff."""
    assert torch.cuda.device_count() <= 1
    torch.set_num_threads(param.Hardware.torch_threads)

    """
    Setup the actual simulation
    
    0. Define PSF function (load the calibration)
    1. Define our struture from which we sample (random prior in 3D) and its photophysics
    2. Define background and noise
    3. Setup simulation and datasets
    """
    psf = deepsmlm.generic.inout.load_calibration.SMAPSplineCoefficient(
        file=param.InOut.calibration_file).init_spline(
            xextent=param.Simulation.psf_extent[0],
            yextent=param.Simulation.psf_extent[1],
            img_shape=param.Simulation.img_size
    )

    """Structure Prior"""
    prior_struct = deepsmlm.simulation.structure_prior.RandomStructure(
        xextent=param.Simulation.emitter_extent[0],
        yextent=param.Simulation.emitter_extent[1],
        zextent=param.Simulation.emitter_extent[2])

    prior = deepsmlm.simulation.emittergenerator.EmitterPopperMultiFrame(
        prior_struct,
        density=param.Simulation.density,
        intensity_mu_sig=param.Simulation.intensity_mu_sig,
        lifetime=param.Simulation.lifetime_avg,
        num_frames=3,
        emitter_av=param.Simulation.emitter_av,
        intensity_th=param.Simulation.intensity_th)

    """Define our background and noise model."""
    if param.Simulation.bg_perlin_amplitude is None:
        bg = deepsmlm.generic.background.UniformBackground.parse(param)
    else:
        bg = deepsmlm.generic.utils.processing.TransformSequence.parse(
            [deepsmlm.generic.background.UniformBackground,
             deepsmlm.generic.background.PerlinBackground], param)

    noise = deepsmlm.generic.phot_camera.Photon2Camera.parse(param)

    """
    Here we define some constants to give all possibilities during training. 
    Unnecessary stuff can be still kicked out later
    """
    # use frame range of 3 all the time, maybe kick out unnecessary frames later
    frame_range = (-1, 1)
    predict_bg = True

    simulation = deepsmlm.simulation.simulator.Simulation(
        em=prior,
        extent=param.Simulation.emitter_extent,
        psf=psf,
        background=bg,
        noise=noise,
        frame_range=frame_range,
        out_bg=predict_bg)

    ds_train = deepsmlm.neuralfitter.dataset.SMLMSimulationDatasetOnFly(simulator=simulation,
                                                                        ds_size=param.HyperParameter.pseudo_ds_size)
    ds_test = deepsmlm.neuralfitter.dataset.SMLMSimulationDatasetOnFly(simulator=simulation,
                                                                       ds_size=param.HyperParameter.test_size)

    simulation_engine = deepsmlm.simulation.engine.SimulationEngine(cache_dir=cache_dir,
                                                                    exp_id=exp_id,
                                                                    cpu_worker=10,
                                                                    buffer_size=3,
                                                                    ds_train=ds_train,
                                                                    ds_test=None)

    simulation_engine.run()
    

if __name__ == '__main__':
    smlm_engine_setup()
