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

    #ToDo: unfinished logging here


if __name__ == '__main__':
    pass