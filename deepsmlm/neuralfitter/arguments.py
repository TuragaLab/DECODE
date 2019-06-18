import pprint
from collections import namedtuple

import torch

InOutParameter = namedtuple('InOutParameter',
                            [
                                'root',
                                'log_comment',
                                'data_mode',
                                'data_set',
                                'model_out',
                                'model_init'])

LoggerParameter = namedtuple('LoggerParamter',
                             [
                                 'tags'
                             ])

SchedulerParameter = namedtuple('SchedulerParameter',
                                [
                                    'lr_factor',
                                    'lr_patience',
                                    'lr_threshold',
                                    'lr_cooldown',
                                    'lr_verbose',
                                    'sim_factor',
                                    'sim_patience',
                                    'sim_threshold',
                                    'sim_cooldown',
                                    'sim_verbose',
                                    'sim_disabled',
                                    'sim_max_value',
                                ])

HyperParamter = namedtuple('HyperParameter',
                           [
                               'dimensions',
                               'channels',
                               'max_emitters',
                               'min_phot',
                               'data_lifetime',
                               'upscaling',
                               'upscaling_mode',
                               'batch_size',
                               'test_size',
                               'num_epochs',
                               'lr',
                               'device'])  # I know that is not a hyper parameter ...

SimulationParam = namedtuple("SimulationParam",
                             [
                                 'pseudo_data_size',
                                 'emitter_extent',
                                 'psf_extent',
                                 'img_size',
                                 'density',
                                 'emitter_av',
                                 'photon_range',
                                 'bg_pois',
                                 'calibration'])

ScalingParam = namedtuple("ScalingParam",
                          [
                              'dx_max',
                              'dy_max',
                              'z_max',
                              'phot_max',
                              'linearisation_buffer'
                          ])

PostProcessingParam = namedtuple("PostProcessingParam",
                                 {
                                     'single_val_th',
                                     'total_th'
                                 })

EvaluationParam = namedtuple("EvaluationParam",
                             [
                                 'dist_lat',
                                 'dist_ax',
                                 'match_dims'
                             ])


class Args:
    """
    Convenience for training arguments.
    """
    def __init__(self, cuda=True, epochs=100, num_prints=5, sm_sigma=1,
                 root_folder=None, data_path=None, model_in_path=None, model_out_path=None):
        self.cuda = cuda if torch.cuda.is_available() else False
        self.epochs = epochs
        self.num_prints = num_prints
        self.sm_sigma = sm_sigma

        self.root_folder = root_folder
        self.data_path = data_path
        self.model_in_path = model_in_path
        self.model_out_path = model_out_path

    def print_confirmation(self):
        """
        Print arguments and wait for confirmation.
        :return: void
        """
        print('The configured arguments are:')
        pp = pprint.PrettyPrinter(width=-1)
        pp.pprint(vars(self))
        # input('Press Enter to continue ...')