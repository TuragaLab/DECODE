from collections import namedtuple
import torch
import pprint

InOutParameter = namedtuple('InOutParameter',
                             ['root',
                              'log_comment',
                              'data_mode',
                              'data_set',
                              'model_out',
                              'model_init'])

HyperParamter = namedtuple('HyperParameter',
                           ['dimensions',
                            'channels',
                            'max_emitters',
                            'batch_size',
                            'test_size',
                            'num_epochs',
                            'lr',
                            'device'])  # I know that is not a hyper parameter ...

SimulationParam = namedtuple("SimulationParam",
                             ['pseudo_data_size',
                              'emitter_extent',
                              'psf_extent',
                              'img_size',
                              'density',
                              'photon_range',
                              'bg_pois',
                              'calibration'])



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