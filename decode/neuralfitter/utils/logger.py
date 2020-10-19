import time

import matplotlib.pyplot as plt
import torch.utils.tensorboard


class SummaryWriter(torch.utils.tensorboard.SummaryWriter):

    def __init__(self, filter_keys=(), *args, **kwargs):
        """

        Args:
            filter_keys: keys to be filtered in add_scalar_dict method
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.filter_keys = filter_keys

    def add_scalar_dict(self, prefix: str, scalar_dict: dict, global_step=None, walltime=None):
        """
        Adds a couple of scalars that are in a dictionary to the summary.
        Note that this is different from 'add_scalars'

        """

        for name, value in scalar_dict.items():
            if name in self.filter_keys:
                continue

            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)


class NoLog(SummaryWriter):
    """The hardcoded No-Op of the tensorboard SummaryWriter."""

    def __init__(self, *args, **kwargs):
        return

    def add_scalar(self, *args, **kwargs):
        return

    def add_scalars(self, *args, **kwargs):
        return

    def add_scalar_dict(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_figure(self, tag, figure, *args, **kwargs):
        plt.close(figure)
        return

    def add_figures(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
        return

    def add_images(self, *args, **kwargs):
        return

    def add_video(self, *args, **kwargs):
        return

    def add_audio(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_embedding(self, *args, **kwargs):
        return

    def add_pr_curve(self, *args, **kwargs):
        return

    def add_custom_scalars(self, *args, **kwargs):
        return

    def add_mesh(self, *args, **kwargs):
        return

    def add_hparams(self, *args, **kwargs):
        return


class DictLogger(NoLog):
    """
    Simple logger that can log scalars to a dictionary
    """

    def __init__(self):
        super().__init__()
        self.log_dict = {}

    # ToDo: Remove Duplication (make DictLogger inherit from both NoLog and SummaryWriter?)
    def add_scalar_dict(self, prefix: str, scalar_dict: dict, global_step=None, walltime=None):
        for name, value in scalar_dict.items():
            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)

    def add_scalar(self, prefix: str, scalar_value: float, global_step=None, walltime=None):

        if walltime is None:
            walltime = time.time()

        if prefix in self.log_dict:
            if global_step is None:
                global_step = self.log_dict['global_step'] + 1

            self.log_dict[prefix]['scalar'].append(scalar_value)
            self.log_dict[prefix]['step'].append(global_step)
            self.log_dict[prefix]['walltime'].append(walltime)

        else:
            if global_step is None:
                global_step = 0

            val_ini = {'scalar': [scalar_value],
                       'step': [global_step],
                       'walltime': [walltime]}

            self.log_dict.update({prefix: val_ini})


class MultiLogger:
    """
    A 'Meta-Logger', i.e. a logger that calls its components.
    Note all component loggers are assumed to have the same methods.
    """

    def __init__(self, logger):
        def do_for_all(cmp, mthd: str):
            """Execute a method which is present in all cmp sequentially."""

            def idk(*args, **kwargs):
                # for c in cmp:
                return [getattr(c, mthd)(*args, **kwargs) for c in cmp]

            return idk

        self.logger = logger

        # methods of 0th logger
        mthds = [method_name for method_name in dir(self.logger[0]) if callable(getattr(self.logger[0], method_name))]
        mthds = [m for m in mthds if '__' not in m]  # only interested in defomed methods

        for m in mthds:
            setattr(self, m, do_for_all(self.logger, m))
