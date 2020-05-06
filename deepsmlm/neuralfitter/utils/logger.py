import matplotlib.pyplot as plt
import torch.utils.tensorboard


class SummaryWriterSoph(torch.utils.tensorboard.SummaryWriter):

    def add_scalar_dict(self, prefix: str, scalar_dict: dict, global_step=None, walltime=None):
        """
        Adds a couple of scalars that are in a dictionary to the summary.
        Note that this is different from 'add_scalars'

        """

        for name, value in scalar_dict.items():
            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)


class NoLog(SummaryWriterSoph):
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
