import torch

from deepsmlm.neuralfitter.model import DeepSLMN


class LoadSaveModel:
    def __init__(self, output_file, cuda, warmstart_file=None):
        self.warmstart_file = warmstart_file
        self.output_file = output_file
        self.cuda = cuda

    def load_init(self):
        model = DeepSLMN()

        if self.warmstart_file is None:
            model.weight_init()
        else:
            model.load_state_dict(torch.load(self.warmstart_file))

        model.eval()
        return model

    def save(self, model):
        torch.save(model.state_dict(), self.output_file)
