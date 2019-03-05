import os
import sys

import torch
from torch.utils.data import DataLoader

from deepsmlm.neuralfitter.model import DeepLoco
from deepsmlm.generic.inout.load_save_emitter import TiffInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.neuralfitter.dataset import UnsupervisedDataset


class Forward:
    def __init__(self, model, extent, frames, model_file, multi_in=False):
        self.data_smlm = UnsupervisedDataset(extent, frames, multi_frame_output=multi_in)
        self.data_loader = DataLoader(self.data_smlm, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)
        self.model = LoadSaveModel(model, cuda=torch.cuda.is_available(), output_file=None, input_file=model_file).load_init()
        self.model.eval()

    def forward_wrapper(self):
        return self.forward(self.data_loader, self.model)

    @staticmethod
    def forward(data_loader, model):
        output = [None] * data_loader.__len__()
        for i, (frames, _) in enumerate(data_loader):
            output[i] = model(frames)

        return output


if __name__ == '__main__':
    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'
    # tiff_file = deepsmlm_root + sys.argv[1]
    #
    # _, extent, frames = TiffInterface().load_binary(tiff_file)
    extent = ((-0.5, 63.5), (-0.5, 63.5), (0, 0))
    frames = torch.rand((1, 1, 64, 64))
    model_file = deepsmlm_root + 'network/2019-03-05 Spline/effective_2d_spline.pt'
    d = Forward(DeepLoco(extent, ch_in=1, dim_out=3), extent, frames, model_file)
