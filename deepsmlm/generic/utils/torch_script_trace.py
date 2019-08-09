import os
import torch

from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.neuralfitter import scale_transform as scaling
from deepsmlm.neuralfitter import post_processing as post
from deepsmlm.generic.utils import processing as utils
from deepsmlm.neuralfitter.models.model_offset import OffsetUnet


def trace_offset_model(model):
    x = torch.rand((2, 3, 64, 64))
    traced_module = torch.jit.trace(model, x)

    return traced_module


def trace_post_processing(post_functional):
    x = torch.rand((2, 5, 64, 64))
    traced_module = torch.jit.trace(post_functional, x)

    return traced_module


if __name__ == '__main__':
    
    """Root folder"""
    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

    file = '/home/lucas/RemoteDeploymentTemp/DeepSMLMv2/network/2019-08-06/model_challenge_roi_10_quant_comp.pt'
    traced_file = file[:-3] + '_jit_traced.pt'

    model = OffsetUnet(3)
    model = LoadSaveModel(model,
                          output_file=None,
                          input_file=file).load_init(False)
    traced_mod = trace_offset_model(model)
    print("This is PyTorch version: {}".format(torch.__version__))
    torch.jit.save(traced_mod, traced_file)

    # psf_extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
    # img_shape = (64, 64)
    #
    # rescale = scaling.OffsetRescale(0.5, 0.5, 750., 10000, 1.2)
    # convert_offset = post.Offset2Coordinate(psf_extent[0], psf_extent[1], img_shape)
    # nms = post.SpeiserPost(0.3, 0.6, 'frames')
    # nms = post.speis_post_functional
    #
    # traced_post = trace_post_processing(nms)
    print("Saved.")
