import os
import torch

from deepsmlm.generic.inout.load_save_model import LoadSaveModel
# from deepsmlm.neuralfitter import scale_transform as scaling
# from deepsmlm.neuralfitter import post_processing as post
# from deepsmlm.generic.utils import processing as utils
from deepsmlm.neuralfitter.models.model_offset import OffsetUnet

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


def trace_offset_model(model):
    x = torch.rand((2, 3, 64, 64))
    traced_module = torch.jit.trace(model, x)

    return traced_module


def trace_post_processing(post_functional):
    x = torch.rand((2, 5, 64, 64))
    traced_module = torch.jit.trace(post_functional, x)

    return


if __name__ == '__main__':

    # psf_extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
    # img_shape = (64, 64)
    #
    # rescale = scaling.OffsetRescale(0.5, 0.5, 750., 10000, 1.2)
    # convert_offset = post.Offset2Coordinate(psf_extent[0], psf_extent[1], img_shape)
    # nms = post.SpeiserPost(0.3, 0.6)
    #
    # post_processing_functional = utils.TransformSequence([convert_offset]).forward
    # trace_post_processing(post_processing_functional)

    model = OffsetUnet(3)
    model = LoadSaveModel(model,
                          output_file=None,
                          input_file='/home/lucas/RemoteDeploymentTemp/deepsmlm/network/2019-05-08/model_offset_47.pt').load_init(False)
    traced_mod = trace_offset_model(model)
    torch.jit.save(traced_mod, 'helloooo.pt')
