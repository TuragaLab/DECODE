import os
import torch

from deepsmlm.utils.model_io import LoadSaveModel
from deepsmlm.neuralfitter.models.model_offset import OffSetUNetBGBranch


def trace_offset_model(model, channels=3):
    x = torch.rand((2, channels, 64, 64))
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

    file = '/Volumes/kreshuk/lucas/GPU7_Cluster_deployment/deepsmlm/network/2019-10-17/model_singleunet_branch_nobg_32.pt'
    traced_file = file[:-3] + '_jit_traced.pt'

    # model = OffsetUnet(3)
    model = OffSetUNetBGBranch(3, 6)
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
