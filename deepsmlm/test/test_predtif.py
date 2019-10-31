import pytest
import torch
import os

from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.generic.inout.write_load_param import load_params
from deepsmlm.generic.utils.processing import TransformSequence
from deepsmlm.neuralfitter.models.model_offset import OffsetUnet
from deepsmlm.neuralfitter.post_processing import Offset2Coordinate, SpeiserPost
from deepsmlm.neuralfitter.pred_tif import PredictEvalTif
from deepsmlm.neuralfitter.scale_transform import OffsetRescale


if __name__ == '__main__':
    deepsmlm_root = '/home/lucas/RemoteDeploymentTemp/DeepSMLMv2/'
    os.chdir(deepsmlm_root)

    tifs = '/home/lucas/Documents/SMLM Challenge/MT0.N1.HD/sequence-as-stack-MT0.N1.HD-AS-Exp.tif'
    activations = '/home/lucas/Documents/SMLM Challenge/MT0.N1.HD/activations.csv'
    model_file = 'network/2019-08-15_replicate/model_nofg_challengebg_7.pt'
    param_file = 'network/2019-08-15_replicate/model_nofg_challengebg_param.json'

    param = load_params(param_file)

    model = LoadSaveModel(OffsetUnet(n_channels=3), None, input_file=model_file).load_init()

    post_processor = TransformSequence([
        OffsetRescale(param['Scaling']['dx_max'],
                      param['Scaling']['dy_max'],
                      param['Scaling']['z_max'],
                      param['Scaling']['phot_max'],
                      param['Scaling']['linearisation_buffer']),
        Offset2Coordinate(param['Simulation']['psf_extent'][0],
                          param['Simulation']['psf_extent'][1],
                          param['Simulation']['img_size']),
        SpeiserPost(param['PostProcessing']['single_val_th'],
                    param['PostProcessing']['total_th'],
                'emitters_batch')
    ])

    pred = PredictEvalTif(tifs, activations, model, post_processor, device='cuda')
    pred.load_tif_csv()

    pred.frames = pred.frames[:16]
    emitters = pred.forward()
