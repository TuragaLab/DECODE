import pathlib

import h5py
import numpy as np
import torch

from deepsmlm.generic import emitter


def load_smap_emitter(file: (str, pathlib.Path), mapping: (dict, None) = None, **emitter_kwargs) -> emitter.EmitterSet:
    """

    Args:
        file: .mat file
        mapping (optional): mapping of matlab fields to emitter
        **emitter_kwargs: additional arguments to be parsed to the emitter initialisation

    Returns:

    """
    if mapping is None:
        mapping = {'x': 'xnm', 'y': 'ynm', 'z': 'znm',
                   'phot': 'phot', 'frame_ix': 'frame', 'bg': 'bg'}

    f = h5py.File(file, 'r')

    loc_dict = f['saveloc']['loc']
    xyz = torch.cat([
        torch.from_numpy(np.array(loc_dict[mapping['x']])).permute(1, 0),
        torch.from_numpy(np.array(loc_dict[mapping['y']])).permute(1, 0),
        torch.from_numpy(np.array(loc_dict[mapping['z']])).permute(1, 0)
    ], 1)

    em = emitter.EmitterSet(xyz=xyz,
                            phot=torch.from_numpy(np.array(loc_dict[mapping['phot']])).squeeze(),
                            frame_ix=torch.from_numpy(np.array(loc_dict[mapping['frame']])).squeeze(),
                            bg=torch.from_numpy(np.array(loc_dict[mapping['bg']])).squeeze(),
                            xy_unit='nm', **emitter_kwargs)

    em.frame_ix -= 1  # MATLAB starts at 1, python and all serious languages at 0

    return em
