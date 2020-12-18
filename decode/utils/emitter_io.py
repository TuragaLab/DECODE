import copy
import pathlib

import h5py
import numpy as np
import pandas as pd
import torch


challenge_mapping = {'x': 'xnano',
                     'y': 'ynano',
                     'z': 'znano',
                     'frame_ix': 'frame',
                     'phot': 'intensity ',  # this is really odd, there is a space after intensity
                     'id': 'Ground-truth'}

deepstorm3d_mapping = copy.deepcopy(challenge_mapping)
deepstorm3d_mapping['phot'] = 'intensity'


def load_csv(file: (str, pathlib.Path), mapping: (None, dict) = None, **pd_csv_args) -> dict:
    """
    Loads a CSV file which does provide a header.

    Args:
        file: path to file
        mapping: mapping dictionary with keys ('x', 'y', 'z', 'phot', 'id', 'frame_ix')
        pd_csv_args: additional keyword arguments to be parsed to the pandas csv reader

    Returns:
        dict: dictionary which can readily be converted to an EmitterSet by EmitterSet(**out_dict)
    """
    if mapping is None:
        mapping = {'x': 'x', 'y': 'y', 'z': 'z', 'phot': 'phot', 'frame_ix': 'frame_ix', 'x_sig': 'x_sig', 'y_sig': 'y_sig', 'z_sig': 'z_sig'}

    chunks = pd.read_csv(file, chunksize=100000, **pd_csv_args)
    data = pd.concat(chunks)

    xyz = torch.stack((torch.from_numpy(data[mapping['x']].to_numpy()).float(),
                       torch.from_numpy(data[mapping['y']].to_numpy()).float(),
                       torch.from_numpy(data[mapping['z']].to_numpy()).float()), 1)

    xyz_sig = torch.stack((torch.from_numpy(data[mapping['x_sig']].to_numpy()).float(),
                           torch.from_numpy(data[mapping['y_sig']].to_numpy()).float(),
                           torch.from_numpy(data[mapping['z_sig']].to_numpy()).float()), 1)

    phot = torch.from_numpy(data[mapping['phot']].to_numpy()).float()
    frame_ix = torch.from_numpy(data[mapping['frame_ix']].to_numpy()).long()

    if 'id' in mapping.keys():
        identifier = torch.from_numpy(data[mapping['id']].to_numpy()).long()
    else:
        identifier = None

    return {'xyz': xyz, 'phot': phot, 'frame_ix': frame_ix, 'id': identifier, 'xyz_sig': xyz_sig}


def save_csv(file: (str, pathlib.Path), data: dict):
    def convert_dict_torch_numpy(data: dict) -> dict:
        """Convert all torch tensors in dict to numpy."""
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.numpy()
        return data

    def change_to_one_dim(data: dict) -> dict:
        """
        Change xyz tensors to be one-dimensional.

        Args:
            data: emitterset as dictionary

        """
        xyz = data.pop('xyz')
        xyz_cr = data.pop('xyz_cr')
        xyz_sig = data.pop('xyz_sig')

        data_one_dim = {'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]}
        data_one_dim.update(data)
        data_one_dim.update({'x_cr': xyz_cr[:, 0], 'y_cr': xyz_cr[:, 1], 'z_cr': xyz_cr[:, 2]})
        data_one_dim.update({'x_sig': xyz_sig[:, 0], 'y_sig': xyz_sig[:, 1], 'z_sig': xyz_sig[:, 2]})

        return data_one_dim

    """Change torch to numpy and convert 2D elements to 1D"""
    data = copy.deepcopy(data)
    data.pop('px_size')
    data = change_to_one_dim(convert_dict_torch_numpy(data))

    df = pd.DataFrame.from_dict(data)
    df.to_csv(file, index=False)


def load_smap(file: (str, pathlib.Path), mapping: (dict, None) = None) -> dict:
    """

    Args:
        file: .mat file
        mapping (optional): mapping of matlab fields to emitter. Keys must be x,y,z,phot,frame_ix,bg
        **emitter_kwargs: additional arguments to be parsed to the emitter initialisation

    Returns:

    """
    if mapping is None:
        mapping = {'x': 'xnm', 'y': 'ynm', 'z': 'znm',
                   'phot': 'phot', 'frame_ix': 'frame', 'bg': 'bg'}

    f = h5py.File(file, 'r')

    loc_dict = f['saveloc']['loc']

    emitter_dict = {
        'xyz': torch.cat([
            torch.from_numpy(np.array(loc_dict[mapping['x']])).permute(1, 0),  # will always be 2D
            torch.from_numpy(np.array(loc_dict[mapping['y']])).permute(1, 0),
            torch.from_numpy(np.array(loc_dict[mapping['z']])).permute(1, 0)
        ], 1),

        'phot': torch.from_numpy(np.array(loc_dict[mapping['phot']])).squeeze(),
        'frame_ix': torch.from_numpy(np.array(loc_dict[mapping['frame_ix']])).squeeze().long(),
        'bg': torch.from_numpy(np.array(loc_dict[mapping['bg']])).squeeze().float()
    }

    emitter_dict['frame_ix'] -= 1  # MATLAB starts at 1, python and all serious languages at 0

    return emitter_dict
