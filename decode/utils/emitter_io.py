import copy
import pathlib
from typing import Union, Tuple

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
        mapping = {'x': 'x', 'y': 'y', 'z': 'z', 'phot': 'phot', 'frame_ix': 'frame_ix'}

    chunks = pd.read_csv(file, chunksize=100000, **pd_csv_args)
    data = pd.concat(chunks)

    xyz = torch.stack((torch.from_numpy(data[mapping['x']].to_numpy()).float(),
                       torch.from_numpy(data[mapping['y']].to_numpy()).float(),
                       torch.from_numpy(data[mapping['z']].to_numpy()).float()), 1)

    phot = torch.from_numpy(data[mapping['phot']].to_numpy()).float()
    frame_ix = torch.from_numpy(data[mapping['frame_ix']].to_numpy()).long()

    if 'id' in mapping.keys():
        identifier = torch.from_numpy(data[mapping['id']].to_numpy()).long()
    else:
        identifier = None

    return {'xyz': xyz, 'phot': phot, 'frame_ix': frame_ix, 'id': identifier}


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


def save_h5(path: Union[str, pathlib.Path], data: dict, metadata: dict):
    def create_volatile_dataset(group, name, tensor):
        """Empty DS if all nan"""
        if torch.isnan(tensor).all():
            group.create_dataset(name, data=h5py.Empty("f"))
        else:
            group.create_dataset(name, data=tensor.numpy())

    with h5py.File(path, 'w') as f:
        m = f.create_group('meta')
        if any(v is None for v in metadata.values()):
            raise ValueError(f"Cannot save to hdf5 because encountered None in one of {metadata.keys()}")
        m.attrs.update(metadata)

        g = f.create_group('data')
        g.create_dataset('xyz', data=data['xyz'].numpy())
        create_volatile_dataset(g, 'xyz_sig', data['xyz_sig'])
        create_volatile_dataset(g, 'xyz_cr', data['xyz_cr'])

        g.create_dataset('phot', data=data['phot'].numpy())
        create_volatile_dataset(g, 'phot_cr', data['phot_cr'])
        create_volatile_dataset(g, 'phot_sig', data['phot_sig'])

        g.create_dataset('frame_ix', data=data['frame_ix'].numpy())
        g.create_dataset('id', data=data['id'].numpy())
        g.create_dataset('prob', data=data['prob'].numpy())

        g.create_dataset('bg', data=data['bg'].numpy())
        create_volatile_dataset(g, 'bg_cr', data['bg_cr'])
        create_volatile_dataset(g, 'bg_sig', data['bg_sig'])


def load_h5(path) -> Tuple[dict, dict]:

    with h5py.File(path, 'r') as h5:

        data = {
            k: torch.from_numpy(v[:]) for k, v in h5['data'].items() if v.shape is not None
        }
        data.update({  # add the None ones
            k: None for k, v in h5['data'].items() if v.shape is None
        })

        meta = dict(h5['meta'].attrs)

    return data, meta
