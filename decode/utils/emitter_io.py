import json
import pathlib

import copy
import h5py
import linecache
import numpy as np
import pandas as pd
import torch
from typing import Union, Tuple, Optional

from decode.generic.emitter import EmitterSet
from decode.utils import bookkeeping

minimal_mapping = {k: k for k in ('x', 'y', 'z', 'phot', 'frame_ix')}

default_mapping = dict(
    minimal_mapping, **{k: k for k in ('x_cr', 'y_cr', 'z_cr', 'phot_cr', 'bg_cr',
                                       'x_sig', 'y_sig', 'z_sig', 'phot_sig', 'bg_sig')})

challenge_mapping = {'x': 'xnano',
                     'y': 'ynano',
                     'z': 'znano',
                     'frame_ix': 'frame',
                     'phot': 'intensity ',  # this is really odd, there is a space after intensity
                     'id': 'Ground-truth'}

deepstorm3d_mapping = copy.deepcopy(challenge_mapping)
deepstorm3d_mapping['phot'] = 'intensity'


def get_decode_meta() -> dict:
    return {
        'version': bookkeeping.decode_state()
    }


def save_h5(path: Union[str, pathlib.Path], data: dict, metadata: dict) -> None:

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

        d = f.create_group('decode')
        d.attrs.update(get_decode_meta())

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


def load_h5(path) -> Tuple[dict, dict, dict]:
    """
    Loads a hdf5 file and returns data, metadata and decode meta.

    Returns:
        (dict, dict, dict): Tuple of dicts containing

            - **emitter_data** (*dict*): core emitter data
            - **emitter_meta** (*dict*): emitter meta information
            - **decode_meta** (*dict*): decode meta information

    """
    with h5py.File(path, 'r') as h5:
        data = {
            k: torch.from_numpy(v[:]) for k, v in h5['data'].items() if v.shape is not None
        }
        data.update({  # add the None ones
            k: None for k, v in h5['data'].items() if v.shape is None
        })

        meta_data = dict(h5['meta'].attrs)
        meta_decode = dict(h5['decode'].attrs)

    return data, meta_data, meta_decode


def save_torch(path: Union[str, pathlib.Path], data: dict, metadata: dict):
    torch.save(
        {
            'data': data,
            'meta': metadata,
            'decode': get_decode_meta(),
        },
        path
    )


def load_torch(path) -> Tuple[dict, dict, dict]:
    """
    Loads a torch saved emitterset and returns data, metadata and decode meta.

    Returns:
        (dict, dict, dict): Tuple of dicts containing

            - **emitter_data** (*dict*): core emitter data
            - **emitter_meta** (*dict*): emitter meta information
            - **decode_meta** (*dict*): decode meta information

    """
    out = torch.load(path)
    return out['data'], out['meta'], out['decode']


def save_csv(path: (str, pathlib.Path), data: dict, metadata: dict) -> None:
    def convert_dict_torch_list(data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.tolist()
        return data

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
    data = change_to_one_dim(convert_dict_torch_numpy(data))

    decode_meta_json = json.dumps(get_decode_meta())
    emitter_meta_json = json.dumps(convert_dict_torch_list(metadata))

    assert "\n" not in decode_meta_json, "Failed to dump decode meta string."
    assert "\n" not in emitter_meta_json, "Failed to dump emitter meta string."

    # create file and add metadata to it
    with pathlib.Path(path).open('w+') as f:
        f.write(f"# DECODE EmitterSet\n# {decode_meta_json}\n# {emitter_meta_json}\n")

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path, mode='a', index=False)


def load_csv(path: (str, pathlib.Path), mapping: (None, dict) = default_mapping, skiprows: int = 3,
             line_em_meta: Optional[int] = 2, line_decode_meta: Optional[int] = 1,
             **pd_csv_args) -> Tuple[dict, dict, dict]:
    """
    Loads a CSV file which does provide a header.

    Args:
        path: path to file
        mapping: mapping dictionary with keys at least ('x', 'y', 'z', 'phot', 'id', 'frame_ix')
        skiprows: number of skipped rows before header
        line_em_meta: line ix where metadata of emitters is present (set None for no meta data)
        line_decode_meta: line ix where decode metadata is present(set None for no decode meta)
        pd_csv_args: additional keyword arguments to be parsed to the pandas csv reader

    Returns:
        (dict, dict, dict): Tuple of dicts containing

            - **emitter_data** (*dict*): core emitter data
            - **emitter_meta** (*dict*): emitter meta information
            - **decode_meta** (*dict*): decode meta information

    """

    chunks = pd.read_csv(path, chunksize=100000, skiprows=skiprows, **pd_csv_args)
    data = pd.concat(chunks)

    data_dict = {
        'xyz': torch.stack((torch.from_numpy(data[mapping['x']].to_numpy()).float(),
                            torch.from_numpy(data[mapping['y']].to_numpy()).float(),
                            torch.from_numpy(data[mapping['z']].to_numpy()).float()), 1),
        'phot': torch.from_numpy(data[mapping['phot']].to_numpy()).float(),
        'frame_ix': torch.from_numpy(data[mapping['frame_ix']].to_numpy()).long(),
        'id': None
    }

    if 'id' in mapping.keys():
        data_dict['id'] = torch.from_numpy(data[mapping['id']].to_numpy()).long()

    if 'x_cr' in mapping.keys():
        data_dict['xyz_cr'] = torch.stack((torch.from_numpy(data[mapping['x_cr']].to_numpy()).float(),
                                           torch.from_numpy(data[mapping['y_cr']].to_numpy()).float(),
                                           torch.from_numpy(data[mapping['z_cr']].to_numpy()).float()), 1)

    if 'x_sig' in mapping.keys():
        data_dict['xyz_sig'] = torch.stack((torch.from_numpy(data[mapping['x_sig']].to_numpy()).float(),
                                            torch.from_numpy(data[mapping['y_sig']].to_numpy()).float(),
                                            torch.from_numpy(data[mapping['z_sig']].to_numpy()).float()), 1)

    for k in ('phot_sig', 'bg_sig', 'phot_cr', 'bg_cr'):
        if k in mapping.keys():
            data_dict[k] = torch.from_numpy(data[mapping[k]].to_numpy()).float()

    """Load metadata. For some reason linecache ix is off by one (as compared to my index)."""
    if line_decode_meta is not None:
        decode_meta = json.loads(linecache.getline(str(path), line_decode_meta + 1).strip('# '))
    else:
        decode_meta = None

    if line_em_meta is not None:
        em_meta = json.loads(linecache.getline(str(path), line_em_meta + 1).strip('# '))
    else:
        em_meta = None

    return data_dict, em_meta, decode_meta


def load_smap(path: (str, pathlib.Path), mapping: (dict, None) = None) -> Tuple[dict, dict, dict]:
    """

    Args:
        path: .mat file
        mapping (optional): mapping of matlab fields to emitter. Keys must be x,y,z,phot,frame_ix,bg
        **emitter_kwargs: additional arguments to be parsed to the emitter initialisation

    Returns:
        (dict, dict, dict): Tuple of dicts containing

            - **emitter_data** (*dict*): core emitter data
            - **emitter_meta** (*dict*): emitter meta information
            - **decode_meta** (*dict*): decode meta information

    """
    if mapping is None:
        mapping = {'x': 'xnm', 'y': 'ynm', 'z': 'znm',
                   'phot': 'phot', 'frame_ix': 'frame', 'bg': 'bg'}

    f = h5py.File(path, 'r')

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

    return emitter_dict, None, None


class EmitterWriteStream:
    def __init__(self, name: str, suffix: str, path: pathlib.Path, last_index: str):
        """
        Stream to save emitters when fitting is performed online and in chunks.

        Args:
            name: name of the stream
            suffix: suffix of the file
            path: destination directory
            last_index: either 'including' or 'excluding' (does 0:500 include or exclude index 500).
            While excluding is pythonic, it is not what is common for saved files.
        """
        self._name = name
        self._suffix = suffix
        self._path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        self._last_index = last_index

    def __call__(self, em: EmitterSet, ix_low: int, ix_high: int):
        return self.write(em, ix_low, ix_high)

    def write(self, em: EmitterSet, ix_low: int, ix_high: int):
        """Write emitter chunk to file."""
        if self._last_index == 'including':
            ix = f'_{ix_low}_{ix_high}'
        elif self._last_index == 'excluding':
            ix = f'_{ix_low}_{ix_high - 1}'
        else:
            raise ValueError

        fname = self._path / (self._name + ix + self._suffix)
        em.save(fname)
