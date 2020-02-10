from abc import ABC, abstractmethod  # abstract class
import numpy as np
import h5py
import torch
import scipy.io as sio
from skimage.io import imread

from deepsmlm.generic.emitter import EmitterSet


"""
Interfaces should provide the minimum set of variables that are obligatory to create an emitterset.
"""


class BinaryInterface(ABC):
    """
    Abstract class to specify binary interfaces
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_binary(self):
        raise NotImplementedError

    @abstractmethod
    def save_binary(self):
        raise NotImplementedError


class MatlabInterface:
    """
    Load an SML localisation file from SMAP. Assumes it is stored in the modern -v7.3 format (i.e. a HDF5).
    """

    x_key = 'xnm'
    y_key = 'ynm'
    z_key = 'znm'
    phot_key = 'phot'
    frame_key = 'frame'
    bg_key = 'bg'

    xy_unit = 'nm'

    def __init__(self, frame_shift=-1, axis=[1, 0, 2]):
        """
        Specifies some transformations

        Args:
            frame_shift: transform frame index. Usually -1 because MATLAB counts starting at 1, Python 0
            axis: swap axis

        """

        self.frame_shift = frame_shift
        self.axis_trafo = axis
        self._cache_sml = None

    def load_binary(self, mat_file):
        """
        Loads the savelocs struct from Matlab. Applies transformations as specified

        Args:
            mat_file:

        Returns:
            (EmitterSet)
        """
        f = h5py.File(mat_file, 'r')
        self._cache_sml = f

        loc_dict = f['saveloc']['loc']
        xyz = torch.cat([
            torch.from_numpy(np.array(loc_dict[self.x_key])).permute(1, 0),
            torch.from_numpy(np.array(loc_dict[self.y_key])).permute(1, 0),
            torch.from_numpy(np.array(loc_dict[self.z_key])).permute(1, 0)
        ], 1)

        em = EmitterSet(xyz=xyz,
                        phot=torch.from_numpy(np.array(loc_dict[self.phot_key])).squeeze(),
                        frame_ix=torch.from_numpy(np.array(loc_dict[self.frame_key])).squeeze(),
                        bg=torch.from_numpy(np.array(loc_dict[self.bg_key])).squeeze(),
                        xy_unit=self.xy_unit)

        em.convert_em_(axis=self.axis_trafo)
        em.frame_ix += self.frame_shift

        return em


class NumpyInterface(BinaryInterface):

    def __init__(self,
                 xyz_key='xyz', phot_key='phot', fix_key='frame_ix',
                 extent_key='extent', frame_key='frames', id_key='id', unsupervised=False):
        super().__init__(unsupervised)

        self.xyz_key = xyz_key
        self.phot_key = phot_key
        self.fix_key = fix_key
        self.id_key = id_key

        self.extent_key = extent_key
        self.frame_key = frame_key

    def load_binary(self, mat_file):

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        bin = np.load(mat_file)

        emitter_set = EmitterSet(xyz=torch.from_numpy(bin[self.xyz_key]),
                                 phot=torch.from_numpy(bin[self.phot_key]),
                                 frame_ix=torch.from_numpy(bin[self.fix_key]),
                                 id=torch.from_numpy(bin[self.id_key]))

        extent = totuple(bin[self.extent_key])
        frames = torch.from_numpy(bin[self.frame_key]).type(torch.FloatTensor)

        return emitter_set, extent, frames

    def save_binary(self, emitter_set, mat_file):
        raise NotImplementedError('Not Implemented.')
