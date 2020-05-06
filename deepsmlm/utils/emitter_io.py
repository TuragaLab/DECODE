from abc import ABC  # abstract class

import numpy as np
import torch

import deepsmlm

"""
Interfaces should provide the minimum set of variables that are obligatory to create an emitterset.
"""


class BinaryInterface(ABC):
    """
    Abstract class to specify binary interfaces
    """

    def __init__(self):
        super().__init__()

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class MatlabInterface(BinaryInterface):
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
        super().__init__()

        self.frame_shift = frame_shift
        self.axis_trafo = axis
        self._cache_sml = None

    def load(self, mat_file):
        """
        Loads the savelocs struct from Matlab. Applies transformations as specified

        Args:
            mat_file:

        Returns:
            (EmitterSet)
        """
        import h5py

        f = h5py.File(mat_file, 'r')
        self._cache_sml = f

        loc_dict = f['saveloc']['loc']
        xyz = torch.cat([
            torch.from_numpy(np.array(loc_dict[self.x_key])).permute(1, 0),
            torch.from_numpy(np.array(loc_dict[self.y_key])).permute(1, 0),
            torch.from_numpy(np.array(loc_dict[self.z_key])).permute(1, 0)
        ], 1)

        em = deepsmlm.EmitterSet(xyz=xyz,
                                 phot=torch.from_numpy(np.array(loc_dict[self.phot_key])).squeeze(),
                                 frame_ix=torch.from_numpy(np.array(loc_dict[self.frame_key])).squeeze(),
                                 bg=torch.from_numpy(np.array(loc_dict[self.bg_key])).squeeze(),
                                 xy_unit=self.xy_unit)

        em.convert_em_(axis=self.axis_trafo)
        em.frame_ix += self.frame_shift

        return em
