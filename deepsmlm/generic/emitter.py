from __future__ import annotations  # otherwise 'self' annotations does not work

import warnings
from deprecated import deprecated
from typing import Union

import numpy as np
import torch
from pathlib import Path

from . import slicing as gutil, test_utils as tutil


class EmitterSet:
    """
    Class, storing a set of emitters and its attributes. Probably the most commonly used class of this framework.

    Attributes:
            xyz: Coordinates of size N x [2,3].
            phot: Photon count of size N
            frame_ix: size N. Index on which the emitter appears.
            id: size N. Identity the emitter.
            prob: size N. Probability estimate of the emitter.
            bg: size N. Background estimate of emitter.
            xyz_cr: size N x 3. Cramer-Rao estimate of the emitters position.
            phot_cr: size N. Cramer-Rao estimate of the emitters photon count.
            bg_cr: size N. Cramer-Rao estimate of the emitters background value.
            sanity_check: performs a sanity check if true.
            xy_unit: Unit of the x and y coordinate.
            px_size: Pixel size for unit conversion. If not specified, derived attributes (xyz_px and xyz_nm)
                can not be accessed
    """
    _eq_precision = 1E-8
    _xy_units = ('px', 'nm')

    def __init__(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.LongTensor,
                 id: torch.LongTensor = None, prob: torch.Tensor = None, bg: torch.Tensor = None,
                 xyz_cr: torch.Tensor = None, phot_cr: torch.Tensor = None, bg_cr: torch.Tensor = None,
                 sanity_check: bool = True, xy_unit: str = None, px_size: Union[tuple, torch.Tensor] = None):
        """
        Initialises EmitterSet of :math:`N` emitters.

        Args:
            xyz: Coordinates of size :math:`(N,3)`
            phot: Photon count of size :math:`N`
            frame_ix: Index on which the emitter appears. Must be integer type. Size :math:`N`
            id: Identity the emitter. Must be tensor integer type and the same type as frame_ix. Size :math:`N`
            prob: Probability estimate of the emitter. Size :math:`N`
            bg: Background estimate of emitter. Size :math:`N`
            xyz_cr: Cramer-Rao estimate of the emitters position. Size :math:`(N,3)`
            phot_cr: Cramer-Rao estimate of the emitters photon count. Size :math:`N`
            bg_cr: Cramer-Rao estimate of the emitters background value. Size :math:`N`
            sanity_check: performs a sanity check.
            xy_unit: Unit of the x and y coordinate.
            px_size: Pixel size for unit conversion. If not specified, derived attributes (xyz_px and xyz_nm)
                may not be accessed because one can not convert units without pixel size.
        """

        self.xyz = None
        self.phot = None
        self.frame_ix = None
        self.id = None
        self.prob = None
        self.bg = None

        # Cramer-Rao values
        self.xyz_cr = None
        self.phot_cr = None
        self.bg_cr = None

        self._set_typed(xyz=xyz, phot=phot, frame_ix=frame_ix, id=id, prob=prob, bg=bg,
                        xyz_cr=xyz_cr, phot_cr=phot_cr, bg_cr=bg_cr)

        self._sorted = False
        # get at least one_dim tensors
        at_least_one_dim(self.xyz,
                         self.phot,
                         self.frame_ix,
                         self.id,
                         self.prob,
                         self.bg,
                         self.xyz_cr,
                         self.phot_cr,
                         self.bg_cr)

        self.xy_unit = xy_unit
        self.px_size = px_size
        if self.px_size is not None:
            if not isinstance(self.px_size, torch.Tensor):
                self.px_size = torch.Tensor(self.px_size)

        if sanity_check:
            self._sanity_check()

    def _set_typed(self, xyz, phot, frame_ix, id, prob, bg, xyz_cr, phot_cr, bg_cr):
        """
        Sets the attributes in the correct type and with default argument if None
        """

        if xyz.dtype not in (torch.float, torch.double, torch.half):
            raise ValueError("XYZ coordinates must be float type.")
        else:
            f_type = xyz.dtype

        if frame_ix.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError(f"Frame index must be integer type and not {frame_ix.dtype}")

        if id is not None and (id.dtype not in (torch.int16, torch.int32, torch.int64) or id.dtype != frame_ix.dtype):
            raise ValueError(f"ID must be None or integer type and the same as frame_ix dtype and not {id.dtype}")
        else:
            i_type = frame_ix.dtype

        xyz = xyz if xyz.shape[1] == 3 else torch.cat((xyz, torch.zeros_like(xyz[:, [0]])), 1)

        num_input = int(xyz.shape[0]) if xyz.shape[0] != 0 else 0

        """Set values"""
        if num_input != 0:
            self.xyz = xyz
            self.phot = phot.type(f_type)
            self.frame_ix = frame_ix

            # Optionals
            self.id = id if id is not None else -torch.ones_like(frame_ix)
            self.prob = prob if prob is not None else torch.ones_like(frame_ix).type(f_type)
            self.bg = bg if bg is not None else float('nan') * torch.ones_like(frame_ix).type(f_type)
            self.xyz_cr = xyz_cr if xyz_cr is not None else float('nan') * torch.ones_like(self.xyz)
            self.phot_cr = phot_cr if phot_cr is not None else float('nan') * torch.ones_like(self.phot)
            self.bg_cr = bg_cr if bg_cr is not None else float('nan') * torch.ones_like(self.bg)

        else:
            self.xyz = torch.zeros((0, 3)).type(f_type)
            self.phot = torch.zeros((0,)).type(f_type)
            self.frame_ix = torch.zeros((0,)).type(i_type)

            # Optionals
            self.id = -torch.ones((0,)).type(i_type)
            self.prob = torch.ones((0,)).type(f_type)
            self.bg = float('nan') * torch.ones_like(self.prob)
            self.xyz_cr = float('nan') * torch.ones((0, 3)).type(f_type)
            self.phot_cr = float('nan') * torch.ones_like(self.prob)
            self.bg_cr = float('nan') * torch.ones_like(self.prob)

    def to_dict(self):
        """
        Returns dictionary representation of this EmitterSet so that the keys and variables correspond to what an
        EmitterSet would be initialised.

        Returns:
            (dict)
        """
        em_dict = {
            'xyz': self.xyz,
            'phot': self.phot,
            'frame_ix': self.frame_ix,
            'id': self.id,
            'prob': self.prob,
            'bg': self.bg,
            'xyz_cr': self.xyz_cr,
            'phot_cr': self.phot_cr,
            'bg_cr': self.bg_cr,
            'xy_unit': self.xy_unit,
            'px_size': self.px_size
        }

        return em_dict

    def save(self, file: Union[str, Path]):
        """
        Pickle save's the dictionary of this instance. No legacy guarantees given.
        Should only be used for short-term storage.

        Args:
            file: path where to save

        Returns:

        """

        if not isinstance(file, Path):
            file = Path(file)

        em_dict = self.to_dict()
        torch.save(em_dict, file)

    @staticmethod
    def load(file: Union[str, Path]):
        """
        Loads the set of emitters which was saved by the 'save' method.

        Args:
            file: path to the emitterset

        Returns:
            EmitterSet

        """

        em_dict = torch.load(file)
        return EmitterSet(**em_dict)

    @property
    def xyz_px(self):
        """
        Returns xyz in pixel coordinates and performs respective transformations if needed.
        """
        return self._pxnm_conversion(self.xyz, in_unit=self.xy_unit, tar_unit='px')

    @xyz_px.setter
    def xyz_px(self, xyz):
        self.xyz = xyz
        self.xy_unit = 'px'

    @property
    def xyz_nm(self):
        """
        Returns xyz in nanometres and performs respective transformations if needed.
        """
        return self._pxnm_conversion(self.xyz, in_unit=self.xy_unit, tar_unit='nm')

    @xyz_nm.setter
    def xyz_nm(self, xyz):  # xyz in nanometres
        self.xyz = xyz
        self.xy_unit = 'nm'

    @property
    def xyz_scr(self):  # sqrt cramer-rao of xyz
        return self.xyz_cr.sqrt()

    @property
    def xyz_cr_px(self):
        return self._pxnm_conversion(self.xyz_cr, in_unit=self.xy_unit, tar_unit='px', power=2)

    @property
    def xyz_scr_px(self):
        return self.xyz_cr_px.sqrt()

    @property
    def xyz_cr_nm(self):
        return self._pxnm_conversion(self.xyz_cr, in_unit=self.xy_unit, tar_unit='nm', power=2)

    @property
    def xyz_scr_nm(self):
        return self.xyz_cr_nm.sqrt()

    @property
    def phot_scr(self):  # sqrt cramer-rao of photon count
        return self.phot_cr.sqrt()

    @property
    def bg_scr(self):  # sqrt cramer-rao of bg count
        return self.bg_cr.sqrt()

    def _inplace_replace(self, em):
        """
        Inplace replacement of this self instance. Does not work for inherited methods ...
        Args:
            em: (EmitterSet) that should replace self

        Returns:
            (None)
        """
        self.__init__(xyz=em.xyz,
                      phot=em.phot,
                      frame_ix=em.frame_ix,
                      id=em.id,
                      prob=em.prob,
                      bg=em.bg,
                      xyz_cr=em.xyz_cr,
                      phot_cr=em.phot_cr,
                      bg_cr=em.bg_cr,
                      sanity_check=True,
                      xy_unit=em.xy_unit,
                      px_size=em.px_size)

    def _sanity_check(self, check_uniqueness=False):
        """
        Performs several integrity tests on the EmitterSet.

        Args:
            check_uniqueness: (bool) check the uniqueness of the ID

        Returns:
            (bool) sane or not sane
        """
        if not same_shape_tensor(0, self.xyz, self.phot, self.frame_ix, self.id, self.bg,
                                 self.xyz_cr, self.phot_cr, self.bg_cr):
            raise ValueError("Coordinates, photons, frame ix, id and prob are not of equal shape in 0th dimension.")

        if not same_dim_tensor(torch.ones(1), self.phot, self.prob, self.frame_ix, self.id):
            raise ValueError("Expected photons, probability frame index and id to be 1D.")

        # Motivate the user to specify an xyz unit.
        if len(self) > 0:
            if self.xy_unit is None:
                warnings.warn("No xyz unit specified. No guarantees given ...")
            else:
                if self.xy_unit not in self._xy_units:
                    raise ValueError("XY unit not supported.")

        # check uniqueness of identity (ID)
        if check_uniqueness:
            if torch.unique(self.id).numel() != self.id.numel():
                raise ValueError("IDs are not unique.")

        return True

    def __len__(self):
        """
        Implements length of EmitterSet. Length of EmitterSet is number of rows of xyz container.

        Returns:
            (int) length of EmitterSet
        """
        return int(self.xyz.shape[0]) if self.xyz.shape[0] != 0 else 0

    def __str__(self):
        """
        Friendly representation of EmitterSet

        Returns:
            (string) representation of this class
        """
        print_str = f"EmitterSet" \
                    f"\n::num emitters: {len(self)}"

        if len(self) == 0:
            print_str += "\n::frame range: n.a." \
                         "\n::spanned volume: n.a."
        else:
            print_str += f"\n::xy unit: {self.xy_unit}"
            print_str += f"\n::frame range: {self.frame_ix.min().item()} - {self.frame_ix.max().item()}" \
                         f"\n::spanned volume: {self.xyz.min(0)[0].numpy()} - {self.xyz.max(0)[0].numpy()}"
        return print_str

    def __eq__(self, other):
        """
        Implements equalness check. Returns true if all attributes are the same and in the same order. The identy
        does not have to be the same, but the values of the attributes have to.
        Args:
            other: (emitterset)

        Returns:
            (bool) true if as stated above.
        """
        if not tutil.tens_almeq(self.xyz, other.xyz, self._eq_precision):
            return False

        if not tutil.tens_almeq(self.frame_ix, other.frame_ix, self._eq_precision):
            return False

        if not tutil.tens_almeq(self.phot, other.phot, self._eq_precision):
            return False

        if not tutil.tens_almeq(self.prob, other.prob, self._eq_precision):
            return False

        if not self.xy_unit == other.xy_unit:
            return False

        if torch.isnan(self.bg).all():
            if not torch.isnan(other.bg).all():
                return False
        else:
            if not tutil.tens_almeq(self.bg, other.bg, self._eq_precision):
                return False

        if torch.isnan(self.xyz_cr).all():
            if not torch.isnan(other.xyz_cr).all():
                return False
        else:
            if not tutil.tens_almeq(self.bg, other.bg, self._eq_precision):
                return False

        if torch.isnan(self.phot_cr).all():
            if not torch.isnan(other.phot_cr).all():
                return False
        else:
            if not tutil.tens_almeq(self.bg, other.bg, self._eq_precision):
                return False

        if torch.isnan(self.bg_cr).all():
            if not torch.isnan(other.bg_cr).all():
                return False
        else:
            if not tutil.tens_almeq(self.bg, other.bg, self._eq_precision):
                return False

        self.eq_attr(other)
        return True

    def eq_attr(self, other):
        """
        Tests whether the meta data attributes are the same

        Args:
            other: EmitterSet

        Returns:
            (bool)
        """
        if self.px_size is None:
            if other.px_size is None:
                return True
            else:
                return False

        if not (self.px_size == other.px_size).all():
            return False

        if not self.xy_unit == other.xy_unit:
            return False

        return True

    def __iter__(self):
        """
        Implements iterator bookkeeping.

        Returns:
            (self)
        """
        self.n = 0
        return self

    def __next__(self):
        """
        Implements next element in iterator method

        Returns:
            (EmitterSet) next element of iterator

        """
        if self.n <= len(self) - 1:
            self.n += 1
            return self.get_subset(self.n - 1)
        else:
            raise StopIteration

    def __getitem__(self, item):
        """
        Implements array indexing for this class.

        Args:
            item: (int), or indexing

        Returns:
            (EmitterSet)
        """

        if isinstance(item, int) and item >= len(self):
            raise IndexError(f"Index {item} out of bounds of EmitterSet of size {len(self)}")

        return self.get_subset(item)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def clone(self):
        """
        Returns a deep copy of this EmitterSet.

        Returns:
            (EmitterSet)
        """
        return EmitterSet(self.xyz.clone(),
                          self.phot.clone(),
                          self.frame_ix.clone(),
                          self.id.clone(),
                          self.prob.clone(),
                          self.bg.clone(),
                          self.xyz_cr.clone(),
                          self.phot_cr.clone(),
                          self.bg_cr.clone(),
                          sanity_check=False,
                          xy_unit=self.xy_unit,
                          px_size=self.px_size)

    @staticmethod
    def cat(emittersets, remap_frame_ix=None, step_frame_ix: int = None):
        """
        Concatenate multiple emittersets into one emitterset which is returned. Optionally modify the frame indices by
        the arguments.

        Args:
            emittersets: (list of emittersets) emittersets to be concatenated
            remap_frame_ix: (torch.Tensor, optional) index of 0th frame to map the corresponding emitterset to.
            Length must correspond to length of list in first argument.
            step_frame_ix: (int, optional) step size of 0th frame between emittersets.

        Returns:
            (emitterset) concatenated emitterset
        """
        num_emittersets = len(emittersets)

        if remap_frame_ix is not None and step_frame_ix is not None:
            raise ValueError("You cannot specify remap frame ix and step frame ix at the same time.")
        elif remap_frame_ix is not None:
            shift = remap_frame_ix.clone()
        elif step_frame_ix is not None:
            shift = torch.arange(0, num_emittersets) * step_frame_ix
        else:
            shift = torch.zeros(num_emittersets).int()

        total_num_emitter = 0
        for i in range(num_emittersets):
            total_num_emitter += len(emittersets[i])

        xyz = torch.cat([emittersets[i].xyz for i in range(num_emittersets)], 0)
        phot = torch.cat([emittersets[i].phot for i in range(num_emittersets)], 0)
        frame_ix = torch.cat([emittersets[i].frame_ix + shift[i] for i in range(num_emittersets)], 0)
        id = torch.cat([emittersets[i].id for i in range(num_emittersets)], 0)
        prob = torch.cat([emittersets[i].prob for i in range(num_emittersets)], 0)
        bg = torch.cat([emittersets[i].bg for i in range(num_emittersets)], 0)
        xyz_cr = torch.cat([emittersets[i].xyz_cr for i in range(num_emittersets)], 0)
        phot_cr = torch.cat([emittersets[i].phot_cr for i in range(num_emittersets)], 0)
        bg_cr = torch.cat([emittersets[i].bg_cr for i in range(num_emittersets)], 0)

        # px_size and xy unit is taken from the first element that is not None
        xy_unit = None
        px_size = None
        for i in range(num_emittersets):
            if emittersets[i].xy_unit is not None:
                xy_unit = emittersets[i].xy_unit
                break
        for i in range(num_emittersets):
            if emittersets[i].px_size is not None:
                px_size = emittersets[i].px_size
                break

        return EmitterSet(xyz, phot, frame_ix, id, prob, bg, xyz_cr, phot_cr, bg_cr, sanity_check=True,
                          xy_unit=xy_unit, px_size=px_size)

    def sort_by_frame_(self):
        """
        Inplace sort this emitterset by its frame index.

        Returns:

        """
        self.frame_ix, ix = self.frame_ix.sort()
        self.xyz = self.xyz[ix, :]
        self.phot = self.phot[ix]
        self.id = self.id[ix]
        self.prob = self.prob[ix]
        self.bg = self.bg[ix]
        self.xyz_cr = self.xyz_cr[ix]
        self.phot_cr = self.phot_cr[ix]
        self.bg_cr = self.bg_cr[ix]

        self._sorted = True

    def sort_by_frame(self):
        """
        Sort a deepcopy of this emitterset and return it.

        Returns:
            (emitterset) Sorted copy of this emitterset.

        """
        em = self.clone()
        em.sort_by_frame_()

        return em

    def get_subset(self, ix):
        """
        Returns subset of emitterset. Implementation of __getitem__ and __next__ methods.
        Args:
            ix: (int, list) integer index or list of indices

        Returns:
            (EmitterSet)
        """
        if isinstance(ix, int):
            ix = [ix]

        # PyTorch single element support
        if not isinstance(ix, torch.BoolTensor) and isinstance(ix, torch.Tensor) and ix.numel() == 1:
            ix = [int(ix)]

        # Todo: Check for numpy boolean array
        if isinstance(ix, (np.ndarray, np.generic)) and ix.size == 1:  # numpy support
            ix = [int(ix)]

        return EmitterSet(self.xyz[ix, :], self.phot[ix], self.frame_ix[ix], self.id[ix], self.prob[ix], self.bg[ix],
                          self.xyz_cr[ix], self.phot_cr[ix], self.bg_cr[ix], sanity_check=False,
                          xy_unit=self.xy_unit, px_size=self.px_size)

    def get_subset_frame(self, frame_start, frame_end, frame_ix_shift=None):
        """
        Returns emitters that are in the frame range as specified.

        Args:
            frame_start: (int) lower frame index limit
            frame_end: (int) upper frame index limit (including)
            frame_ix_shift:

        Returns:

        """

        ix = (self.frame_ix >= frame_start) * (self.frame_ix <= frame_end)
        em = self[ix]

        if not frame_ix_shift:
            return em
        elif len(em) != 0:  # only shift if there is actually something
            em.frame_ix += frame_ix_shift

        return em

    @property
    def single_frame(self):
        """
        Check if all emitters belong to the same frame.

        Returns:
            (bool)
        """
        return True if torch.unique(self.frame_ix).shape[0] == 1 else False

    @deprecated(reason="Needs to be debugged.")
    def chunks(self, n: int):
        """
        Splits the EmitterSet into (almost) equal chunks

        Args:
            n (int): number of splits

        Returns:
            list: of emittersets

        """
        from itertools import islice, chain

        def chunky(iterable, size=10):
            iterator = iter(iterable)
            for first in iterator:
                yield chain([first], islice(iterator, size - 1))

        return chunky(self, n)

    def split_in_frames(self, ix_low: int = 0, ix_up: int = None):
        """
        Splits a set of emitters in a list of emittersets based on their respective frame index.

        Args:
            ix_low: (int, 0) lower bound
            ix_up: (int, None) upper bound

        Returns:

        """

        """The first frame is assumed to be 0. If it's negative go to the lowest negative."""
        ix_low = ix_low if ix_low is not None else self.frame_ix.min().item()
        ix_up = ix_up if ix_up is not None else self.frame_ix.max().item()

        return gutil.split_sliceable(x=self, x_ix=self.frame_ix, ix_low=ix_low, ix_high=ix_up)

    def _pxnm_conversion(self, xyz, in_unit, tar_unit, power: float = 1.):

        if in_unit is None:
            raise ValueError("Conversion not possible if unit not specified.")

        if in_unit == tar_unit:
            return xyz

        elif in_unit == 'nm' and tar_unit == 'px':
            """px check needs to happen here, because in _convert_coordinates, factor is an optional argument."""
            if self.px_size is None:
                raise ValueError("Conversion not possible if px size is not specified.")

            return self._convert_coordinates(factor=1/self.px_size ** power, xyz=xyz)

        elif in_unit == 'px' and tar_unit == 'nm':
            if self.px_size is None:
                raise ValueError("Conversion not possible if px size is not specified.")

            return self._convert_coordinates(factor=self.px_size ** power, xyz=xyz)

        else:
            raise ValueError("Unsupported conversion.")

    def _convert_coordinates(self, factor=None, shift=None, axis=None, xyz=None):
        """
        Convert coordinates. Order: factor -> shift -> axis

        Args:
            factor: (torch.Tensor, None)
            shift: (torch.Tensor, None)
            axis: (list)
            xyz (torch.Tensor, None): use different coordinates (not self.xyz)

        Returns:
            xyz (torch.Tensor) modified coordinates.

        """
        if xyz is None:
            xyz = self.xyz.clone()

        if factor is not None:
            if factor.size(0) == 2:
                factor = torch.cat((factor, torch.tensor([1.])), 0)

            xyz = xyz * factor.float().unsqueeze(0)

        if shift is not None:
            xyz += shift.float().unsqueeze(0)

        if axis is not None:
            xyz = xyz[:, axis]

        return xyz

    def populate_crlb(self, psf, **kwargs):
        """
        Populate the CRLB values by the PSF function.

        Args:
            psf (PSF): Point Spread function with CRLB implementation
            **kwargs: additional arguments to be parsed to the CRLB method

        Returns:

        """

        crlb, _ = psf.crlb(self.xyz, self.phot, self.bg, **kwargs)
        self.xyz_cr = crlb[:, :3]
        self.phot_cr = crlb[:, 3]
        self.bg_cr = crlb[:, 4]


class RandomEmitterSet(EmitterSet):
    """
    A helper calss when we only want to provide a number of emitters.
    """

    def __init__(self, num_emitters: int, extent: float = 32, xy_unit: str = 'px'):
        xyz = torch.rand((num_emitters, 3)) * extent
        super().__init__(xyz, torch.ones_like(xyz[:, 0]), torch.zeros_like(xyz[:, 0]).long(), xy_unit=xy_unit)

    def _inplace_replace(self, em):
        super().__init__(xyz=em.xyz,
                         phot=em.phot,
                         frame_ix=em.frame_ix,
                         id=em.id,
                         prob=em.prob,
                         bg=em.bg,
                         xyz_cr=em.xyz_cr,
                         phot_cr=em.phot_cr,
                         bg_cr=em.bg_cr,
                         sanity_check=False,
                         xy_unit=em.xy_unit,
                         px_size=em.px_size)


class CoordinateOnlyEmitter(EmitterSet):
    """
    A helper class when we only want to provide xyz, but not photons and frame_ix.
    Useful for testing. Photons will be tensor of 1, frame_ix tensor of 0.
    """

    def __init__(self, xyz: torch.Tensor, xy_unit: str = None, px_size=None):
        """

        :param xyz: (torch.tensor) N x 2, N x 3
        """
        super().__init__(xyz, torch.ones_like(xyz[:, 0]), torch.zeros_like(xyz[:, 0]).int(),
                         xy_unit=xy_unit, px_size=px_size)

    def _inplace_replace(self, em):
        super().__init__(xyz=em.xyz,
                         phot=em.phot,
                         frame_ix=em.frame_ix,
                         id=em.id,
                         prob=em.prob,
                         bg=em.bg,
                         xyz_cr=em.xyz_cr,
                         phot_cr=em.phot_cr,
                         bg_cr=em.bg_cr,
                         sanity_check=False,
                         xy_unit=em.xy_unit,
                         px_size=em.px_size)


class EmptyEmitterSet(CoordinateOnlyEmitter):
    """An empty emitter set."""

    def __init__(self, xy_unit=None, px_size=None):
        super().__init__(torch.zeros((0, 3)), xy_unit=xy_unit, px_size=px_size)

    def _inplace_replace(self, em):
        super().__init__(xyz=em.xyz,
                         phot=em.phot,
                         frame_ix=em.frame_ix,
                         id=em.id,
                         prob=em.prob,
                         bg=em.bg,
                         xyz_cr=em.xyz_cr,
                         phot_cr=em.phot_cr,
                         bg_cr=em.bg_cr,
                         sanity_check=False,
                         xy_unit=em.xy_unit,
                         px_size=em.px_size)


class LooseEmitterSet:
    """
    Related to the standard EmitterSet. However, here we do not specify a frame_ix but rather a (non-integer)
    initial point in time where the emitter starts to blink and an on-time.

    Attributes:
        xyz (torch.Tensor): coordinates. Dimension: N x 3
        intensity (torch.Tensor): intensity, i.e. photon flux per time unit. Dimension N
        id (torch.Tensor, int): identity of the emitter. Dimension: N
        t0 (torch.Tensor, float): initial blink event. Dimension: N
        ontime (torch.Tensor): duration in frame-time units how long the emitter blinks. Dimension N
        xy_unit (string): unit of the coordinates
    """

    def __init__(self, xyz: torch.Tensor, intensity: torch.Tensor, ontime: torch.Tensor, t0: torch.Tensor,
                 xy_unit: str, px_size, id: torch.Tensor = None, sanity_check=True):
        """

        Args:
            xyz (torch.Tensor): coordinates. Dimension: N x 3
            intensity (torch.Tensor): intensity, i.e. photon flux per time unit. Dimension N
            t0 (torch.Tensor, float): initial blink event. Dimension: N
            ontime (torch.Tensor): duration in frame-time units how long the emitter blinks. Dimension N
            id (torch.Tensor, int, optional): identity of the emitter. Dimension: N
            xy_unit (string): unit of the coordinates
        """

        """If no ID specified, give them one."""
        if id is None:
            id = torch.arange(xyz.shape[0])

        self.xyz = xyz
        self.xy_unit = xy_unit
        self.px_size = px_size
        self._phot = None
        self.intensity = intensity
        self.id = id
        self.t0 = t0
        self.ontime = ontime

        if sanity_check:
            self.sanity_check()

    def sanity_check(self):

        """Check IDs"""
        if self.id.unique().numel() != self.id.numel():
            raise ValueError("IDs are not unique.")

        """Check xyz"""
        if self.xyz.dim() != 2 or self.xyz.size(1) != 3:
            raise ValueError("Wrong xyz dimension.")

        """Check intensity"""
        if (self.intensity < 0).any():
            raise ValueError("Negative intensity values encountered.")

        """Check timings"""
        if (self.ontime < 0).any():
            raise ValueError("Negative ontime encountered.")

    @property
    def te(self):  # end time
        return self.t0 + self.ontime

    def _distribute_framewise(self):
        """
        Distributes the emitters framewise and prepares them for EmitterSet format.

        Returns:
            xyz_ (torch.Tensor): coordinates
            phot_ (torch.Tensor): photon count
            frame_ (torch.Tensor): frame indices (the actual distribution)
            id_ (torch.Tensor): identities

        """

        def cum_count_per_group(arr):
            """
            Helper function that returns the cumulative sum per group.

            Example:
                [0, 0, 0, 1, 2, 2, 0] --> [0, 1, 2, 0, 0, 1, 3]
            """

            def grp_range(counts: torch.Tensor):
                assert counts.dim() == 1

                idx = counts.cumsum(0)
                id_arr = torch.ones(idx[-1], dtype=int)
                id_arr[0] = 0
                id_arr[idx[:-1]] = -counts[:-1] + 1
                return id_arr.cumsum(0)

            if arr.numel() == 0:
                return arr

            _, cnt = torch.unique(arr, return_counts=True)
            return grp_range(cnt)[torch.argsort(arr).argsort()]

        frame_start = torch.floor(self.t0).long()
        frame_last = torch.floor(self.te).long()
        frame_count = (frame_last - frame_start).long()

        frame_count_full = frame_count - 2
        ontime_first = torch.min(self.te - self.t0, frame_start + 1 - self.t0)
        ontime_last = torch.min(self.te - self.t0, self.te - frame_last)

        """Repeat by full-frame duration"""

        # kick out everything that has no full frame_duration
        ix_full = frame_count_full >= 0
        xyz_ = self.xyz[ix_full, :]
        flux_ = self.intensity[ix_full]
        id_ = self.id[ix_full]
        frame_start_full = frame_start[ix_full]
        frame_dur_full_clean = frame_count_full[ix_full]

        xyz_ = xyz_.repeat_interleave(frame_dur_full_clean + 1, dim=0)
        phot_ = flux_.repeat_interleave(frame_dur_full_clean + 1, dim=0)  # because intensity * 1 = phot
        id_ = id_.repeat_interleave(frame_dur_full_clean + 1, dim=0)
        # because 0 is first occurence
        frame_ix_ = frame_start_full.repeat_interleave(frame_dur_full_clean + 1, dim=0) + cum_count_per_group(id_) + 1

        """First frame"""
        # first
        xyz_ = torch.cat((xyz_, self.xyz), 0)
        phot_ = torch.cat((phot_, self.intensity * ontime_first), 0)
        id_ = torch.cat((id_, self.id), 0)
        frame_ix_ = torch.cat((frame_ix_, frame_start), 0)

        # last (only if frame_last != frame_first
        ix_with_last = frame_last >= frame_start + 1
        xyz_ = torch.cat((xyz_, self.xyz[ix_with_last]))
        phot_ = torch.cat((phot_, self.intensity[ix_with_last] * ontime_last[ix_with_last]), 0)
        id_ = torch.cat((id_, self.id[ix_with_last]), 0)
        frame_ix_ = torch.cat((frame_ix_, frame_last[ix_with_last]))

        return xyz_, phot_, frame_ix_, id_

    def return_emitterset(self):
        """
        Returns EmitterSet with distributed emitters. The ID is preserved such that localisations coming from the same
        fluorophore will have the same ID.

        Returns:
            EmitterSet
        """

        xyz_, phot_, frame_ix_, id_ = self._distribute_framewise()
        return EmitterSet(xyz_, phot_, frame_ix_.int(), id_.int(), xy_unit=self.xy_unit, px_size=self.px_size)


def at_least_one_dim(*args):
    for arg in args:
        if arg.dim() == 0:
            arg.unsqueeze_(0)


def same_shape_tensor(dim, *args):
    for i in range(args.__len__() - 1):
        if args[i].size(dim) == args[i + 1].size(dim):
            continue
        else:
            return False

    return True


def same_dim_tensor(*args):
    for i in range(args.__len__() - 1):
        if args[i].dim() == args[i + 1].dim():
            continue
        else:
            return False

    return True