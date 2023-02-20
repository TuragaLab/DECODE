import copy
import warnings
from pathlib import Path
from typing import Union, Optional, Iterable

import numpy as np
import torch

import decode.generic.utils
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
    _power_auto_conversion_attrs = {'xyz_cr': 2, 'xyz_sig': 1}
    _xy_units = ('px', 'nm')

    def __init__(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.LongTensor,
                 id: torch.LongTensor = None, prob: torch.Tensor = None, bg: torch.Tensor = None,
                 xyz_cr: torch.Tensor = None, phot_cr: torch.Tensor = None, bg_cr: torch.Tensor = None,
                 xyz_sig: torch.Tensor = None, phot_sig: torch.Tensor = None, bg_sig: torch.Tensor = None,
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
            xyz_sig: Error estimate of the emitters position. Size :math:`(N,3)`
            phot_sig: Error estimate of the photon count. Size :math:`N`
            bg_sig: Error estimate of the background value. Size :math:`N`
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

        # Error estimates
        self.xyz_sig = None
        self.phot_sig = None
        self.bg_sig = None

        self._set_typed(xyz=xyz, phot=phot, frame_ix=frame_ix, id=id, prob=prob, bg=bg,
                        xyz_cr=xyz_cr, phot_cr=phot_cr, bg_cr=bg_cr,
                        xyz_sig=xyz_sig, phot_sig=phot_sig, bg_sig=bg_sig)

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

    @property
    def xyz_px(self) -> torch.Tensor:
        """
        Returns xyz in pixel coordinates and performs respective transformations if needed.
        """
        return self._pxnm_conversion(self.xyz, in_unit=self.xy_unit, tar_unit='px', power=1.)

    @xyz_px.setter
    def xyz_px(self, xyz):
        self.xyz = xyz
        self.xy_unit = 'px'

    @property
    def xyz_nm(self) -> torch.Tensor:
        """
        Returns xyz in nanometres and performs respective transformations if needed.
        """
        return self._pxnm_conversion(self.xyz, in_unit=self.xy_unit, tar_unit='nm', power=1.)

    @xyz_nm.setter
    def xyz_nm(self, xyz):  # xyz in nanometres
        self.xyz = xyz
        self.xy_unit = 'nm'

    @property
    def xyz_scr(self) -> torch.Tensor:
        """
        Square-Root cramer rao of xyz.
        """
        return self.xyz_cr.sqrt()

    @property
    def xyz_scr_px(self) -> torch.Tensor:
        """
        Square-Root cramer rao of xyz in px units.
        """
        return self.xyz_cr_px.sqrt()

    @property
    def xyz_scr_nm(self) -> torch.Tensor:
        return self.xyz_cr_nm.sqrt()

    @property
    def xyz_sig_tot_nm(self) -> torch.Tensor:
        return (self.xyz_sig_nm ** 2).sum(1).sqrt()

    @property
    def xyz_sig_weighted_tot_nm(self) -> torch.Tensor:
        return self._calc_sigma_weighted_total(self.xyz_sig_nm, self.dim() == 3)

    @property
    def phot_scr(self) -> torch.Tensor:  # sqrt cramer-rao of photon count
        return self.phot_cr.sqrt()

    @property
    def bg_scr(self) -> torch.Tensor:  # sqrt cramer-rao of bg count
        return self.bg_cr.sqrt()

    def __getattr__(self, item):
        """Auto unit convert a couple of attributes by trailing unit specification"""
        attr_base = item.rstrip('_nm').rstrip('_px')

        if attr_base in self._power_auto_conversion_attrs.keys():
            tar_unit = item[-2:]
            if tar_unit not in ('nm', 'px'):
                raise NotImplementedError

            return self._pxnm_conversion(
                getattr(self, attr_base),
                in_unit=self.xy_unit,
                tar_unit=tar_unit,
                power=self._power_auto_conversion_attrs[attr_base])

        raise AttributeError

    @property
    def meta(self) -> dict:
        """Return metadata of EmitterSet"""
        return {
            'xy_unit': self.xy_unit,
            'px_size': self.px_size
        }

    @property
    def data(self) -> dict:
        """Return intrinsic data (without metadata)"""
        return {
            'xyz': self.xyz,
            'phot': self.phot,
            'frame_ix': self.frame_ix,
            'id': self.id,
            'prob': self.prob,
            'bg': self.bg,
            'xyz_cr': self.xyz_cr,
            'phot_cr': self.phot_cr,
            'bg_cr': self.bg_cr,
            'xyz_sig': self.xyz_sig,
            'phot_sig': self.phot_sig,
            'bg_sig': self.bg_sig,
        }

    def dim(self) -> int:
        """
        Returns dimensionality of coordinates. If z is 0 everywhere, it returns 2, else 3.

        Note:
            Does not do PCA or other sophisticated things.

        """

        if (self.xyz[:, 2] == 0).all():
            return 2
        else:
            return 3

    def to_dict(self) -> dict:
        """
        Returns dictionary representation of this EmitterSet so that the keys and variables correspond to what an
        EmitterSet would be initialised.

        Example:
            >>> em_dict = em.to_dict()  # any emitterset instance
            >>> em_clone = EmitterSet(**em_dict)  # returns a clone of the emitterset

        """
        em_dict = {}
        em_dict.update(self.meta)
        em_dict.update(self.data)

        return em_dict

    def save(self, file: Union[str, Path]):
        """
        Pickle save's the dictionary of this instance. No legacy guarantees given.
        Should only be used for short-term storage.

        Args:
            file: path where to save

        """
        from decode.utils import emitter_io

        if not isinstance(file, Path):
            file = Path(file)

        if file.suffix == '.pt':
            emitter_io.save_torch(file, self.data, self.meta)
        elif file.suffix in ('.h5', '.hdf5'):
            emitter_io.save_h5(file, self.data, self.meta)
        elif file.suffix == '.csv':
            emitter_io.save_csv(file, self.data, self.meta)
        else:
            raise ValueError

    @staticmethod
    def load(file: Union[str, Path]):
        """
        Loads the set of emitters which was saved by the 'save' method.

        Args:
            file: path to the emitterset

        Returns:
            EmitterSet

        """
        from decode.utils import emitter_io

        file = Path(file) if not isinstance(file, Path) else file

        if file.suffix == '.pt':
            em_dict, meta, _ = emitter_io.load_torch(file)
        elif file.suffix in ('.h5', '.hdf5'):
            em_dict, meta, _ = emitter_io.load_h5(file)
        elif file.suffix == '.csv':
            warnings.warn("For .csv files, implicit usage of .load() is discouraged. "
                          "Please use 'decode.utils.emitter_io.load_csv' explicitly.")
            em_dict, meta, _ = emitter_io.load_csv(file)
        else:
            raise ValueError

        em_dict.update(meta)

        return EmitterSet(**em_dict)

    def _set_typed(self, xyz, phot, frame_ix, id, prob, bg, xyz_cr, phot_cr, bg_cr, xyz_sig, phot_sig, bg_sig):
        """
        Sets the attributes in the correct type and with default argument if None
        """

        if xyz.dtype not in (torch.float, torch.double, torch.half):
            raise ValueError("XYZ coordinates must be float type.")
        else:
            f_type = xyz.dtype

        if frame_ix.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError(f"Frame index must be integer type and not {frame_ix.dtype}.")

        if id is not None and (id.dtype not in (torch.int16, torch.int32, torch.int64)):
            raise ValueError(f"ID must be None or integer type not {id.dtype}.")

        i_type = torch.int64

        # make xyz always 3 dim
        xyz = xyz if xyz.shape[1] == 3 else torch.cat((xyz, torch.zeros_like(xyz[:, [0]])), 1)

        num_input = int(xyz.shape[0]) if xyz.shape[0] != 0 else 0

        """Set values"""
        if num_input != 0:
            self.xyz = xyz
            self.phot = phot.type(f_type)
            self.frame_ix = frame_ix.type(i_type)

            # Optionals
            self.id = id if id is not None else -torch.ones_like(frame_ix)
            self.prob = prob.type(f_type) if prob is not None else torch.ones_like(frame_ix).type(f_type)
            self.bg = bg.type(f_type) if bg is not None else float('nan') * torch.ones_like(frame_ix).type(f_type)

            self.xyz_cr = xyz_cr.type(f_type) if xyz_cr is not None else float('nan') * torch.ones_like(self.xyz)
            self.phot_cr = phot_cr.type(f_type) if phot_cr is not None else float('nan') * torch.ones_like(self.phot)
            self.bg_cr = bg_cr.type(f_type) if bg_cr is not None else float('nan') * torch.ones_like(self.bg)

            self.xyz_sig = xyz_sig.type(f_type) if xyz_sig is not None else float('nan') * torch.ones_like(self.xyz)
            self.phot_sig = phot_sig.type(f_type) if phot_sig is not None else float('nan') * torch.ones_like(self.phot)
            self.bg_sig = bg_sig.type(f_type) if bg_sig is not None else float('nan') * torch.ones_like(self.bg)

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
            self.bg_cr = float('nan') * torch.ones_like(self.bg)

            self.xyz_sig = float('nan') * torch.ones((0, 3)).type(f_type)
            self.phot_sig = float('nan') * torch.ones_like(self.prob)
            self.bg_sig = float('nan') * torch.ones_like(self.bg)

    def _inplace_replace(self, em):
        """
        Inplace replacement of this self instance. Does not work for inherited methods ...
        Args:
            em: other EmitterSet instance that should replace self


        """
        self.__init__(**em.to_dict(), sanity_check=False)

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

        if len(self) >= 1:
            print_str += f"\n::xy unit: {self.xy_unit}"
            print_str += f"\n::px size: {self.px_size}"
            print_str += f"\n::frame range: {self.frame_ix.min().item()} - {self.frame_ix.max().item()}" \
                         f"\n::spanned volume: {self.xyz.min(0)[0].numpy()} - {self.xyz.max(0)[0].numpy()}"
        return print_str

    def __add__(self, other):
        return self.cat((self, other), None, None)

    def __iadd__(self, other):
        self._inplace_replace(self + other)
        return self

    def __eq__(self, other) -> bool:
        """
        Implements equalness check. Returns true if all attributes are the same and in the same order.
        If it fails, you may want to sort by the ID first and then check again.

        Args:
            other: (emitterset)

        Returns:
            true if as stated above.

        """

        def check_em_dict_equality(em_a: dict, em_b: dict) -> bool:

            for k in em_a.keys():
                if not tutil.tens_almeq(em_a[k], em_b[k], nan=True):
                    return False

            return True

        if not self.eq_attr(other):
            return False

        if not check_em_dict_equality(self.data, other.data):
            return False

        return True

    def eq_attr(self, other) -> bool:
        """
        Tests whether the meta attributes (xy_unit and px size) are the same

        Args:
            other: the EmitterSet to compare to

        """
        if self.px_size is None:
            if other.px_size is not None:
                return False

        elif not (self.px_size == other.px_size).all():
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
            return self._get_subset(self.n - 1)
        else:
            raise StopIteration

    def __getitem__(self, item):
        """
        Implements array indexing for this class.

        Args:
            item: (int), or indexing

        Returns:
            EmitterSet

        """

        if isinstance(item, int) and item >= len(self):
            raise IndexError(f"Index {item} out of bounds of EmitterSet of size {len(self)}")

        return self._get_subset(item)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def clone(self):
        """
        Returns a deep copy of this EmitterSet.

        Returns:
            EmitterSet

        """
        return copy.deepcopy(self)

    def _calc_sigma_weighted_total(self, xyz_sigma_nm, use_3d):

        x_sig_var = torch.var(xyz_sigma_nm[:, 0])
        y_sig_var = torch.var(xyz_sigma_nm[:, 1])
        tot_var = xyz_sigma_nm[:, 0] ** 2 + (torch.sqrt(x_sig_var / y_sig_var) * xyz_sigma_nm[:, 1]) ** 2

        if use_3d:
            z_sig_var = torch.var(xyz_sigma_nm[:, 2])
            tot_var += (torch.sqrt(x_sig_var / z_sig_var) * xyz_sigma_nm[:, 2]) ** 2

        return torch.sqrt(tot_var)

    @staticmethod
    def cat(emittersets: Iterable, remap_frame_ix: Union[None, torch.Tensor] = None, step_frame_ix: int = None):
        """
        Concatenate multiple emittersets into one emitterset which is returned. Optionally modify the frame indices by
        the arguments.

        Args:
            emittersets: iterable of emittersets to be concatenated
            remap_frame_ix: new index of the 0th frame of each iterable
            step_frame_ix: step size between 0th frame of each iterable

        Returns:
            concatenated emitters

        """

        meta = []
        data = []
        for em in emittersets:
            meta.append(em.meta)
            data.append(em.data)

        n_chunks = len(data)

        if remap_frame_ix is not None and step_frame_ix is not None:
            raise ValueError("You cannot specify remap frame ix and step frame ix at the same time.")
        elif remap_frame_ix is not None:
            shift = remap_frame_ix.clone()
        elif step_frame_ix is not None:
            shift = torch.arange(0, n_chunks) * step_frame_ix
        else:
            shift = torch.zeros(n_chunks).int()

        # apply shift
        for d, s in zip(data, shift):
            d['frame_ix'] = d['frame_ix'] + s

        # list of dicts to dict of lists
        data = {k: torch.cat([x[k] for x in data], 0) for k in data[0]}
        # meta = {k: [x[k] for x in meta] for k in meta[0]}

        # px_size and xy unit is taken from the first element that is not None
        xy_unit = None
        px_size = None

        for m in meta:
            if m['xy_unit'] is not None:
                xy_unit = m['xy_unit']
                break
        for m in meta:
            if m['px_size'] is not None:
                px_size = m['px_size']
                break

        return EmitterSet(xy_unit=xy_unit, px_size=px_size, **data)

    def sort_by_frame_(self):
        """
        Inplace sort this emitterset by its frame index.

        """
        em = self.sort_by_frame()
        self._inplace_replace(em)

    def sort_by_frame(self):
        """
        Sort a deepcopy of this emitterset and return it.

        Returns:
            Sorted copy of this emitterset

        """
        _, ix = self.frame_ix.sort()
        em = self[ix]
        em._sorted = True

        return em

    def _get_subset(self, ix):
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

        return EmitterSet(xyz=self.xyz[ix], phot=self.phot[ix], frame_ix=self.frame_ix[ix], id=self.id[ix],
                          prob=self.prob[ix], bg=self.bg[ix],
                          xyz_sig=self.xyz_sig[ix], phot_sig=self.phot_sig[ix], bg_sig=self.bg_sig[ix],
                          xyz_cr=self.xyz_cr[ix], phot_cr=self.phot_cr[ix], bg_cr=self.bg_cr[ix],
                          sanity_check=False, xy_unit=self.xy_unit, px_size=self.px_size)

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
    def single_frame(self) -> bool:
        """
        Check if all emitters are on the same frame.

        Returns:
            bool

        """
        return True if torch.unique(self.frame_ix).shape[0] == 1 else False

    def chunks(self, chunks: int):
        """
        Splits the EmitterSet into (almost) equal chunks

        Args:
            chunks (int): number of splits

        Returns:
            list: of emittersets

        """
        n = len(self)
        l = self
        k = chunks
        # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115#37414115
        return [l[i * (n // k) + min(i, n % k):(i + 1) * (n // k) + min(i + 1, n % k)] for i in range(k)]

    def filter_by_sigma(self, fraction: float, dim: Optional[int] = None, return_low=True):
        """
        Filter by sigma values. Returns EmitterSet.

        Args:
            fraction: relative fraction of emitters remaining after filtering. Ranges from 0. to 1.
            dim: 2 or 3 for taking into account z. If None, it will be autodetermined.
            return_low:
                if True return the fraction of emitter with the lowest sigma values.
                if False return the (1-fraction) with the highest sigma values.

        """
        if dim is None:
            is_3d = False if self.dim() == 2 else True
        else:
            is_3d = False if dim == 2 else True

        if fraction == 1.:
            return self

        xyz_sig = self.xyz_sig

        x_sig_var = torch.var(xyz_sig[:, 0])
        y_sig_var = torch.var(xyz_sig[:, 1])
        z_sig_var = torch.var(xyz_sig[:, 2])
        tot_var = xyz_sig[:, 0] ** 2 + (torch.sqrt(x_sig_var / y_sig_var) * xyz_sig[:, 1]) ** 2

        if is_3d:
            tot_var += (np.sqrt(x_sig_var / z_sig_var) * xyz_sig[:, 2]) ** 2

        max_s = np.percentile(tot_var.cpu().numpy(), fraction * 100.)
        if return_low:
            filt_sig = torch.where(tot_var < max_s)
        else:
            filt_sig = torch.where(tot_var > max_s)

        return self[filt_sig]

    def hist_detection(self) -> dict:
        """
        Compute hist for detection associated attributes.

        """

        return {
            'prob': np.histogram(self.prob),
            'sigma_x': np.histogram(self.xyz_sig[:, 0]),
            'sigma_y': np.histogram(self.xyz_sig[:, 1]),
            'sigma_z': np.histogram(self.xyz_sig[:, 2]),
        }

    def split_in_frames(self, ix_low: int = 0, ix_up: int = None) -> list:
        """
        Splits a set of emitters in a list of emittersets based on their respective frame index.

        Args:
            ix_low: (int, 0) lower bound
            ix_up: (int, None) upper bound

        Returns:
            list

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

            return self._convert_coordinates(factor=1 / self.px_size ** power, xyz=xyz)

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

    def __init__(self, num_emitters: int, extent: float = 32, xy_unit: str = 'px', px_size: tuple = None):
        xyz = torch.rand((num_emitters, 3)) * extent
        super().__init__(xyz, torch.ones_like(xyz[:, 0]), torch.zeros_like(xyz[:, 0]).long(),
                         xy_unit=xy_unit, px_size=px_size)

    def _inplace_replace(self, em):
        super().__init__(**em.to_dict(), sanity_check=False)


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
        super().__init__(**em.to_dict(), sanity_check=False)


class EmptyEmitterSet(CoordinateOnlyEmitter):
    """An empty emitter set."""

    def __init__(self, xy_unit=None, px_size=None):
        super().__init__(torch.zeros((0, 3)), xy_unit=xy_unit, px_size=px_size)

    def _inplace_replace(self, em):
        super().__init__(**em.to_dict())


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
        frame_ix_ = frame_start_full.repeat_interleave(frame_dur_full_clean + 1, dim=0) \
                    + decode.generic.utils.cum_count_per_group(id_) + 1

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
        return EmitterSet(xyz_, phot_, frame_ix_.long(), id_.long(), xy_unit=self.xy_unit, px_size=self.px_size)


def at_least_one_dim(*args) -> None:
    """Make tensors at least one dimensional (inplace)"""
    for arg in args:
        if arg.dim() == 0:
            arg.unsqueeze_(0)


def same_shape_tensor(dim, *args) -> bool:
    """Test if tensors are of same size in a certain dimension."""
    for i in range(args.__len__() - 1):
        if args[i].size(dim) == args[i + 1].size(dim):
            continue
        else:
            return False

    return True


def same_dim_tensor(*args) -> bool:
    """Test if tensors are of same dimensionality"""
    for i in range(args.__len__() - 1):
        if args[i].dim() == args[i + 1].dim():
            continue
        else:
            return False

    return True
