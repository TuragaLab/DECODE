import copy
import warnings
from pathlib import Path
from typing import Union, Optional, Iterable

import numpy as np
import torch
from deprecated import deprecated
from pydantic import BaseModel, root_validator, validator

import decode.generic.utils
from decode.generic import slicing
from decode.generic import test_utils


class _Tensor(torch.Tensor):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)

        return v


class _LongTensor(_Tensor):
    @classmethod
    def validate(cls, v):
        if not isinstance(v, torch.LongTensor):
            v = torch.tensor(v, dtype=torch.long)

        return v


class EmitterData(BaseModel):
    """
    Helper class which holds and validates the data fields of the EmitterSet.
    Usually this class not used directly.
    """

    xyz: _Tensor
    phot: _Tensor
    frame_ix: _LongTensor

    id: Optional[_LongTensor]
    code: Optional[_LongTensor]
    prob: Optional[_Tensor]
    bg: Optional[_Tensor]

    xyz_cr: Optional[_Tensor]
    phot_cr: Optional[_Tensor]
    bg_cr: Optional[_Tensor]

    xyz_sig: Optional[_Tensor]
    phot_sig: Optional[_Tensor]
    bg_sig: Optional[_Tensor]

    @staticmethod
    def _val_coord(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1 or v.dim() > 3:
            raise ValueError("Not supported shape.")

        if v.size(-1) == 2:
            v = torch.cat((v, torch.zeros_like(v[..., [0]])), -1)

        return v

    @validator("xyz")
    def xyz_prep(cls, v):
        return cls._val_coord(v)

    @validator("xyz_cr", "xyz_sig")
    def xyz_optional(cls, v):
        if v is None:
            return v

        return cls._val_coord(v)

    @root_validator
    def equal_length(cls, v):
        n = {len(vv) for vv in v.values() if vv is not None}
        if len(n) >= 2:
            raise ValueError("Unequal length of fields")

        return v


class EmitterSet:
    """
    Class, storing a set of emitters and its attributes.
    Probably the most commonly used class of this framework.

    Attributes:
            xyz: Coordinates of size N x [2,3].
            phot: Photon count of size N
            frame_ix: size N. Index on which the emitter appears.
            id: size N. Identity the emitter.
            code: size N.
            prob: size N. Probability estimate of the emitter.
            bg: size N. Background estimate of emitter.
            xyz_cr: size N x 3. Cramer-Rao estimate of the emitters position.
            phot_cr: size N. Cramer-Rao estimate of the emitters photon count.
            bg_cr: size N. Cramer-Rao estimate of the emitters background value.
            sanity_check: performs a sanity check if true.
            xy_unit: Unit of the x and y coordinate.
            px_size: Pixel size for unit conversion. If not specified,
                derived attributes (xyz_px and xyz_nm) can not be accessed
            iframe: select by frame_index (e.g. emitter.iframe[0:10] will output
                emitters on frames 0-9
            icode: select by code (similiar to iframe)
    """

    _data_holder = {
        "xyz",
        "phot",
        "frame_ix",
        "id",
        "code",
        "prob",
        "bg",
        "xyz_cr",
        "phot_cr",
        "bg_cr",
        "xyz_sig",
        "phot_sig",
        "bg_sig",
    }
    _eq_precision = 1e-8
    _power_auto_conversion_attrs = {"xyz_cr": 2, "xyz_sig": 1}
    _xy_units = ("px", "nm")

    def __init__(
        self,
        xyz: torch.Tensor,
        phot: torch.Tensor,
        frame_ix: torch.LongTensor,
        id: Optional[torch.LongTensor] = None,
        *,
        code: Optional[torch.Tensor] = None,
        prob: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
        xyz_cr: Optional[torch.Tensor] = None,
        phot_cr: Optional[torch.Tensor] = None,
        bg_cr: Optional[torch.Tensor] = None,
        xyz_sig: Optional[torch.Tensor] = None,
        phot_sig: Optional[torch.Tensor] = None,
        bg_sig: Optional[torch.Tensor] = None,
        sanity_check: bool = True,
        xy_unit: str = None,
        px_size: Union[tuple, torch.Tensor] = None,
    ):
        """
        Initialises EmitterSet of :math:`N` emitters.

        Args:
            xyz: Coordinates of size :math:`(N,3)`
            phot: Photon count of size :math:`N`
            frame_ix: Index on which the emitter appears. Must be integer type. Size :math:`N`
            id: Identity the emitter. Must be tensor integer type and the same type as frame_ix. Size :math:`N`
            code: Code of emitter. Useful e.g. for representing its color, channel ix or similar,
             usually size :math:`(N)`
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

        # EmitterData validates and holds the data
        self._data = EmitterData(
            xyz=xyz,
            phot=phot,
            frame_ix=frame_ix,
            id=id,
            code=code,
            prob=prob,
            bg=bg,
            xyz_cr=xyz_cr,
            phot_cr=phot_cr,
            bg_cr=bg_cr,
            xyz_sig=xyz_sig,
            phot_sig=phot_sig,
            bg_sig=bg_sig,
        )

        self.xy_unit = xy_unit
        self.px_size = px_size
        self.iframe = slicing.SliceForward(self._iframe_hook)
        self.icode = slicing.SliceForward(self._icode_hook)
        self._sorted = False

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
        return self._pxnm_conversion(
            self.xyz, in_unit=self.xy_unit, tar_unit="px", power=1.0
        )

    @xyz_px.setter
    def xyz_px(self, xyz):
        self.xyz = xyz
        self.xy_unit = "px"

    @property
    def xyz_nm(self) -> torch.Tensor:
        """
        Returns xyz in nanometres and performs respective transformations if needed.
        """
        return self._pxnm_conversion(
            self.xyz, in_unit=self.xy_unit, tar_unit="nm", power=1.0
        )

    @xyz_nm.setter
    def xyz_nm(self, xyz):  # xyz in nanometres
        self.xyz = xyz
        self.xy_unit = "nm"

    @property
    def xyz_scr(self) -> torch.Tensor:
        """
        Square-Root cramer rao of xyz.
        """
        return self.xyz_cr.sqrt() if self.xyz_cr is not None else None

    @property
    def xyz_scr_px(self) -> torch.Tensor:
        """
        Square-Root cramer rao of xyz in px units.
        """
        return self.xyz_cr_px.sqrt() if self.xyz_cr_px is not None else None

    @property
    def xyz_scr_nm(self) -> torch.Tensor:
        return self.xyz_cr_nm.sqrt() if self.xyz_cr_nm is not None else None

    @property
    def xyz_sig_tot_nm(self) -> torch.Tensor:
        return (
            (self.xyz_sig_nm**2).sum(1).sqrt()
            if self.xyz_sig_nm is not None
            else None
        )

    @property
    def xyz_sig_weighted_tot_nm(self) -> torch.Tensor:
        return self._calc_sigma_weighted_total(self.xyz_sig_nm, self.dim() == 3)

    @property
    def phot_scr(self) -> torch.Tensor:  # sqrt cramer-rao of photon count
        return self.phot_cr.sqrt() if self.phot_cr is not None else None

    @property
    def bg_scr(self) -> torch.Tensor:  # sqrt cramer-rao of bg count
        return self.bg_cr.sqrt() if self.bg_cr is not None else None

    def __getattr__(self, item):
        # refer to data holder
        if item in self._data_holder:
            return getattr(self._data, item)

        # auto unit convert a couple of attributes by trailing unit specification
        attr_base = item.rstrip("_nm").rstrip("_px")

        if attr_base in self._power_auto_conversion_attrs.keys():
            tar_unit = item[-2:]
            if tar_unit not in ("nm", "px"):
                raise NotImplementedError

            return self._pxnm_conversion(
                getattr(self, attr_base),
                in_unit=self.xy_unit,
                tar_unit=tar_unit,
                power=self._power_auto_conversion_attrs[attr_base],
            )

        raise AttributeError

    @property
    def meta(self) -> dict:
        """Return metadata of EmitterSet"""
        return {"xy_unit": self.xy_unit, "px_size": self.px_size}

    @property
    def data(self) -> dict:
        """Return intrinsic data (without metadata)"""
        return {
            "xyz": self.xyz,
            "phot": self.phot,
            "frame_ix": self.frame_ix,
            "id": self.id,
            "code": self.code,
            "prob": self.prob,
            "bg": self.bg,
            "xyz_cr": self.xyz_cr,
            "phot_cr": self.phot_cr,
            "bg_cr": self.bg_cr,
            "xyz_sig": self.xyz_sig,
            "phot_sig": self.phot_sig,
            "bg_sig": self.bg_sig,
        }

    @property
    def data_used(self) -> dict:
        """Return intrinsic data without non-used optionals."""
        return {k: v for k, v in self.data.items() if v is not None}

    @property
    def single_frame(self) -> bool:
        """All emitters are on the same frame?"""
        return True if torch.unique(self.frame_ix).shape[0] == 1 else False

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
        Returns dictionary representation of this EmitterSet so that the keys and
        variables correspond to what an EmitterSet would be initialised.

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

        if file.suffix == ".pt":
            emitter_io.save_torch(file, self.data, self.meta)
        elif file.suffix in (".h5", ".hdf5"):
            emitter_io.save_h5(file, self.data, self.meta)
        elif file.suffix == ".csv":
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

        if file.suffix == ".pt":
            em_dict, meta, _ = emitter_io.load_torch(file)
        elif file.suffix in (".h5", ".hdf5"):
            em_dict, meta, _ = emitter_io.load_h5(file)
        elif file.suffix == ".csv":
            warnings.warn(
                "For .csv files, implicit usage of .load() is discouraged. "
                "Please use 'decode.utils.emitter_io.load_csv' explicitly."
            )
            em_dict, meta, _ = emitter_io.load_csv(file)
        else:
            raise ValueError

        em_dict.update(meta)

        return EmitterSet(**em_dict)

    def _inplace_replace(self, em):
        """
        Inplace replacement of this self instance.
        Does not work for inherited methods ...

        Args:
            em: other EmitterSet instance that should replace self

        """
        self.__init__(sanity_check=False, **em.to_dict())

    def _sanity_check(self, check_uniqueness=False):
        """
        Performs several integrity tests on the EmitterSet.

        Args:
            check_uniqueness: (bool) check the uniqueness of the ID

        Returns:
            (bool) sane or not sane
        """
        attr_check = [x for x in self.data.values() if x is not None]
        if not test_utils.same_shape_tensor(0, *attr_check):
            raise ValueError("Data attributes should be of same length.")

        attr_check = [
            getattr(self, a)
            for a in ("prob", "frame_ix", "id")
            if getattr(self, a) is not None
        ]
        if not test_utils.same_dim_tensor(torch.ones(1), *attr_check):
            raise ValueError("Expected probability frame index and id to be 1D.")

        # motivate to specify an xyz unit.
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
        print_str = f"EmitterSet" f"\n::num emitters: {len(self)}"

        if len(self) >= 1:
            print_str += f"\n::xy unit: {self.xy_unit}"
            print_str += f"\n::px size: {self.px_size}"
            print_str += (
                f"\n::frame range: {self.frame_ix.min().item()} - {self.frame_ix.max().item()}"
                f"\n::spanned volume: {self.xyz.min(0)[0].numpy()} - {self.xyz.max(0)[0].numpy()}"
            )
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
        if len(self) != len(other):
            return False

        if not self.eq_meta(other):
            return False

        if not self.eq_data(other):
            return False

        return True

    def eq_meta(self, other) -> bool:
        """Tests whether the meta attributes are the same"""
        if self.px_size is None:
            if other.px_size is not None:
                return False

        elif not (self.px_size == other.px_size).all():
            return False

        if not self.xy_unit == other.xy_unit:
            return False

        return True

    def eq_data(self, other) -> bool:
        """Tests whether data attributes are the same"""

        def check_em_dict_equality(em_a: dict, em_b: dict) -> bool:
            for k in em_a.keys():
                # finally check tensors, reject if one is None and the other is set
                if not test_utils.tens_almeq(em_a[k], em_b[k], nan=True, none="either"):
                    return False

            return True

        return check_em_dict_equality(self.data, other.data)

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

    def __getitem__(self, item: Union[int, slice]):
        """
        Implements array indexing for this class.

        Args:
            item:

        Returns:
            EmitterSet

        Notes:
            Single element access will still not change the dimensionality of the data
            attributes, i.e. if xyz is 2 dimensional em[0] will still result in an
            EmitterSet where xyz is 2 dimensional (contrary to xyz[0] which would be
            reduced by one dim).

        """

        if isinstance(item, int) and item >= len(self):
            raise IndexError(
                f"Index {item} out of bounds of EmitterSet of size {len(self)}"
            )

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
        tot_var = (
            xyz_sigma_nm[:, 0] ** 2
            + (torch.sqrt(x_sig_var / y_sig_var) * xyz_sigma_nm[:, 1]) ** 2
        )

        if use_3d:
            z_sig_var = torch.var(xyz_sigma_nm[:, 2])
            tot_var += (torch.sqrt(x_sig_var / z_sig_var) * xyz_sigma_nm[:, 2]) ** 2

        return torch.sqrt(tot_var)

    @staticmethod
    def cat(
        emittersets: Iterable,
        remap_frame_ix: Union[None, torch.Tensor] = None,
        step_frame_ix: int = None,
    ):
        """
        Concatenate multiple emittersets into one emitterset which is returned.
        Optionally modify the frame index by providing a step between elements or
        directly specifying to what new index the 0th frame of each element maps to.

        Args:
            emittersets: sequence of emitters to be concatenated
            remap_frame_ix: new index of the 0th frame of each element
            step_frame_ix: step size between 0th frame of each element

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
            raise ValueError(
                "You cannot specify remap frame ix and step frame ix at the same time."
            )
        elif remap_frame_ix is not None:
            shift = remap_frame_ix.clone()
        elif step_frame_ix is not None:
            shift = torch.arange(0, n_chunks) * step_frame_ix
        else:
            shift = torch.zeros(n_chunks).int()

        # apply frame index shift
        for d, s in zip(data, shift):
            d["frame_ix"] = d["frame_ix"] + s

        # concatenate data in list of dicts to dict of concatenated tensors
        data = {
            # cat key-wise, if value of key is None in first emitter, it will be None in all
            k: torch.cat([x[k] for x in data], 0) if data[0][k] is not None else None
            for k in data[0]
        }

        # px_size and xy unit is taken from the first element that is not None
        xy_unit = None
        px_size = None
        for m in meta:
            if m["xy_unit"] is not None:
                xy_unit = m["xy_unit"]
                break
        for m in meta:
            if m["px_size"] is not None:
                px_size = m["px_size"]
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

    def _get_subset(self, ix: Union[int, slice]):
        """
        Returns subset of emitterset. Implementation of __getitem__ and __next__ methods.
        Args:
            ix: (int, list) integer index or list of indices

        Returns:
            (EmitterSet)
        """
        # ToDo: Does this makes sense? This leads to always keeping a batch dim, which
        #   is maybe not wanted?
        if isinstance(ix, int):
            ix = [ix]

        # PyTorch single element support
        if (
            not isinstance(ix, torch.BoolTensor)
            and isinstance(ix, torch.Tensor)
            and ix.numel() == 1
        ):
            ix = [int(ix)]

        # Todo: Check for numpy boolean array
        if isinstance(ix, (np.ndarray, np.generic)) and ix.size == 1:
            ix = [int(ix)]

        data_subset = {k: v[ix] for k, v in self.data.items() if v is not None}

        return EmitterSet(
            sanity_check=False,
            xy_unit=self.xy_unit,
            px_size=self.px_size,
            **data_subset,
        )

    def get_subset_frame(
        self, frame_start: Optional[int], frame_end: Optional[int], frame_ix_shift=None
    ):
        """
        Returns emitters that are in the frame range as specified.

        Args:
            frame_start: (int) lower frame index limit
            frame_end: (int) upper frame index limit (pythonic, exclusive)
            frame_ix_shift: shift frame index (additive)

        """
        ix_low = (
            self.frame_ix >= frame_start
            if frame_start is not None
            else torch.ones_like(self.frame_ix, dtype=torch.bool)
        )
        ix_high = (
            self.frame_ix < frame_end
            if frame_end is not None
            else torch.ones_like(self.frame_ix, dtype=torch.bool)
        )

        ix = ix_low * ix_high
        em = self[ix]

        if not frame_ix_shift:
            return em
        elif len(em) != 0:  # only shift if there is actually something
            em.frame_ix += frame_ix_shift

        return em

    def _iframe_hook(self, item):
        if isinstance(item, slice):
            if item.step is not None:
                raise NotImplementedError("Step argument in slicing not supported.")
            return self.get_subset_frame(item.start, item.stop, None)

        if isinstance(item, int):
            return self.get_subset_frame(item, item + 1)

        raise NotImplementedError

    def _icode_hook(self, item):
        if isinstance(item, slice):
            if item.step is not None:
                raise NotImplementedError("Step argument in slicing not supported.")
            item = (item.start, item.stop)
        elif isinstance(item, int):
            item = (item, item + 1)
        else:
            raise NotImplementedError

        ix = (self.code >= item[0]) * (self.code < item[1])
        return self[ix]

    def chunks(self, chunks: int) -> list:
        """
        Splits the EmitterSet into (almost) equal chunks

        Args:
            chunks (int): number of splits

        Returns:
            list: of emitters
        """
        n = len(self)
        l = self
        k = chunks
        # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115#37414115
        return [
            l[i * (n // k) + min(i, n % k) : (i + 1) * (n // k) + min(i + 1, n % k)]
            for i in range(k)
        ]

    def filter_by_sigma(
        self, fraction: float, dim: Optional[int] = None, return_low=True
    ):
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

        if fraction == 1.0:
            return self

        xyz_sig = self.xyz_sig

        x_sig_var = torch.var(xyz_sig[:, 0])
        y_sig_var = torch.var(xyz_sig[:, 1])
        z_sig_var = torch.var(xyz_sig[:, 2])
        tot_var = (
            xyz_sig[:, 0] ** 2
            + (torch.sqrt(x_sig_var / y_sig_var) * xyz_sig[:, 1]) ** 2
        )

        if is_3d:
            tot_var += (np.sqrt(x_sig_var / z_sig_var) * xyz_sig[:, 2]) ** 2

        max_s = np.percentile(tot_var.cpu().numpy(), fraction * 100.0)
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
            "prob": np.histogram(self.prob),
            "sigma_x": np.histogram(self.xyz_sig[:, 0]),
            "sigma_y": np.histogram(self.xyz_sig[:, 1]),
            "sigma_z": np.histogram(self.xyz_sig[:, 2]),
        }

    def split_in_frames(self, ix_low: int = 0, ix_up: int = None) -> list:
        """
        Splits a set of emitters in a list of emittersets based on their respective frame index.

        Args:
            ix_low: lower frame index
            ix_high: upper frame index (pythonic, exclusive)

        Returns:
            list of emitters
        """

        # the first frame is assumed to be 0. If it's negative go to the lowest negative
        ix_low = ix_low if ix_low is not None else self.frame_ix.min().item()
        ix_up = ix_up if ix_up is not None else self.frame_ix.max().item() + 1

        return slicing.split_sliceable(
            x=self, x_ix=self.frame_ix, ix_low=ix_low, ix_high=ix_up
        )

    def _pxnm_conversion(self, xyz, in_unit, tar_unit, power: float = 1.0):

        if in_unit is None:
            raise ValueError("Conversion not possible if unit not specified.")

        if in_unit == tar_unit:
            return xyz

        elif in_unit == "nm" and tar_unit == "px":
            """px check needs to happen here, because in _convert_coordinates, factor is an optional argument."""
            if self.px_size is None:
                raise ValueError("Conversion not possible if px size is not specified.")

            return self._convert_coordinates(factor=1 / self.px_size**power, xyz=xyz)

        elif in_unit == "px" and tar_unit == "nm":
            if self.px_size is None:
                raise ValueError("Conversion not possible if px size is not specified.")

            return self._convert_coordinates(factor=self.px_size**power, xyz=xyz)

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
                factor = torch.cat((factor, torch.tensor([1.0])), 0)

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

    def repeat(self, repeats: Union[int, torch.Tensor], step_frames: bool):
        """
        Repeat EmitterSet in interleave style

        Args:
            repeats: number of repeats or tensor of repeats per emitter
            step_frames: adjust frame index as by repeat (true) or keep constant (False)

        Returns:
            repeated EmitterSet

        Examples:
            - interleave, i.e. [0, 1] becomes [0, 0, 1, 1]
        """
        data_repeat = {
            k: torch.repeat_interleave(v, repeats, dim=0)
            for k, v in self.data_used.items()
        }

        em = EmitterSet(**data_repeat, **self.meta)

        # adjust frame index
        if step_frames:
            # use pseudo id because we don't care about identity of emitter
            pseudo_id = torch.arange(len(self))
            pseudo_id = torch.repeat_interleave(pseudo_id, repeats, dim=0)
            cum_count = decode.generic.utils.cum_count_per_group(pseudo_id)

            em.frame_ix += cum_count

        return em

    def _get_channeled_attrs(self) -> tuple[dict, dict]:
        """Get coord-like attributes that have a channel dimension.

        Returns:
            - coord like attributes with channels
            - other (generic) attributes with channels
        """
        attr_coord = {
            k: v
            for k, v in self.data.items()
            if ("xyz" in k) and (v is not None) and (v.dim() == 3)
        }

        # get remaining attributes that have a channel dimension
        attr_generic = {
            k: v
            for k, v in self.data.items()
            if ("xyz" not in k) and (v is not None) and (v.dim() == 2)
        }
        return attr_coord, attr_generic

    @staticmethod
    def _infer_code_length(attr_coord: dict, attr_generic: dict) -> int:
        # infers code length by channel dimension
        if not (len(attr_coord) >= 1 or len(attr_generic) >= 1):  # nothing to do
            raise ValueError("Can not infer code length if neither coords nor other "
                             "attributes have channel dimension")

        # get number of channel dims
        n_repeats = {xyz.size(1) for xyz in attr_coord.values()} | {
            generic.size(-1) for generic in attr_generic.values()
        }

        if len(n_repeats) != 1:
            raise ValueError(
                "Inconsistent number of channels but channels are present."
            )

        return next(iter(n_repeats))

    def infer_code(self) -> torch.Tensor:
        """
        Infers code from attributes that have a channel dimension.

        Returns:
            - codes with integer indexing of size N x C
        """
        attr_coord, attr_gen = self._get_channeled_attrs()
        n_repeats = self._infer_code_length(attr_coord, attr_gen)

        code = torch.arange(n_repeats).unsqueeze(0).repeat(len(self), 1)
        return code

    def linearize(self, infer_code: bool = False) -> "EmitterSet":
        """
        Linearizes an EmitterSet that has some 2D attributes (e.g. photons).

        Args:
            infer_code: assigns integer code by channel dimension, no-op if `code`
             attribute is present
        """
        attr_coord_ch, attr_2d = self._get_channeled_attrs()
        n_repeats = self._infer_code_length(attr_coord_ch, attr_2d)

        if self.code is None and infer_code:
            code = self.infer_code()
            attr_2d.update({"code": code})

        em = self.clone()
        em = em.repeat(n_repeats, False)

        if len(attr_coord_ch) >= 1:
            for k, v in attr_coord_ch.items():
                v = v.reshape(-1, v.size(-1))  # linearize away N x C x 3 to (N*C) x 3
                setattr(em, k, v)

        if len(attr_2d) >= 1:
            for k, v in attr_2d.items():
                v = v.reshape(-1)  # linearize away N x C to (N*C)
                setattr(em, k, v)

        return em


class FluorophoreSet:
    def __init__(
        self,
        xyz: torch.Tensor,
        flux: torch.Tensor,
        t0: torch.Tensor,
        ontime: torch.Tensor,
        xy_unit: str,
        px_size: Union[tuple, torch.Tensor] = None,
        id: Optional[torch.LongTensor] = None,
        sanity_check=True,
        **kwargs,
    ):
        """
        Something that starts to emit light at time `t0` and is on for a specific
        ontime. Related to the standard EmitterSet. However, here we do not specify a
        frame_ix but rather a (non-integer) initial point in time where the emitter
        starts to blink and an on-time.

        Args:
            xyz: coordinates. Dimension: N x 3
            flux: flux, i.e. photon flux per time unit. Dimension N
            t0: initial blink event. Dimension: N
            ontime: duration in frame-time units how long the emitter blinks.
                Dimension N
            id: identity of the emitter. Dimension: N
            xy_unit: unit of the coordinates
            id: id of the emitter
        """

        self.xyz = xyz
        self.flux = flux
        self.t0 = t0
        self.ontime = ontime
        self.id = id if id is not None else torch.arange(len(flux))
        self.xy_unit = xy_unit
        self.px_size = px_size
        self._em_kwargs = kwargs

        if sanity_check:
            self.sanity_check()

    def __len__(self) -> int:
        return len(self.xyz)

    def sanity_check(self):

        # check ids
        if self.id.unique().numel() != self.id.numel():
            raise ValueError("IDs are not unique.")

        if (self.flux < 0).any():
            raise ValueError("Negative flux values encountered.")

        if (self.ontime < 0).any():
            raise ValueError("Negative ontime encountered.")

    @property
    def te(self):
        # end time
        return self.t0 + self.ontime

    @staticmethod
    def _compute_time_distribution(
        t_start: torch.FloatTensor, t_end: torch.FloatTensor
    ) -> (torch.LongTensor, torch.Tensor):
        """
        Compute time distribution, i.e. on how many frames an emitter is visible
        and what the ontime per emitter per frame is

        Args:
            t_start: start time
            t_end: end time
        """
        # compute total number of frames per emitter
        ix_start = torch.floor(t_start).long()
        ix_end = torch.floor(t_end).long()

        n_frames = (ix_end - ix_start + 1).long()

        # compute ontime per frame
        # ontime = torch.repeat_interleave(torch.ones_like(t_start), n_frames, 0)
        pseudo_id = torch.repeat_interleave(torch.arange(len(t_start)), n_frames, 0)
        n_frame_per_emitter = decode.generic.utils.cum_count_per_group(pseudo_id)

        # ontime since start, to end
        t_since_start = (
            torch.repeat_interleave(t_start.ceil(), n_frames)
            + n_frame_per_emitter
            - torch.repeat_interleave(t_start, n_frames)
        )

        t_to_end = torch.repeat_interleave(t_end, n_frames) - (
            n_frame_per_emitter + torch.repeat_interleave(t_start.floor(), n_frames)
        )

        t_total_diff = torch.repeat_interleave(
            t_end, n_frames
        ) - torch.repeat_interleave(t_start, n_frames)

        ontime = t_since_start.minimum(t_to_end).minimum(t_total_diff).clamp(max=1)

        return n_frames, ontime

    def frame_bucketize(self) -> EmitterSet:
        """
        Returns EmitterSet with distributed emitters.
        The emitters ID is preserved such that localisations coming from the same
        fluorophore will have the same ID.

        Returns:
            EmitterSet
        """
        n_frames, ontime = self._compute_time_distribution(self.t0, self.te)

        em = EmitterSet(
            xyz=self.xyz,
            frame_ix=self.t0.floor().long(),
            phot=self.flux,
            id=self.id,
            xy_unit=self.xy_unit,
            px_size=self.px_size,
            **self._em_kwargs,
        ).repeat(n_frames, step_frames=True)

        # adjust photons by ontime (i.e. flux * ontime)
        em.phot *= ontime

        return em


def factory(n: Optional[int] = None, extent: float = 32.0, **kwargs) -> EmitterSet:
    """
    Generate a random EmitterSet

    Args:
        n: number of emitters in set. Can be omitted if length can be inferred from
            xyz, phot or frame_ix
        extent: spread in xyz
        **kwargs: arbitrary arguments to specify
    """
    # infer length from one of the specified kwargs
    inferrables = {"xyz", "phot", "frame_ix", "id"}
    if n is None:
        if len(set(kwargs.keys()).intersection(inferrables)) >= 1:
            n = [len(v) for k, v in kwargs.items() if k in inferrables][0]
        else:
            raise NotImplementedError(
                f"Length can only be inferred if one of {inferrables} is specified."
            )

    essentials = {
        "xyz": torch.rand(n, 3) * extent,
        "phot": torch.ones(n),
        "frame_ix": torch.zeros(n),
    }
    # overwrite essentials in case these are specified
    for k in set(essentials.keys()).intersection(kwargs.keys()):
        essentials[k] = kwargs.pop(k)

    return EmitterSet(**essentials, **kwargs)


@deprecated("deprecated in favor of factory", version="0.11")
class CoordinateOnlyEmitter(EmitterSet):
    """
    A helper class when we only want to provide xyz, but not photons and frame_ix.
    Useful for testing. Photons will be tensor of 1, frame_ix tensor of 0.
    """


@deprecated("deprecated in favor of factory", version="0.11")
class EmptyEmitterSet(CoordinateOnlyEmitter):
    """
    An empty emitter set.
    """
