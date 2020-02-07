import warnings
import numpy as np
import torch

import torch_cpp
from .utils import test_utils as tutil


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


class EmitterSet:
    """
    Class, storing a set of emitters. Each attribute is a torch.Tensor.
    """
    eq_precision = 1E-8
    xy_units = ('px', 'nm')

    def __init__(self, xyz, phot, frame_ix, id=None, prob=None, bg=None, xyz_cr=None, phot_cr=None, bg_cr=None,
                 sanity_check=True, xy_unit=None, px_size=None):
        """
        Constructor. Coordinates, photons, frame_ix must be provided. Id is optional.

        :param xyz: torch.Tensor of size N x 3 (2). x, y, z are in arbitrary units.
        often [x] in px, [y] in px, [z] in nm.
        :param phot: torch.Tensor of size N. number of photons.
        :param frame_ix: integer or torch.Tensor. If it's one element, the whole set belongs to
        the frame, if it's a tensor, it must be of length N.
        :param id: torch.Tensor of size N. id of an emitter. -1 is an arbitrary non uniquely used
         fallback id.
        :param prob: torch.Tensor of size N. probability of observation. will be 1 by default.
        :param bg: constant assumed background value, or N x {Background - Dim}
        :param xyz_cr: Cramer Rao Bound of xyz
        :param phot_cr: Cramer Rao of phot
        :param bg_cr: Cramer Rao of background
        """
        num_emitter_input = int(xyz.shape[0]) if xyz.shape[0] != 0 else 0

        if num_emitter_input != 0:
            self.xyz = xyz if xyz.shape[1] == 3 else torch.cat((xyz, torch.zeros_like(xyz[:, [0]])), 1)
            self.phot = phot.type(xyz.dtype)
            self.frame_ix = frame_ix.type(xyz.dtype)
            self.id = id if id is not None else -torch.ones_like(frame_ix).type(xyz.dtype)
            self.prob = prob if prob is not None else torch.ones_like(frame_ix).type(xyz.dtype)
            self.bg = bg if bg is not None else float('nan') * torch.ones_like(frame_ix).type(xyz.dtype)
            self.xyz_cr = xyz_cr if xyz_cr is not None else float('nan') * torch.ones_like(self.xyz).type(xyz.dtype)
            self.phot_cr = phot_cr if phot_cr is not None else float('nan') * torch.ones_like(self.phot).type(xyz.dtype)
            self.bg_cr = bg_cr if bg_cr is not None else float('nan') * torch.ones_like(self.bg).type(xyz.dtype)

        else:
            self.xyz = torch.zeros((0, 3), dtype=torch.float)
            self.phot = torch.zeros((0,), dtype=torch.float)
            self.frame_ix = torch.zeros((0,), dtype=torch.float)
            self.id = -torch.ones((0,), dtype=torch.float)
            self.prob = torch.ones((0,), dtype=torch.float)
            self.bg = float('nan') * torch.ones_like(self.prob)
            self.xyz_cr = float('nan') * torch.ones((0, 3), dtype=torch.float)
            self.phot_cr = float('nan') * torch.ones_like(self.prob)
            self.bg_cr = float('nan') * torch.ones_like(self.prob)

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
    def num_emitter(self):
        # warnings.warn("This will be soon deprecated. Use len() instead.", DeprecationWarning, stacklevel=0)
        # return int(self.xyz.shape[0]) if self.xyz.shape[0] != 0 else 0
        raise DeprecationWarning

    @property
    def xyz_px(self):
        if self.xy_unit is None:
            warnings.warn("If unit is unspecified, can not convert to px coordinates.")
            return
        elif self.xy_unit == 'nm':
            if self.px_size is None:
                raise ValueError("Cannot convert between px and nm without px-size specified.")
            return self.convert_coordinates(factor=1/self.px_size)
        elif self.xy_unit == 'px':
            return self.xyz

    @xyz_px.setter
    def xyz_px(self, xyz):
        self.xyz = xyz
        self.xy_unit = 'px'

    @property
    def xyz_nm(self):
        if self.xy_unit is None:
            warnings.warn("If unit is unspecified, can not convert to px coordinates.")
            return
        elif self.xy_unit == 'px':
            if self.px_size is None:
                raise ValueError("Cannot convert between px and nm without px-size specified.")
            return self.convert_coordinates(factor=self.px_size)
        elif self.xy_unit == 'nm':
            return self.xyz

    @xyz_nm.setter
    def xyz_nm(self, xyz):
        self.xyz = xyz
        self.xy_unit = 'nm'

    @property
    def xyz_scr(self):  # sqrt crlb
        return self.xyz_cr.sqrt()

    @property
    def xyz_nm_scr(self):
        if self.px_size is None:
            raise ValueError("Cannot convert between px and nm without px-size specified.")
        return self.convert_coordinates(factor=self.px_size, xyz=self.xyz_scr)

    @property
    def phot_scr(self):
        return self.phot_cr.sqrt()

    @property
    def bg_scr(self):
        return self.bg_cr.sqrt()

    def _inplace_replace(self, em):
        """If self is derived class of EmitterSet, call the constructor of the parent instead of self.
        However, I don't know why super().__init__(...) does not work."""
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
        Tests the integrity of the EmitterSet
        :return: None
        """
        if not same_shape_tensor(0, self.xyz, self.phot, self.frame_ix, self.id, self.bg,
                                 self.xyz_cr, self.phot_cr, self.bg_cr):
            raise ValueError("Coordinates, photons, frame ix, id and prob are not of equal shape in 0th dimension.")

        if not same_dim_tensor(torch.ones(1), self.phot, self.prob, self.frame_ix, self.id):
            raise ValueError("Expected photons, probability frame index and id to be 1D.")

        if self.xy_unit is not None:
            if self.xy_unit not in self.xy_units:
                warnings.warn("XY unit not known.")

        # check uniqueness of ID
        if check_uniqueness:
            if torch.unique(self.id).numel() != self.id.numel():
                raise ValueError("IDs are not unique.")

    def __len__(self):
        return int(self.xyz.shape[0]) if self.xyz.shape[0] != 0 else 0

    def __str__(self):
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
        Two emittersets are equal if all of it's (tensor) members are equal within a certain float precision
        :param other: EmitterSet
        :return: (bool)
        """
        is_equal = True
        is_equal *= tutil.tens_almeq(self.xyz, other.xyz, self.eq_precision)
        is_equal *= tutil.tens_almeq(self.frame_ix, other.frame_ix, self.eq_precision)
        is_equal *= tutil.tens_almeq(self.phot, other.phot, self.eq_precision)
        # is_equal *= tutil.tens_almeq(self.id, other.id, self.eq_precision)  # this is in question
        is_equal *= tutil.tens_almeq(self.prob, other.prob, self.eq_precision)

        if torch.isnan(self.bg).all():
            is_equal *= torch.isnan(other.bg).all()
        else:
            is_equal *= tutil.tens_almeq(self.bg, other.bg, self.eq_precision)
        if torch.isnan(self.xyz_cr).all():
            is_equal *= torch.isnan(other.xyz_cr).all()
        else:
            is_equal *= tutil.tens_almeq(self.bg, other.bg, self.eq_precision)
        if torch.isnan(self.phot_cr).all():
            is_equal *= torch.isnan(other.phot_cr).all()
        else:
            is_equal *= tutil.tens_almeq(self.bg, other.bg, self.eq_precision)
        if torch.isnan(self.bg_cr).all():
            is_equal *= torch.isnan(other.bg_cr).all()
        else:
            is_equal *= tutil.tens_almeq(self.bg, other.bg, self.eq_precision)

        return is_equal.item()

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
        Implements array indexing for Emitterset

        Args:
            item: (int), or indexing

        Returns:
            (EmitterSet)
        """
        if item >= len(self):
            raise IndexError(f"Index {item} out of bounds of EmitterSet of size {len(self)}")

        return self.get_subset(item)

    def clone(self):
        """
        Make a deep copy of this EmitterSet.

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
    def cat_emittersets(emittersets, remap_frame_ix=None, step_frame_ix=None):
        """
        Concatenates list of emitters and rempas there frame indices if they start over with 0 per item in list.

        :param emittersets: iterable of instances of this class
        :param remap_frame_ix: tensor of frame indices to which the 0th frame index in the emitterset corresponds to
        :param step_frame_ix: step of frame indices between items in list
        :return: emitterset
        """
        num_emittersets = len(emittersets)

        if remap_frame_ix is not None and step_frame_ix is not None:
            raise ValueError("You cannot specify remap frame ix and step frame ix at the same time.")
        elif remap_frame_ix is not None:
            shift = remap_frame_ix.clone()
        elif step_frame_ix is not None:
            shift = torch.arange(0, num_emittersets) * step_frame_ix
        else:
            shift = torch.zeros(num_emittersets)

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

    def sort_by_frame(self):
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

    def get_subset(self, ix):
        """
        Returns subset of emitterset. Main implementation of __getitem__ and __next__ methods.
        Args:
            ix: (int, list) integer index or list of indices

        Returns:
            (EmitterSet)
        """
        if isinstance(ix, int):
            ix = [ix]

        return EmitterSet(self.xyz[ix, :], self.phot[ix], self.frame_ix[ix], self.id[ix], self.prob[ix], self.bg[ix],
                          self.xyz_cr[ix], self.phot_cr[ix], self.bg_cr[ix], sanity_check=False,
                          xy_unit=self.xy_unit, px_size=self.px_size)

    def get_subset_frame(self, frame_start, frame_end, shift_to=None):
        """
        Get Emitterset for a certain frame range.
        Inclusive behaviour, so start and end are included.

        :param frame_start: (int)
        :param frame_end: (int)
        :shift_to: shift frame indices to a certain start value
        """

        ix = (self.frame_ix >= frame_start) * (self.frame_ix <= frame_end)
        em = self.get_subset(ix)
        if not shift_to:
            return em
        else:
            if em.num_emitter != 0:  # shifting makes only sense if we have an emitter.
                em.frame_ix = em.frame_ix - em.frame_ix.min() + shift_to
            return em

    @property
    def single_frame(self):
        return True if torch.unique(self.frame_ix).shape[0] == 1 else False

    def compute_grand_matrix(self, ix=None, xy_unit=None, xy_unit2=None):
        """
        Computes a grand matrix to put everything in one tensor.
        Args:
            ix: limit the indices
            xy_unit: xy unit to write to the matrix
            xy_unit2: secondary xy unit to append to the back of the grand matrix, helpful for csv export, but redundant info

        Returns:
            grand_matrix: (torch.Tensor) of size num_emitter x large_number
        """

        def xy_mapping(unit):
            if unit is None:
                return self.xyz
            elif unit == 'px':
                return self.xyz_px
            elif xy_unit == 'nm':
                return self.xyz_nm
            else:
                raise ValueError("Other units not supported.")

        if ix is None:
            ix = slice(self.xyz.size(0))

        xyz = xy_mapping(xy_unit)
        xyz2 = xy_mapping(xy_unit2)  # secondary xy unit (it's redundant but sometimes helpful)

        grand_matrix = torch.cat((xyz[ix, :],
                                  self.phot[ix].unsqueeze(1),
                                  self.frame_ix[ix].unsqueeze(1),
                                  self.id[ix].unsqueeze(1),
                                  self.prob[ix].unsqueeze(1),
                                  self.bg[ix].unsqueeze(1),
                                  self.xyz_cr[ix, :],
                                  self.phot_cr[ix].unsqueeze(1),
                                  self.bg_cr[ix].unsqueeze(1),
                                  xyz2[ix, :]), dim=1)

        return grand_matrix

    @staticmethod
    def _construct_from_grand_matrix(gmat):
        return EmitterSet(xyz=gmat[:, :3], phot=gmat[:, 3], frame_ix=gmat[:, 4], id=gmat[:, 5], prob=gmat[:, 6],
                          bg=gmat[:, 7], xyz_cr=gmat[:, 8:11], phot_cr=gmat[:, 11], bg_cr=gmat[:, 12],
                          sanity_check=False)

    def split_in_frames(self, ix_low=0, ix_up=None):
        """
        plit an EmitterSet into list of emitters (framewise).
        This calls C++ implementation torch_cpp for performance.
        If neither lower nor upper are inferred (via None values),
        output size will be a list of length (ix_up - ix_low + 1).
        If we have an empty set of emitters which we want to split, we get a one-element empty
        emitter 

        :param ix_low: (int) lower index, if None, use min value
        :param ix_up: (int) upper index, if None, use max value
        :return: list of instances of this class.
        """

        frame_ix, ix = self.frame_ix.sort()
        frame_ix = frame_ix.type(self.xyz.dtype)

        if self.id is not None:
            grand_matrix = torch.cat((self.xyz[ix, :],
                                      self.phot[ix].unsqueeze(1),
                                      frame_ix.unsqueeze(1),
                                      self.id[ix].unsqueeze(1),
                                      self.prob[ix].unsqueeze(1),
                                      self.bg[ix].unsqueeze(1),
                                      self.xyz_cr[ix, :],
                                      self.phot_cr[ix].unsqueeze(1),
                                      self.bg_cr[ix].unsqueeze(1)), dim=1)
        else:
            raise DeprecationWarning("No Id is not supported any more.")

        """The first frame is assumed to be 0. If it's negative go to the lowest negative."""
        if len(self) != 0:
            ix_low_ = ix_low if ix_low is not None else frame_ix.min()
            ix_up_ = ix_up if ix_up is not None else frame_ix.max()

            grand_matrix_list = torch_cpp.split_tensor(grand_matrix, frame_ix, ix_low_, ix_up_)

        else:
            """
            If there is absolutelty nothing to split (i.e. empty emitterset) we may want to have a list of
            empty sets of emitters. This only applies if ix_l is not inferred (i.e. -1).
            Otherwise we will have a one element list with an empty emitter set.
            """
            if ix_low is None:
                grand_matrix_list = [grand_matrix]
            else:
                grand_matrix_list = [grand_matrix] * (ix_up - ix_low + 1)
        em_list = []

        for i, em in enumerate(grand_matrix_list):
            em_list.append(EmitterSet(xyz=em[:, :3], phot=em[:, 3], frame_ix=em[:, 4], id=em[:, 5], prob=em[:, 6],
                                      bg=em[:, 7], xyz_cr=em[:, 8:11], phot_cr=em[:, 11], bg_cr=em[:, 12],
                                      sanity_check=False, xy_unit=self.xy_unit, px_size=self.px_size))

        return em_list

    def convert_coordinates(self, factor=None, shift=None, axis=None, xyz=None):
        """
        Convert coordinates. The order is factor -> shift -> axis
        :param factor: scale up factor
        :param shift: shift
        :param axis: permute axis
        :param xyz: overwrite xyz tensor (e.g. for using it with crlb)
        :return:
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

    def convert_em(self, factor=None, shift=None, axis=None, frame_shift=None, new_xy_unit=None):
        """
        Convert a clone of the current emitter
        :param factor:
        :param shift:
        :param axis:
        :param frame_shift:
        :param new_xy_unit: set the name of the unit
        :return:
        """

        emn = self.clone()
        emn.xyz = emn.convert_coordinates(factor, shift, axis)
        if frame_shift is not None:
            emn.frame_ix += frame_shift
        if new_xy_unit is not None:
            emn.xy_unit = new_xy_unit
        return emn

    def convert_em_(self, factor=None, shift=None, axis=None, frame_shift=0, new_xy_unit=None):
        """
        Inplace conversion of emitter
        :param factor:
        :param shift:
        :param axis:
        :param frame_shift:
        :param new_xy_unit: set the name of the xy unit
        :return:
        """
        self.xyz = self.convert_coordinates(factor, shift, axis)
        self.frame_ix += frame_shift
        if new_xy_unit is not None:
            self.xy_unit = new_xy_unit

    def write_to_binary(self, filename):
        """
        Writes the "grand_matrix" representation to a binary using pickle.

        Args:
            filename: output file

        Returns:

        """

        gmat = self.compute_grand_matrix().clone()
        torch.save(gmat, filename)

    @staticmethod
    def load_from_binary(filename):
        """
        Loads a grand matrix and converts it into an EmitterSet
        
        Args:
            filename:

        Returns:
            EmitterSet without px size and xy unit

        """
        gmat = torch.load(filename)
        return EmitterSet._construct_from_grand_matrix(gmat)

    def write_to_csv(self, filename, xy_unit=None, model=None, comments=None, plain_header=False, xy_unit2=None):
        """
        Writes the emitterset to a csv file.

        Args:
            filename:  csv file name
            xy_unit:  xy unit (typically 'nm' or 'px') for automatic conversion
            model:  model to incroporate hash into the csv
            comments:  additional comments to put into the csv
            plain_header: no # at beginning of first line

        Returns:
            grand_matrix: (torch.Tensor) the matrix which was effectively written to csv

        """

        grand_matrix = self.compute_grand_matrix(xy_unit=xy_unit, xy_unit2=xy_unit2)

        # reorder the grand_matrix according to the header below
        grand_matrix = grand_matrix[:, [5, 4, 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

        header = 'id, frame_ix, x, y, z, phot, prob, bg, x_cr, y_cr, z_cr, phot_cr, bg_cr, x2, y2, z2' \
                 '\nThis is an export from ' \
                 'DeepSMLM.\n' \
                 'Total number of emitters: {}'.format(len(self))

        if model is not None:
            if hasattr(model, 'hash'):
                header += '\nModel initialisation file SHA-1 hash: {}'.format(model.hash)

        if comments is not None:
            header += '\nUser comment during export: {}'.format(comments)
        np.savetxt(filename, grand_matrix.numpy(), delimiter=',', header=header)

        if plain_header:
            with open(filename, "r+") as f:
                content = f.read()  # read everything in the file
                content = content[2:]  # remove hashtag at first line (matlab ...)

            with open(filename, "w") as f:
                f.write(content)  # write the file again

        return grand_matrix

    def write_to_csv_format(self, filename, xy_unit=None, xy_unit2=None, model=None, comments=None,
                            xyz_shift=None, frame_shift=None, axis=None, plain_header=False, lud=None, lud_name=None):
        """
        Transforms the data and writes it to a csv.
        Args:
            filename:
            xy_unit:
            xy_unit2: secondary xy unit
            model:
            comments:
            xyz_shift:
            frame_shift:
            axis:
            plain_header: (bool) remove hashtag in first line of csv
            lud: look-up-dictionary, replaces the need for all keyword-arguments and uses a predefined dictionary
            lud_name: name of a look-up-dictionary predifined in the code base.

        Returns:
            (None)
        """
        import deepsmlm.generic.inout.csv_transformations as tra
        """Checks before actual run"""
        # lud and lud_name are XOR
        if lud is not None and lud_name is not None:
            raise ValueError("You can not specify lud and lud_name at the same time.")

        if (lud is not None or lud_name is not None) and (xyz_shift is not None or axis is not None):
            raise ValueError("You can not specify factor, shift, axis and lud / lud_name at the same time.")

        if lud_name is not None:
            lud = tra.pre_trafo[lud_name]
        if lud_name is not None or lud is not None:
            xyz_shift = torch.tensor(lud['xyz_shift'])
            xy_unit = lud['xy_unit']
            xy_unit2 = lud['xy_unit2']
            frame_shift = lud['frame_shift']
            axis = lud['axis']
            plain_header = lud['plain_header']
            comments = lud['comments']

        em_clone = self.convert_em(factor=None,
                                   shift=xyz_shift,
                                   axis=axis,
                                   frame_shift=frame_shift,
                                   new_xy_unit=None)

        em_clone.write_to_csv(filename, xy_unit, model, comments, plain_header=plain_header, xy_unit2=xy_unit2)

    @staticmethod
    def read_csv(filename):
        grand_matrix = np.loadtxt(filename, delimiter=",", comments='#')
        grand_matrix = torch.from_numpy(grand_matrix).float()
        if grand_matrix.size(1) == 7:
            return EmitterSet(xyz=grand_matrix[:, 2:5], frame_ix=grand_matrix[:, 1],
                              phot=grand_matrix[:, 5], id=grand_matrix[:, 0],
                              prob=grand_matrix[:, 6])
        else:
            return EmitterSet(xyz=grand_matrix[:, 2:5], frame_ix=grand_matrix[:, 1],
                              phot=grand_matrix[:, 5], id=grand_matrix[:, 0])

    def populate_crlb(self, psf, mode='multi'):
        """
        Calculate the CRLB
        :return:
        """
        if len(self) == 0:
            return

        if mode == 'single':
            crlb, _ = psf.crlb_single(self.xyz_px, self.phot, self.bg, crlb_order='xyzpb')
            self.xyz_cr = crlb[:, :3]
            self.phot_cr = crlb[:, 3]
            self.bg_cr = crlb[:, 4]
            return

        elif mode == 'multi':
            warnings.warn(
                "Be advised, that at the moment, calculating the crlb in multi-mode for an EmitterSet can and most "
                "likely will change the order of the elements in the set."
                "If you compare this against another set, be careful.")
            em_split = self.split_in_frames(self.frame_ix.min(), self.frame_ix.max())

            for em in em_split:

                crlb, _ = psf.crlb(em.xyz_px, em.phot, em.bg, crlb_order='xyzpb')

                em.xyz_cr = crlb[:, :3]
                em.phot_cr = crlb[:, 3]
                em.bg_cr = crlb[:, 4]

            remerged_set = self.cat_emittersets(em_split)
            self._inplace_replace(remerged_set)
            return

        else:
            raise ValueError("Mode must be single or multi.")


class RandomEmitterSet(EmitterSet):
    """
    A helper calss when we only want to provide a number of emitters.
    """
    def __init__(self, num_emitters, extent=32):
        """

        :param num_emitters:
        """
        xyz = torch.rand((num_emitters, 3)) * extent
        super().__init__(xyz, torch.ones_like(xyz[:, 0]), torch.zeros_like(xyz[:, 0]))

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
    def __init__(self, xyz):
        """

        :param xyz: (torch.tensor) N x 2, N x 3
        """
        super().__init__(xyz, torch.ones_like(xyz[:, 0]), torch.zeros_like(xyz[:, 0]))
    
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
    def __init__(self):
        super().__init__(torch.zeros((0, 3)))

    def _inplace_replace(self, em):
        raise NotImplementedError("Inplace not yet implemented.")


class LooseEmitterSet:
    """
    An emitterset where we don't specify the frame_ix of an emitter but rather it's (real) time when
    it's starts to blink and it's ontime and then construct the EmitterSet (framewise) out of it.
    """
    def __init__(self, xyz, intensity, id=None, t0=None, ontime=None):
        """

        :param xyz: Coordinates
        :param phot: Photons
        :param intensity: Intensity (i.e. photons per time)
        :param id: ID
        :param t0: Timepoint of first occurences
        :param ontime: Duration in frames how long the emitter is on.
        """

        """If no ID specified, give them one."""
        if id is None:
            id = torch.arange(xyz.shape[0])

        self.xyz = xyz
        self.phot = None
        self.intensity = intensity
        self.id = id
        self.t0 = t0
        self.te = None
        self.ontime = ontime

    def return_emitterset(self):
        """
        Returns an emitter set

        :return: Instance of EmitterSet class.
        """
        xyz_, phot_, frame_ix_, id_ = self.distribute_framewise_py()
        return EmitterSet(xyz_, phot_, frame_ix_, id_)

    def distribute_framewise_py(self):
        """
        Distribute the emitters with arbitrary starting point and intensity over the frames so as to get a proper
        set of emitters (instance of EmitterSet) with photons.
        :return:
        """
        frame_start = torch.floor(self.t0)
        self.te = self.t0 + self.ontime  # endpoint
        frame_last = torch.ceil(self.te)
        frame_dur = (frame_last - frame_start).type(torch.LongTensor)

        num_emitter_brut = frame_dur.sum()

        xyz_ = torch.zeros(num_emitter_brut, self.xyz.shape[1])
        phot_ = torch.zeros_like(xyz_[:, 0])
        frame_ix_ = torch.zeros_like(phot_)
        id_ = torch.zeros_like(frame_ix_)

        c = 0
        for i in range(self.xyz.shape[0]):
            for j in range(frame_dur[i]):
                xyz_[c, :] = self.xyz[i]
                frame_ix_[c] = frame_start[i] + j
                id_[c] = self.id[i]

                """Calculate time on frame and multiply that by the intensity."""
                ontime_on_frame = torch.min(self.te[i], frame_ix_[c] + 1) - torch.max(self.t0[i], frame_ix_[c])
                phot_[c] = ontime_on_frame * self.intensity[i]

                c += 1
        return xyz_, phot_, frame_ix_, id_

    def _inplace_replace(self, em):
        raise NotImplementedError("Inplace not yet implemented.")


if __name__ == '__main__':
    num_emitter = 25000
    xyz = torch.rand((num_emitter, 3))
    phot = torch.ones_like(xyz[:, 0])
    t0 = torch.rand((num_emitter,)) * 10 - 1
    ontime = torch.rand((num_emitter,)) * 1.5

    LE = LooseEmitterSet(xyz, phot, None, t0, ontime)
    E = LE.return_emitterset()

    frame_ix = torch.zeros_like(xyz[:,0])
    em = EmitterSet(xyz, phot, frame_ix)
    em_splitted = em.split_in_frames(0, 0)

    print("Pseudo-Test successfull.")
