import os
import sys
import torch

import torch_cpp


class EmitterSet:
    """
    Class, storing a set of emitters. Each attribute is a torch.Tensor.
    """
    def __init__(self, xyz, phot, frame_ix, id=None):
        """
        Constructor. Coordinates, photons, frame_ix must be provided. Id is optional.

        :param xyz: torch.Tensor of size N x 3 (2). x, y, z are in arbitrary units.
        often [x] in px, [y] in px, [z] in nm.
        :param phot: torch.Tensor of size N. number of photons.
        :param frame_ix: integer or torch.Tensor. If it's one element, the whole set belongs to
        the frame, if it's a tensor, it must be of length N.
        :param id: torch.Tensor of size N. id of an emitter. -1 is an arbitrary non uniquely used
         fallback id.
        """
        self.num_emitter = int(xyz.shape[0]) if xyz.shape[0] != 0 else 0

        if self.num_emitter != 0:
            self.xyz = xyz if xyz.shape[1] == 3 else torch.cat((xyz, torch.zeros_like(xyz[:, [0]])), 1)
            self.phot = phot.type(xyz.dtype)
            self.frame_ix = frame_ix.type(xyz.dtype)
            self.id = id if id is not None else -torch.ones_like(frame_ix).type(xyz.dtype)

        else:
            self.xyz = torch.zeros((0, 3), dtype=torch.float)
            self.phot = torch.zeros((0,), dtype=torch.float)
            self.frame_ix = torch.zeros((0,), dtype=torch.float)
            self.id = -torch.ones((0,), dtype=torch.float)

        self._sorted = False

        self._sanity_check()

    def _sanity_check(self):
        """
        Tests the integrity of the EmitterSet
        :return: None
        """
        if not ((self.xyz.shape[0] == self.phot.shape[0])
                and (self.phot.shape[0] == self.frame_ix.shape[0])
                and (self.frame_ix.shape[0] == self.id.shape[0])):
            raise ValueError("Coordinates, photons, frame ix and id are not of equal shape in 0th dimension.")

    def sort_by_frame(self):
        self.frame_ix, ix = self.frame_ix.sort()
        self.xyz = self.xyz[ix, :]
        self.phot = self.phot[ix]
        self.id = self.id[ix]

        self._sorted = True

    def get_subset(self, ix):
        return EmitterSet(self.xyz[ix, :], self.phot[ix], self.frame_ix[ix], self.id[ix])

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
                                      self.id[ix].unsqueeze(1)), dim=1)
        else:
            raise ValueError("No Id is not supported any more.")

        """The first frame is assumed to be 0. If it's negative go to the lowest negative."""
        if self.num_emitter != 0:
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
            em_list.append(EmitterSet(xyz=em[:, :3], phot=em[:, 3], frame_ix=em[:, 4], id=em[:, 5]))

        return em_list


class LooseEmitterSet:
    """
    An emitterset where we don't specify the frame_ix of an emitter but rather it's (real) time when
    it's starts to blink and it's ontime and then construct the EmitterSet (framewise) out of it.
    """
    def __init__(self, xyz, phot, id=None, t0=None, ontime=None):
        """

        :param xyz: Coordinates
        :param phot: Photons
        :param id: ID
        :param t0: Timepoint of first occurences
        :param ontime: Duration in frames how long the emitter is on.
        """

        """If no ID specified, give them one."""
        if id is None:
            id = torch.arange(xyz.shape[0])

        self.xyz = xyz
        self.phot = phot
        self.id = id
        self.t0 = t0
        self.te = None
        self.ontime = ontime

    def return_emitterset(self):
        """
        Returns an emitter set

        :return: Instance of EmitterSet class.
        """
        xyz_, phot_, frame_ix_, id_ = self.distribute_framewise()
        return EmitterSet(xyz_, phot_, frame_ix_, id_)

    def distribute_framewise(self):
        """
        Wrapper to call C++ function to distribute the stuff over the frames.
        Unfortunately this does not seem to be way faster than the Py version ...

        :return: coordinates, photons, frame_ix, _id where for every frame-ix
        """
        _xyz, _phot, _frame_ix, _id = torch_cpp.distribute_frames(self.t0,
                                                                  self.ontime,
                                                                  self.xyz,
                                                                  self.phot,
                                                                  self.id)
        return _xyz, _phot, _frame_ix, _id

    def distribute_framewise_py(self):
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

                """Split photons linearly across the frames. This is an approximation."""
                ontime_on_frame = torch.min(self.te[i], frame_ix_[c] + 1) - torch.max(self.t0[i], frame_ix_[c])
                phot_[c] = ontime_on_frame / self.ontime[i] * self.phot[i]

                c += 1
        return xyz_, phot_, frame_ix_, id_


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
