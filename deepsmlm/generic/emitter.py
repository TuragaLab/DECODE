import os
import sys
import hashlib
import numpy as np
import torch

import torch_cpp


class EmitterSet:
    """
    Class, storing a set of emitters. Each attribute is a torch.Tensor.
    """
    def __init__(self, xyz, phot, frame_ix, id=None, prob=None):
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
        """
        self.num_emitter = int(xyz.shape[0]) if xyz.shape[0] != 0 else 0

        if self.num_emitter != 0:
            self.xyz = xyz if xyz.shape[1] == 3 else torch.cat((xyz, torch.zeros_like(xyz[:, [0]])), 1)
            self.phot = phot.type(xyz.dtype)
            self.frame_ix = frame_ix.type(xyz.dtype)
            self.id = id if id is not None else -torch.ones_like(frame_ix).type(xyz.dtype)
            self.prob = prob if prob is not None else torch.ones_like(frame_ix).type(xyz.dtype)

        else:
            self.xyz = torch.zeros((0, 3), dtype=torch.float)
            self.phot = torch.zeros((0,), dtype=torch.float)
            self.frame_ix = torch.zeros((0,), dtype=torch.float)
            self.id = -torch.ones((0,), dtype=torch.float)
            self.prob = torch.ones((0,), dtype=torch.float)

        self._sorted = False

        self._sanity_check()

    def _sanity_check(self):
        """
        Tests the integrity of the EmitterSet
        :return: None
        """
        if not ((self.xyz.shape[0] == self.phot.shape[0])
                and (self.phot.shape[0] == self.frame_ix.shape[0])
                and (self.frame_ix.shape[0] == self.id.shape[0])
                and (self.id.shape[0] == self.prob.shape[0])):
            raise ValueError("Coordinates, photons, frame ix, id and prob are not of equal shape in 0th dimension.")

    def clone(self):
        """
        Clone method to generate a deep copy.
        :return: Deep copy of self.
        """
        return EmitterSet(self.xyz.clone(),
                          self.phot.clone(),
                          self.frame_ix.clone(),
                          self.id.clone(),
                          self.prob.clone())

    @staticmethod
    def cat_emittersets(emittersets, remap_frame_ix=None, step_frame_ix=None):
        """
        Concatenates list of emitters and rempas there frame indices if they start over with 0 per item in list.

        :param emittersets: iterable of instances of this class
        :param remap_frame_ix: tensor of frame indices to which the 0th frame index in the emitterset corresponds to
        :param step_frame_ix: step of frame indices between items in list
        :return: emitterset
        """
        num_emittersets = emittersets.__len__()

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
            total_num_emitter += emittersets[i].num_emitter

        xyz = torch.cat([emittersets[i].xyz for i in range(num_emittersets)], 0)
        phot = torch.cat([emittersets[i].phot for i in range(num_emittersets)], 0)
        frame_ix = torch.cat([emittersets[i].frame_ix + shift[i] for i in range(num_emittersets)], 0)
        id = torch.cat([emittersets[i].id for i in range(num_emittersets)], 0)
        prob = torch.cat([emittersets[i].prob for i in range(num_emittersets)], 0)

        return EmitterSet(xyz, phot, frame_ix, id, prob)

    def sort_by_frame(self):
        self.frame_ix, ix = self.frame_ix.sort()
        self.xyz = self.xyz[ix, :]
        self.phot = self.phot[ix]
        self.id = self.id[ix]
        self.prob = self.prob[ix]

        self._sorted = True

    def get_subset(self, ix):
        return EmitterSet(self.xyz[ix, :], self.phot[ix], self.frame_ix[ix], self.id[ix], self.prob[ix])

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
                                      self.id[ix].unsqueeze(1),
                                      self.prob[ix].unsqueeze(1)), dim=1)
        else:
            raise DeprecationWarning("No Id is not supported any more.")

        """The first frame is assumed to be 0. If it's negative go to the lowest negative."""
        if self.num_emitter != 0:
            ix_low_ = ix_low if ix_low is not None else frame_ix.min()
            ix_up_ = ix_up if ix_up is not None else frame_ix.max()

            # if not np.diff(frame_ix.numpy()) >= 0:
            #     raise ValueError("Array is not sorted even though it is supposed to be.")

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
            em_list.append(EmitterSet(xyz=em[:, :3], phot=em[:, 3], frame_ix=em[:, 4], id=em[:, 5], prob=em[:, 6]))

        return em_list

    def write_to_csv(self, filename, model=None, comments=None):
        """
        Write the prediction to a csv file.
        :param filename: output filename
        :param model: model file which was being used (will create a hash out of it)
        :return:
        """
        grand_matrix = torch.cat((self.id.unsqueeze(1),
                                  self.frame_ix.unsqueeze(1),
                                  self.xyz,
                                  self.phot.unsqueeze(1),
                                  self.prob.unsqueeze(1)), 1)
        header = 'id, frame_ix, x, y, z, phot, prob\nThis is an export from DeepSMLM.\n' \
                 'Total number of emitters: {}'.format(self.num_emitter)

        if model is not None:
            if hasattr(model, 'hash'):
                header += '\n Model initialisation file SHA-1 hash: {}'.format({model.hash})

        if comments is not None:
            header += '\nUser comment during export: {}'.format(comments)
        np.savetxt(filename, grand_matrix.numpy(), delimiter=',', header=header)


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


class EmptyEmitterSet(CoordinateOnlyEmitter):
    """An empty emitter set."""
    def __init__(self):
        super().__init__(torch.zeros((0, 3)))


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
