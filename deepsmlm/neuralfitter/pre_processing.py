import torch

from deepsmlm.generic.emitter import EmitterSet


class RemoveOutOfFOV:
    def __init__(self, xextent, yextent):
        self.xextent = xextent
        self.yextent = yextent

    def clean_emitter(self, em_mat):

        is_emit = torch.mul((em_mat[:, [0]] >= self.xextent[0]).all(1),
                            (em_mat[:, [0]] < self.xextent[1]).all(1),
                            (em_mat[:, [1]] >= self.yextent[0]).all(1),
                            (em_mat[:, [1]] < self.yextent[1]).all(1))

        return is_emit

    def clean_emitter_set(self, em_set):
        em_mat = em_set.xyz
        is_emit = self.clean_emitter(em_mat)

        return EmitterSet(xyz=em_set.xyz[is_emit, :],
                          phot=em_set.phot[is_emit],
                          frame_ix=em_set[is_emit],
                          id=(None if em_set.id is None else em_set_id[is_emit]))
