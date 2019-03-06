from abc import ABC, abstractmethod
import torch

from deepsmlm.generic.emitter import EmitterSet


class Preprocessing(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, in_tensor):
        return in_tensor.type(torch.FloatTensor)


class N2C(Preprocessing):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        in_tensor = super().forward(in_tensor)
        if in_tensor.shape[1] != 1:
            raise ValueError("Shape is wrong.")
        return in_tensor.transpose(0, 1).view(-1, in_tensor.shape[-2], in_tensor.shape[-1])


class RemoveOutOfFOV:
    def __init__(self, xextent, yextent):
        self.xextent = xextent
        self.yextent = yextent

    def clean_emitter(self, em_mat):

        is_emit = (em_mat[:, 0] >= self.xextent[0]) * \
                  (em_mat[:, 0] < self.xextent[1]) * \
                  (em_mat[:, 1] >= self.yextent[0]) * \
                  (em_mat[:, 1] < self.yextent[1])

        return is_emit

    def clean_emitter_set(self, em_set):
        em_mat = em_set.xyz
        is_emit = self.clean_emitter(em_mat)

        return EmitterSet(xyz=em_set.xyz[is_emit, :],
                          phot=em_set.phot[is_emit],
                          frame_ix=em_set.frame_ix[is_emit],
                          id=(None if em_set.id is None else em_set.id[is_emit]))
