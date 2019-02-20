import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp_source/libtorchInterface'))
import torch_cpp


class EmitterSet:
    """
    Struct-like class, storing a set of emitters. Each attribute is a torch.Tensor.
    It can be constructed from a binary.
    """
    def __init__(self, xyz, phot, frame_ix, id=None):
        """
        Constructor

        :param xyz: torch.Tensor of size N x 3. x, y are in px units; z in units of nm (hence the name).
        :param phot: torch.Tensor of size N. number of photons.
        :param frame_ix: integer or torch.Tensor. If it's one element, the whole set belongs to the frame,
            if it's a tensor, it must be of length N.
        :param id: torch.Tensor of size N. id of an emitter.
        """
        self.num_emitter = int(xyz.shape[0]) if xyz.shape[0] != 0 else 0

        if self.num_emitter != 0:
            self.xyz = xyz
            self.phot = phot.type(xyz.dtype)
            self.frame_ix = frame_ix.type(xyz.dtype)
            self.id = id

        else:
            self.xyz = None
            self.phot = None
            self.frame_ix = None
            self.id = None

    @property
    def single_frame(self):
        return True if torch.unique(self.frame_ix).shape[0] == 1 else False

    def split_in_frames(self):
        """
        Recursive function to split an EmitterSet into list of emitters per frame. This calls torch_cpp for performance.

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
            grand_matrix = torch.cat((self.xyz[ix, :],
                                      self.phot[ix].unsqueeze(1),
                                      frame_ix.unsqueeze(1)), dim=1)

        grand_matrix_list = torch_cpp.split_tensor(grand_matrix, frame_ix, 0, -1)
        em_list = []

        if self.id is not None:
            for i, em in enumerate(grand_matrix_list):
                em_list.append(EmitterSet(xyz=em[:, :3],
                                        phot=em[:, 3],
                                        frame_ix=em[:, 4],
                                        id=em[:, 5]))
        else:
            for i, em in enumerate(grand_matrix_list):
                em_list.append(EmitterSet(xyz=em[:, :3],
                                        phot=em[:, 3],
                                        frame_ix=em[:, 4],
                                        id=None))

        return em_list
