import math
from abc import ABC, abstractmethod
import torch
from torch.nn import functional

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.generic.noise import GaussianSmoothing
from deepsmlm.generic.psf_kernel import ListPseudoPSF, DeltaPSF, OffsetPSF


class Preprocessing(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, in_tensor):
        return in_tensor.type(torch.FloatTensor)


class Identity(Preprocessing):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor, em_target=None):
        """

        :param in_tensor: input tensor
        :param em_target: (instance of emittersset)
        :return:
        """
        return super().forward(in_tensor)


class N2C(Preprocessing):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        in_tensor = super().forward(in_tensor)
        if in_tensor.shape[1] != 1:
            raise ValueError("Shape is wrong.")
        return in_tensor.transpose(0, 1).view(-1, in_tensor.shape[-2], in_tensor.shape[-1])


class EasyZ(Preprocessing):
    """
    A preprocessor class which includes the input image as well as the target for xy. This is useful when we want to
    test whether we can get other features or not.
    """
    def __init__(self, delta_psf):
        """
        :param delta_psf: psf to generate the xy tar helper
        """
        super().__init__()
        self.delta_psf = delta_psf

    def forward(self, in_tensor, em_target):
        input_ch = N2C().forward(in_tensor)
        input_ch = functional.interpolate(input_ch.unsqueeze(0), scale_factor=8, mode='nearest').squeeze(0)
        xy_ch = self.delta_psf.forward(em_target, weight=None)
        return torch.cat((input_ch, xy_ch), 0)


class TargetGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """

        :param x: input. Will be usually instance of emitterset.
        :return: target
        """
        return x


class OffsetRep(TargetGenerator):
    def __init__(self, xextent, yextent, zextent, img_shape, cat_output=True):
        super().__init__()
        self.delta = DeltaPSF(xextent,
                              yextent,
                              zextent,
                              img_shape,
                              photon_normalise=False)

        self.offset = OffsetPSF(xextent, yextent, img_shape)
        self.cat_out = cat_output

    def forward(self, x):
        """
        Create 5 channel output, decode_like
        :param x:
        :return: concatenated maps, or single maps with 1 x H x W each (channel order: p, I, x  y, z)
        """
        p_map = self.delta.forward(x, torch.ones_like(x.phot))
        """It might happen that we see that two emitters are in the same px. p_map will then be 2 or greater. 
        As the offset map allows for only one emitter, set the greater than 1 px to 1."""
        p_map[p_map > 1] = 1

        I_map = self.delta.forward(x, x.phot)
        xy_map = self.offset.forward(x)
        z_map = self.delta.forward(x, x.xyz[:, 2])

        if self.cat_out:
            return torch.cat((p_map, I_map, xy_map, z_map), 0)
        else:
            return p_map, I_map, xy_map[[0]], xy_map[[1]], z_map


class ZasOneHot(TargetGenerator):
    def __init__(self, delta_psf, kernel_size=5, sigma=0.8, scale_z=1/750.):
        super().__init__()
        self.delta_psf = delta_psf
        self.padding_same_v = math.ceil((kernel_size - 1) / 2)
        self.scale_z = scale_z

        def padding(x): return functional.pad(x, [self.padding_same_v,
                                                  self.padding_same_v,
                                                  self.padding_same_v,
                                                  self.padding_same_v], mode='reflect')

        self.gaussian_kernel = GaussianSmoothing(channels=1,
                                                 kernel_size=[kernel_size, kernel_size],
                                                 sigma=sigma,
                                                 dim=2,
                                                 cuda=False,
                                                 padding=padding,
                                                 kernel_f='gaussian')

        self.gaussian_kernel.kernel = self.gaussian_kernel.kernel * 57.6502

    def forward(self, x):
        z = self.delta_psf.forward(x, x.xyz[:, 2] * self.scale_z)
        z = self.gaussian_kernel.forward(z.unsqueeze(0)).squeeze(0)
        return z


class ZPrediction(ListPseudoPSF):
    def __init__(self):
        super().__init__(None, None, None, dim=3)

    def forward(self, x):
        pos, _ = super().forward(x)
        return pos[[0], 2]


class SingleEmitterOnlyZ(ListPseudoPSF):
    def __init__(self=3):
        super().__init__(None, None, None, dim=3)

    def forward(self, x):
        pos, phot = super().forward(x)
        return pos[[0], :], phot[[0]]


class ZasSimpleRegression(SingleEmitterOnlyZ):
    def __init__(self=3):
        super().__init__()

    def forward(self, x):
        # output
        z = super().forward(x)
        if z > 200:
            out = torch.tensor([1.])
        elif z < - 200:
            out = torch.tensor([-1.])
        else:
            out = torch.tensor([0.])

        return out.type(torch.FloatTensor)


class ZasClassification(SingleEmitterOnlyZ):
    def __init__(self=3):
        super().__init__()

    def forward(self, x):
        # output
        z = super().forward(x)
        if z > 0:
            out = torch.tensor([0])
        else:
            out = torch.tensor([1])

        return out.type(torch.LongTensor)


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
