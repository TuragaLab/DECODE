import math
from abc import ABC, abstractmethod
import torch
import torch.nn
import numpy as np
from torch.nn import functional
from sklearn import neighbors, datasets
import warnings

from deepsmlm.generic.coordinate_trafo import A2BTransform
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


class DiscardBackground(Preprocessing):
    """
    A simple class which discards the background which comes out of the simulator because this will be target not input.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        If x is tuple, second element (bg) will be discarded. If not nothing happens.
        :param x: tuple or tensor.
        :return: tensor
        """
        if not isinstance(x, torch.Tensor):
            return x[0]
        else:
            return x


class N2C(Preprocessing):
    """
    Change from Batch to channel dimension.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """

        :param x: input. tensor or tuple / list of tensors.
        :return:
        """
        if isinstance(x, tuple) or isinstance(x, list):
            out = [None] * x.__len__()
            for i in range(x.__len__()):
                out[i] = self.forward(x[i])
            return out

        in_tensor = super().forward(x)
        if in_tensor.shape[1] != 1:
            raise ValueError("Shape is wrong.")
        return in_tensor.squeeze(1)


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
    def forward(self, x_in, weight):
        """

        :param x: input. Will be usually instance of emitterset.
        :return: target
        """
        if isinstance(x_in, EmitterSet):
            pos = x_in.xyz
            if weight is None:
                weight = x_in.phot
            return pos, weight
        else:
            return x_in, weight


class CombineTargetBackground(TargetGenerator):
    def __init__(self, target_seq, num_input_frames: int):
        """

        :param target_seq: target generator or transform sequence (which implements a forward class)
        :param num_input_frames: number of frames for network input. the middle frame will be used since it is the target
        """
        super().__init__()
        self.target_seq = target_seq
        self.num_input_frames = num_input_frames
        self.tar_ix = ((num_input_frames - 1) // 2)

    def forward(self, x, bg):
        """

        :param x:
        :param bg:
        :return: cat of target seq and bg
        """
        target = self.target_seq.forward(x)
        return torch.cat((target, bg[self.tar_ix]), 0)


class OffsetRep(TargetGenerator):
    """
    Target generator to generate a 5 channel target.
    0th ch: emitter prob
    1st ch: photon value
    2nd ch: dx subpixel offset
    3rd ch: dy subpixel offset
    4th ch: z values
    """
    def __init__(self, xextent, yextent, zextent, img_shape, cat_output=True, photon_threshold=None):
        super().__init__()
        if photon_threshold is not None:
            raise ValueError("Photon threshold not supported within here.")
        self.delta = DeltaPSF(xextent,
                              yextent,
                              zextent,
                              img_shape,
                              photon_normalise=False,
                              photon_threshold=photon_threshold)

        # this might seem as a duplicate, but we need to make sure not to use a photon threshold for generating the z map.
        self.delta_z = DeltaPSF(xextent,
                              yextent,
                              zextent,
                              img_shape,
                              photon_normalise=False,
                              photon_threshold=None)

        self.offset = OffsetPSF(xextent, yextent, img_shape)
        self.cat_out = cat_output

    def forward(self, x, weight=None):
        """
        Create 5 channel output, decode_like
        :param x: emitter instance
        :return: concatenated maps, or single maps with 1 x H x W each (channel order: p, I, x  y, z)
        """
        if weight is None:
            weight = torch.ones_like(x.phot)

        p_map = self.delta.forward(x, weight)
        """It might happen that we see that two emitters are in the same px. p_map will then be 2 or greater.
        As the offset map allows for only one emitter, set the greater than 1 px to 1."""
        p_map[p_map > 1] = 1

        phot_map = self.delta.forward(x, x.phot)
        xy_map = self.offset.forward(x)
        z_map = self.delta_z.forward(x, x.xyz[:, 2])

        if self.cat_out:
            return torch.cat((p_map, phot_map, xy_map, z_map), 0)
        else:
            return p_map, phot_map, xy_map[[0]], xy_map[[1]], z_map


class ROIOffsetRep(OffsetRep):
    """
    Generate a target with increased size as compared to OffsetRep.
    """
    def __init__(self, xextent, yextent, zextent, img_shape, roi_size=3):
        super().__init__(xextent, yextent, zextent, img_shape, cat_output=True, photon_threshold=None)
        self.roi_size = roi_size
        if self.roi_size != 3:
            raise NotImplementedError("Currently only ROI size 3 is implemented and tested.")

        """Addition 'kernel' for phot channel, dxyz"""
        self.kern_dy = torch.tensor([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
        self.kern_dx = torch.tensor([[1., 1., 1], [0., 0., 0.], [-1., -1., -1.]])

    @staticmethod
    def parse(param: dict):
        return ROIOffsetRep(param['Simulation']['psf_extent'][0],
                            param['Simulation']['psf_extent'][1],
                            None,
                            param['Simulation']['img_size'],
                            param['HyperParameter']['target_roi_size'])

    def forward(self, x):
        offset_target = super().forward(x)

        """In the photon, dx, dy, dz channel we want to increase the area of information."""
        # zero pad the target in image space so that we don't have border problems
        offset_target_pad = functional.pad(offset_target, (1, 1, 1, 1), mode='constant', value=0.)
        target = torch.zeros_like(offset_target_pad)

        is_emitter = offset_target_pad[0].nonzero()
        # loop over the non-zero elements
        for i in range(is_emitter.size(0)):
            ix_x = slice(is_emitter[i, 0].item() - 1, is_emitter[i, 0].item() + 2)
            ix_y = slice(is_emitter[i, 1].item() - 1, is_emitter[i, 1].item() + 2)

            # leave p_channel unchanged, all others use addition kernel
            target[1, ix_x, ix_y] = offset_target_pad[1, is_emitter[i, 0], is_emitter[i, 1]]
            target[2, ix_x, ix_y] = self.kern_dx + offset_target_pad[2, is_emitter[i, 0], is_emitter[i, 1]]
            target[3, ix_x, ix_y] = self.kern_dy + offset_target_pad[3, is_emitter[i, 0], is_emitter[i, 1]]
            target[4, ix_x, ix_y] = offset_target_pad[4, is_emitter[i, 0], is_emitter[i, 1]]

        # set px centres to original value
        target[:, is_emitter[:, 0], is_emitter[:, 1]] = offset_target_pad[:, is_emitter[:, 0], is_emitter[:, 1]]

        # remove padding
        return target[:, 1:-1, 1:-1]


class GlobalOffsetRep(OffsetRep):
    def __init__(self, xextent, yextent, zextent, img_shape, photon_threshold=None, masked=False):
        super().__init__(xextent, yextent, zextent, img_shape, cat_output=True, photon_threshold=photon_threshold)
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.img_shape = img_shape
        self.masked = masked

        self.nearest_neighbor = neighbors.KNeighborsClassifier(1, weights='uniform')

        """Grid coordinates"""
        # ToDo: Double check
        x = torch.linspace(xextent[0], xextent[1], steps=img_shape[0] + 1).float()[:-1]
        y = torch.linspace(yextent[0], yextent[1], steps=img_shape[1] + 1).float()[:-1]

        # add half a pixel to get the center
        x += (x[1] - x[0]) / 2
        y += (y[1] - y[0]) / 2
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
        self.xy_list = torch.cat((self.x_grid.contiguous().view(-1, 1), self.y_grid.contiguous().view(-1, 1)), dim=1)

    def assign_emitter(self, x, coordinates=None):
        """
        Assign px to the closest emitter
        :param x: emitter instance
        :return:
        """
        """Fit the emitters. Fit only in 2D"""
        dummy_index = torch.arange(0, x.num_emitter)
        self.nearest_neighbor.fit(x.xyz[:, :2].numpy(), dummy_index)

        """Predict NN for all image coordinates"""
        if coordinates is None:
            pred_ix = self.nearest_neighbor.predict(self.xy_list)
        else:
            pred_ix = self.nearest_neighbor.predict(coordinates)

        return torch.from_numpy(pred_ix)

    def forward(self, x):
        """

        :param x: emitter instance
        :return: 5 ch output
        """
        p_map = self.delta.forward(x, torch.ones_like(x.phot))
        p_map[p_map > 1] = 1

        """If masked: Do not output phot / dx, dy, dz everywhere but just around an emitter."""
        if self.masked:
            conv_kernel = torch.ones((1, 1, 3, 3))
            photxyz_mask = functional.conv2d(p_map.unsqueeze(0), conv_kernel, padding=1).squeeze(0).squeeze(0)
            photxyz_mask[photxyz_mask >= 1.] = 1

            img_pseudo_extent = ((-0.5, self.img_shape[0]-0.5), (-0.5, self.img_shape[1]-0.5))
            # find indices
            mat_indices = photxyz_mask.squeeze(0).nonzero()
            coordinates = A2BTransform(img_pseudo_extent, (self.xextent, self.yextent)).a2b(mat_indices.float())

            pred_ix = self.assign_emitter(x, coordinates)
            pred_image = (-1) * torch.ones((self.img_shape[0], self.img_shape[1]))
            pred_image[photxyz_mask.byte()] = pred_ix.float()

        else:
            pred_ix = self.assign_emitter(x, None)
            pred_image = pred_ix.reshape(self.img_shape[0], self.img_shape[1])
            photxyz_mask = torch.ones_like(pred_image)

        """Calculate l1 distance per dimension to predicted index for all image coordinates"""
        dx = torch.zeros(self.img_shape[0], self.img_shape[1])
        dy = torch.zeros_like(dx)
        z = torch.zeros_like(dx)
        phot_map = torch.zeros_like(dx)

        for i in range(x.num_emitter):
            mask = (pred_image == i)

            dx[mask] = x.xyz[i, 0] - self.x_grid[mask]
            dy[mask] = x.xyz[i, 1] - self.y_grid[mask]
            z[mask] = x.xyz[i, 2]
            phot_map[mask] = x.phot[i]

        target = torch.cat((p_map, phot_map.unsqueeze(0), dx.unsqueeze(0), dy.unsqueeze(0), z.unsqueeze(0)), 0)
        target[1:] *= photxyz_mask.float()
        return target


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


class ThresholdPhotons:
    def __init__(self, photon_threshold, mode=None):
        """
        Thresholds the photon for prediction. Useful for CRLB calculation.

        Args:
            photon_threshold: threshold values for photon count
            mode: this makes it possible to use this as a pre-step for the weight generator and the target generator
        """
        self.photon_threshold = photon_threshold
        self._mode = mode

        if self._mode not in (None, 'target', 'weight'):
            raise ValueError("Not supported.")

    @staticmethod
    def parse(param, mode=None):
        return ThresholdPhotons(photon_threshold=param.HyperParameter.photon_threshold, mode=mode)

    def forward_impl(self, em):
        if self.photon_threshold is None:
            return em

        ix = em.phot >= self.photon_threshold
        return em[ix]

    def forward(self, *args):
        """
        Removes the emitters that have too few localisations.
        Cumbersome implementation because this can be used in multiple places.

        Args:
            args: various arguments. In standard / default mode just the emitterset.

        Returns:
            emitterset + args without the emitters with too low photon value

        """
        if self._mode is None:
            return self.forward_impl(args[0])
        elif self._mode == 'target':
            if args.__len__() == 1:
                return self.forward_impl(args[0])
            else:
                return (self.forward_impl(args[0]), *args[1:])
        elif self._mode == 'weight':
            if args.__len__() == 2:
                return args[0], self.forward_impl(args[1])
            else:
                return (args[0], self.forward_impl(args[1]), *args[2:])
