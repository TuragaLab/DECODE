import math
from abc import ABC, abstractmethod
import torch
import torch.nn
import numpy as np
from torch.nn import functional
from sklearn import neighbors, datasets
import warnings

import deepsmlm.generic.emitter as emc
from deepsmlm.generic.psf_kernel import ListPseudoPSF, DeltaPSF, OffsetPSF


class Delta2ROI:
    def __init__(self, roi_size, overlap_mode='zero'):
        self.roi_size = roi_size
        self.overlap_mode = overlap_mode

        if self.roi_size != 3:
            raise NotImplementedError("Currently only ROI size 3 is implemented and tested.")

        if self.overlap_mode not in ('zero', 'mean'):
            raise NotImplementedError("Only mean and zero are supported.")

        self.pad = torch.nn.ConstantPad2d(1, 0.)
        self.rep_kernel = torch.ones((1, 1, self.roi_size, self.roi_size))

    def is_overlap(self, x):
        # x non zero
        xn = torch.zeros_like(x)
        xn[x != 0] = 1.

        xn_count = torch.nn.functional.conv2d(self.pad(xn), self.rep_kernel)
        xn_count *= xn
        is_overlap = xn_count >= 2.

        return is_overlap, xn_count

    def forward(self, x):

        xrep = torch.nn.functional.conv2d(self.pad(x), self.rep_kernel)
        overlap_mask, overlap_count = self.is_overlap(x)

        if self.overlap_mode == 'zero':
            xrep[overlap_mask] = 0.
        elif self.overlap_mode == 'mean':
            xrep[overlap_mask] /= overlap_count[overlap_mask]

        return xrep


class WeightGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, frames: torch.Tensor, target_em: emc.EmitterSet, target_opt, weight_mask=None):
        """

        :param frames:
        :param targets_em: main target, emitterset
        :param target_opt: optional targets (e.g. background)
        :return:
        """

        return frames, target_em, target_opt, weight_mask


class DerivePseudobgFromBg(WeightGenerator):
    def __init__(self, xextent, yextent, img_shape, bg_roi_size, padding):
        """

        :param bg_roi_size:
        :param padding:
        """
        super().__init__()
        self.roi_size = bg_roi_size
        self.img_shape = img_shape
        self.padding = torch.nn.ReplicationPad2d(padding)  # to get the same output dim

        if (self.roi_size[0] % 2 != 0) or (self.roi_size[1] % 2 != 0):
            warnings.warn('ROI Size should be odd.')

        self.kernel = torch.ones((bg_roi_size[0], bg_roi_size[1])).unsqueeze(0).unsqueeze(0) / (bg_roi_size[0] * bg_roi_size[1])
        self.delta_psf = DeltaPSF(xextent, yextent, None, img_shape)
        self.bin_x = self.delta_psf.bin_x
        self.bin_y = self.delta_psf.bin_y

    def forward(self, frames, tar_em, tar_bg):
        """

        :param bg_frames: bg_frames of size N x C=1 x H x W
        :param tar_em: emtiters with frame_indices matching the bg_frames, so frame_ix.min() corresponds to bg_frames[0]
        :return: void
        """

        bg_framesp = self.padding(tar_bg)
        local_mean = torch.nn.functional.conv2d(bg_framesp, self.kernel, stride=1, padding=0)

        bg_start_ix = int(tar_em.frame_ix.min())

        pos_x = tar_em.xyz[:, 0]
        pos_y = tar_em.xyz[:, 1]

        ix_x = np.digitize(pos_x.numpy(), self.bin_x)
        ix_y = np.digitize(pos_y.numpy(), self.bin_y)

        for i in range(tar_em.num_emitter):
            rg_ix = slice(max(0, ix_x[i] - (self.roi_size[0] - 1) // 2),
                          min(self.img_shape[0] - 1, ix_x[i] + (self.roi_size[0] - 1) // 2))
            rg_iy = slice(max(0, ix_y[i] - (self.roi_size[1] - 1) // 2),
                          min(self.img_shape[1] - 1, ix_y[i] + (self.roi_size[1] - 1) // 2))
            tar_em.bg[i] = local_mean[bg_start_ix + int(tar_em.frame_ix[i]), 0, rg_ix, rg_iy].mean()

        return frames, tar_em, tar_bg


class CalcCRLB(WeightGenerator):
    def __init__(self, psf):
        super().__init__()
        self.psf = psf

    def forward(self, frames, tar_em, tar_bg):
        tar_em.populate_crlb(self.psf)

        return frames, tar_em, tar_bg


class GenerateWeightMaskFromCRLB(WeightGenerator):
    def __init_(self, xextent, yextent, img_shape, roi_size):
        super().__init__()

        self.weight_psf = [DeltaPSF(self.xextent, self.yextent, None, self.img_shape, None)] * 3
        self.rep_kernel = torch.ones((1, 1, self.roi_size, self.roi_size))

        self.roi_increaser = Delta2ROI(roi_size, overlap_mode='zero')

    def forward(self, frames, tar_em, tar_bg):

        # The weights
        weight = torch.zeros((frames.size(0), 3, frames.size(2), frames.size(3)))
        weight[:, 0] = self.weight_psf[0].forward(tar_em.xyz_cr)
        weight[:, 1] = self.weight_psf[1].forward(tar_em.phot_cr)
        weight[:, 2] = self.weight_psf[2].forward(tar_em.bg_cr)

        weight = self.roi_increaser.forward(weight)

        return weight

