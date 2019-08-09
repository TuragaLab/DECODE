from abc import ABC, abstractmethod  # abstract class
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsmlm.generic.noise import GaussianSmoothing


class FocalLoss(nn.Module):
    """
    Taken from: https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
    """

    def __init__(self, focusing_param=2, balance_param=0.25):
        super().__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        """

        :param output: N C d1 d2 d3 ...
        :param target: N d1 d2 d3 ...
        where C are the classes!

        :return:
        """

        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss

class Loss(ABC):
    """Abstract class for my loss functions."""

    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def functional():
        """
        static / functional of the loss
        :return:
        """
        pass

    @abstractmethod
    def __call__(self, output, target):
        """
        calls functional
        """

        return dummyLoss


class LossLog(Loss):
    """
    "Pseudo" Abstract loss class which should be used to log individual loss components.
    """
    def __init__(self, cmp_desc=None, logger=None):
        super().__init__()

        self.cmp_desc = cmp_desc
        self.logger = logger

    @abstractmethod
    def log_components(self, loss_vec):
        pass


class SpeiserLoss(Loss):
    cmp_suffix = ('p', 'phot', 'dx', 'dy', 'dz')

    def __init__(self, weight_sqrt_phot, class_freq_weight=None, pch_weight=1., cmp_prefix='loss', logger=None):
        """

        :param weight_sqrt_phot: weight phot, dx, dy, dz etc. by sqrt(phot), i.e. weight the l2 loss
        :param class_freq_weight: weight positives by a factor (to balance fore / background). a good starting point
            might be num_px / avg. emitter per image
        :param pch_weight: weight of the p channel
        :param cmp_desc
        :param logger: logging instance (tensorboard)
        :param log: log? true / false
        """
        super().__init__()
        self.weight_sqrt_phot = weight_sqrt_phot
        self.class_freq_weight = class_freq_weight
        self.pch_weight = pch_weight

        self.p_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.phot_xyz_loss = torch.nn.MSELoss(reduction='none')

        self.cmp_desc = [cmp_prefix + '/' + v for v in SpeiserLoss.cmp_suffix]
        self.cmp_val = None
        self.logger = logger

        self.cmp_values = None
        self._reset_batch_log()

    def __call__(self, output, target):
        return self.functional(output, target, self.p_loss, self.phot_xyz_loss,
                               self.weight_sqrt_phot, self.class_freq_weight, self.pch_weight)

    def _reset_batch_log(self):
        self.cmp_values = torch.zeros((0, 5))

    @staticmethod
    def functional(output, target, p_loss, phot_xyz_loss, weight_by_phot, class_freq_weight, pch_weight):
        mask = target[:, [0], :, :]

        p_loss = p_loss(output[:, [0], :, :], target[:, [0], :, :])
        if class_freq_weight is not None:
            weight = torch.ones_like(p_loss)
            weight[target[:, [0], :, :] == 1.] = class_freq_weight
            p_loss *= weight

        if weight_by_phot:
            mask *= target[:, [1], :, :].sqrt()

        """Mask and weight the loss"""
        xyzi_loss = phot_xyz_loss(output[:, 1:, :, :], target[:, 1:, :, :])
        xyzi_loss *= mask

        return torch.cat((pch_weight * p_loss, xyzi_loss), 1)

    def log_batch_loss_cmp(self, loss_vec):
        self.cmp_values = torch.cat((self.cmp_values, loss_vec.mean(-1).mean(-1).mean(0).view(1, 5).cpu()), dim=0)

    def log_components(self, ix):

        cmp_values = self.cmp_values.mean(0)  # loss wrt to the batches

        for i, cmp in enumerate(self.cmp_desc):
            self.logger.add_scalar(cmp, cmp_values[i].item(), ix)

        self._reset_batch_log()


class OffsetROILoss(SpeiserLoss):
    def __init__(self, roi_size=3, weight_sqrt_phot=False, class_freq_weight=None, ch_weight=None, cmp_prefix='loss', logger=None):
        """

        :param weight_sqrt_phot:
        :param class_freq_weight:
        :param ch_weight: tensor of size 5
        :param cmp_prefix:
        :param logger:
        """
        super().__init__(weight_sqrt_phot, class_freq_weight, None, cmp_prefix, logger)
        self.roi_size = roi_size
        if roi_size != 3:
            raise NotImplementedError('Only ROI size 3 supported currently.')

        if ch_weight is None:
            self.ch_weight = torch.ones((1, 5, 1, 1))
        else:
            self.ch_weight = ch_weight.view(1, 5, 1, 1)

    def __call__(self, output, target):
        return self.functional(output, target, self.roi_size, self.p_loss, self.phot_xyz_loss,
                               self.weight_sqrt_phot, self.class_freq_weight, self.ch_weight)

    @staticmethod
    def functional(output, target, roi_size, p_loss, phot_xyz_loss, weight_by_phot, class_freq_weight, ch_weight):
        mask = target[:, [0], :, :]
        is_emitter = target[:, [0], :, :].byte()  # save indexing tensor where we have an emitter

        conv_kernel = torch.tensor([[1 / 4, 1 / 2, 1 / 4],
                                    [1 / 2, 1., 1 / 2],
                                    [1 / 4, 1 / 2, 1 / 4]])
        conv_kernel /= conv_kernel.sum()
        conv_kernel = conv_kernel.unsqueeze(0).unsqueeze(0).to(output.device)
        mask = F.conv2d(mask, conv_kernel, padding=1)

        """Calculate overlapping ROI"""
        overlap_kernel = torch.ones((1, 1, roi_size, roi_size)).to(output.device)
        overlap_mask = F.conv2d(target[:, [0]], overlap_kernel, padding=1)

        """Where we have overlapping ROI, set mask to zero but not in the ground truth pixel"""
        mask_ = mask.clone()
        is_overlap = overlap_mask >= 2.
        mask_[is_overlap] = 0.
        mask_[is_emitter] = mask[is_emitter]
        mask = mask_

        p_loss = p_loss(output[:, [0], :, :], target[:, [0], :, :])
        if class_freq_weight is not None:
            weight = torch.ones_like(p_loss)
            weight[target[:, [0], :, :] == 1.] = class_freq_weight
            p_loss *= weight

        if weight_by_phot:
            mask *= target[:, [1], :, :].sqrt()

        """Mask and weight the loss"""
        xyzi_loss = phot_xyz_loss(output[:, 1:, :, :], target[:, 1:, :, :])
        xyzi_loss *= mask

        out = torch.cat((p_loss, xyzi_loss), 1)
        out *= ch_weight.to(out.device)
        return out


class FocalOffsetLoss(SpeiserLoss):
    def __init__(self, alpha, gamma):
        """

        :param alpha: balance factor. roughly emitters / number of pixels
        :param gamma: parameter for focal loss. 0 for cross entropy, > 0.5 weights hard examples more
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss = FocalLoss(num_class=2, gamma=self.gamma)
        self.l2 = torch.nn.MSELoss()

    def forward(self, input, target):
        mask = target[:, [0], :, :]

        p_loss = self.focal_loss(input[:, [0]], target[:, [0]])
        phot_loss = self.l2(input[:, 1], target[:, 1])

        xy_loss = self.l2(torch.log(input[:, 2:4].abs() + 1), torch.log(target[:, 2:4].abs() + 1))
        z_loss = self.l2(input[:, -1], target[:, -1])

        return p_loss + phot_loss + xy_loss + z_loss


class MaskedOnlyZLoss:
    def __init__(self):
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()

    @staticmethod
    def loss(z_out, z_tar, mask, loss_kernel):
        """

        :param z_out: z-output (N x C x H x W)
        :param z_tar: z-target (N x C x H x W)
        :param mask: (torch.Tensor 2D) mask to weight the loss dependened on region
        :return: (float)
        """
        return loss_kernel(z_out * mask, z_tar * mask)

    def return_criterion(self):

        def loss_return(output, target, mask):
            return self.loss(output, target, mask, self.l1)

        return loss_return


class MultiScaleLaplaceLoss:
    """From Boyd."""
    def __init__(self, kernel_sigmas, pos_mul_f=torch.Tensor([1.0, 1.0, 0.2])):
        super().__init__()

        self.kernel_sigmas = kernel_sigmas
        self.pos_mul =  pos_mul_f

    @staticmethod
    def loss(xyz_out, xyz_tar, phot_out, phot_tar, kernel_sigmas, pw_l1, kernel_l):
        """

        :param kernel_sigmas: tuple of sigmas()
        :param pw_l1: pairwise l1 loss function
        :param kernel_l: kernel_loss function
        :return:
        """

        D = pw_l1(xyz_out, xyz_tar)
        losses = [kernel_l(torch.exp(-D / sf), phot_out, phot_tar) for sf in kernel_sigmas]
        return sum(losses).mean()

    @staticmethod
    def pairwise_l2_dist(x, y, eps=1E-10):
        if y is not None:
            p = torch.cat((x, y), dim=1)
        else:
            p = torch.cat((x, x), dim=1)
        differences = p.unsqueeze(2) - p.unsqueeze(1)
        return (torch.sum(differences * differences, -1) + eps).sqrt()

    @staticmethod
    def pairwise_l1_dist(x, y):
        if y is not None:
            p = torch.cat((x, y), dim=1)
        else:
            p = torch.cat((x, x), dim=1)
        differences = p.unsqueeze(2) - p.unsqueeze(1)
        return torch.sum(differences.abs(), -1)

    @staticmethod
    def kernel_loss(kernel, weight_pred, weight_target):
        weight = torch.cat([weight_pred, -weight_target], 1).unsqueeze(-1)
        embedding_loss = torch.matmul(weight.transpose(1, 2),
                                      torch.matmul(kernel, weight))
        return embedding_loss.squeeze()

    def return_criterion(self):

        def loss_return(output, target):
            xyz_out = output[0] * self.pos_mul.to(output[0].device)
            xyz_tar = target[0] * self.pos_mul.to(target[0].device)
            phot_out = output[1]
            phot_tar = target[1]
            return self.loss(xyz_out, xyz_tar,
                             phot_out, phot_tar,
                             self.kernel_sigmas, self.pairwise_l1_dist, self.kernel_loss)

        return loss_return


class RepulsiveLoss:

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.metric_kernel = MultiScaleLaplaceLoss.pairwise_l2_dist

    @staticmethod
    def loss(pos_out, sf, metric_kernel):
        D = metric_kernel(pos_out)
        loss = torch.exp(-D / (sf * 5))
        return loss.mean()

    def return_criterion(self):

        def loss_return(xyz_out):
            return self.loss(xyz_out, self.scale_factor, self.metric_kernel)

        return loss_return


class MultiSLLRedClus(MultiScaleLaplaceLoss):
    """From Boyd."""
    def __init__(self, kernel_sigmas, pos_mul_f=torch.Tensor([1.0, 1.0, 0.2]), phot_loss_sc=1, loc=0.2, scale=0.05):
        self.boyd = MultiScaleLaplaceLoss(kernel_sigmas, pos_mul_f).return_criterion()
        self.phot_loss_sc = phot_loss_sc
        norm = torch.distributions.normal.Normal(loc, scale)
        norm.maxv = torch.exp(norm.log_prob(loc))
        norm.pdf_norm = lambda x: torch.exp(norm.log_prob(x)) / norm.maxv

        self.phot_loss_sc = phot_loss_sc
        self.phot_loss = norm.pdf_norm

    def return_criterion(self):

        def loss_comp(output, target):
            phot_out = output[1]
            return self.boyd(output, target) + self.phot_loss_sc * (self.phot_loss(phot_out).sum(1).mean())

        return loss_comp


class BumpMSELoss(Loss):
    """
    Loss which comprises a L2 loss of heatmap (single one hot output convolved by gaussian)
    plus L1 loss (output, 0)
    """

    def __init__(self, kernel_sigma, cuda, l1_f=1, kernel_size=16, kernel_mode='gaussian'):
        """

        :param kernel_sigma: sigma value of gaussian kernel
        :param cuda:
        :param l1_f: scaling factor of L1
        """
        super().__init__()

        self.l1_f = l1_f
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.padding_same_v = math.ceil((self.kernel_size - 1) / 2)

        def padding(x): return functional.pad(x, [self.padding_same_v,
                                                  self.padding_same_v,
                                                  self.padding_same_v,
                                                  self.padding_same_v], mode='reflect')

        self.hm_kernel = GaussianSmoothing(channels=1,
                                           kernel_size=[
                                               self.kernel_size,
                                               self.kernel_size],
                                           sigma=kernel_sigma,
                                           dim=2,
                                           cuda=cuda,
                                           padding=padding,
                                           kernel_f=self.kernel_mode)

        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()

    @staticmethod
    def loss(output, target, kernel, l1, l2, l1_f):
        """

        :param output: torch.tensor of size N x C x H x W
        :param target: torch.tensor of size N x C x H x W
        :param kernel: kernel function to produce heatmap
        :param l1: l1 loss function
        :param l2: l2 loss function
        :param l1_f: weighting factor of l1 term
        :return: loss value
        """
        hm_out = kernel(output)
        hm_tar = kernel(target)

        l1_loss = l1(output, torch.zeros_like(output))
        l2_loss = l2(hm_out, hm_tar)

        return l1_f * l1_loss + l2_loss

    def return_criterion(self):
        """
        :return: criterion function
        """
        def loss_return(output, target):
            return self.loss(output, target, self.hm_kernel, self.l1, self.l2, self.l1_f)

        return loss_return


class BumpMSELoss3DzLocal(Loss):
    """
    Class to output a composite loss, comprised of a N photons loss and a z value loss.
    The loss value of z is only taken into account where we detect a photon in the first channel.
    """
    def __init__(self, kernel_sigma_photons, kernel_sigma_z, cuda, phot_thres, l1_f=1, d3_f=1):
        """

        :param kernel_sigma_photons: kernel to produce the heatmap for photon
        :param kernel_sigma_z: kernel to produce the heatmap for z values, this can be dangerous!
        :param cuda:
        :param phot_thres:
        :param l1_f:
        :param d3_f:
        """
        super().__init__()

        self.phot_thres = phot_thres
        self.d3_f = d3_f

        self.loss_photon = BumpMSELoss(kernel_sigma=kernel_sigma_photons, cuda=cuda, l1_f=l1_f)
        self.hm_kernel_z = lambda x: x  # BumpMSELoss(kernel_sigma=kernel_sigma_z, cuda=cuda).hm_kernel

        self.z_loss_l2 = torch.nn.MSELoss(reduction='sum')

    @staticmethod
    def loss(output, target, kernel, threshold, l2, r):

        hm_out = kernel(output[:, [1], :, :])
        hm_tar = kernel(target[:, [1], :, :])

        is_emit = output[:, [0], :, :] >= threshold
        loss_z = l2(hm_out[is_emit], hm_tar[is_emit]) / r

        return loss_z

    def return_criterion(self):
        def loss_compositum(output, target):
            total_loss = self.d3_f * self.loss(output, target,
                                               self.hm_kernel_z,
                                               self.phot_thres,
                                               self.z_loss_l2,
                                               output.numel()) \
                + self.loss_photon.return_criterion()(output[:, [0], :, :], target[:, [0], :, :])

            return total_loss

        return loss_compositum


class BumpMSELoss3DzLocal(Loss):
    """
    Class to output a composite loss, comprised of a N photons loss and a z value loss.
    """
    def __init__(self, kernel_sigma_photons, kernel_sigma_z, cuda, phot_thres, l1_f=1, d3_f=1):
        """

        :param kernel_sigma_photons: kernel to produce the heatmap for photon
        :param kernel_sigma_z: kernel to produce the heatmap for z values, this can be dangerous!
        :param cuda:
        :param phot_thres:
        :param l1_f:
        :param d3_f:
        """
        super().__init__()

        self.phot_thres = phot_thres
        self.d3_f = d3_f

        self.loss_photon = BumpMSELoss(kernel_sigma=kernel_sigma_photons, cuda=cuda, l1_f=l1_f)
        self.hm_kernel_z = BumpMSELoss(kernel_sigma=kernel_sigma_z, cuda=cuda).hm_kernel

        self.z_loss_l2 = torch.nn.MSELoss(reduction='sum')

    @staticmethod
    def loss(output, target, kernel, threshold, l2, r):

        hm_out = kernel(output[:, [1], :, :])
        hm_tar = kernel(target[:, [1], :, :])

        is_emit = output[:, [0], :, :] >= threshold
        loss_z = l2(hm_out[is_emit], hm_tar[is_emit]) / r

        return loss_z

    def return_criterion(self):
        def loss_compositum(output, target):
            total_loss = self.d3_f * self.loss(output, target,
                                               self.hm_kernel_z,
                                               self.phot_thres,
                                               self.z_loss_l2,
                                               output.numel()) \
                + self.loss_photon.return_criterion()(output[:, [0], :, :], target[:, [0], :, :])

            return total_loss

        return loss_compositum

def combine_msc_repulsive(kernel, repulsive_loc, repulsive_sc):
    pass

if __name__ == '__main__':
    xyz = torch.tensor([[0., 0., 0], [0., 0., 0.], [0., 0., 0.]])
    phot = torch.tensor([1., 0., 0.])

    xyz_out = torch.tensor([[0., 0., 0], [0., 0., 0.], [0., 0., 0.]])
    phot_out = torch.tensor([0.3, 0.3, 0.3])

    xyz_out = torch.cat((xyz_out.unsqueeze(0), xyz.unsqueeze(0)), 0)
    phot_out = torch.cat((phot_out.unsqueeze(0), phot.unsqueeze(0)), 0)

    xyz = torch.cat((xyz.unsqueeze(0), xyz.unsqueeze(0)), 0)
    phot = torch.cat((phot.unsqueeze(0), phot.unsqueeze(0)), 0)

    target = (xyz, phot)
    output = (xyz_out, phot_out)
    loss = MultiSLLRedClus((0.64, 3.20, 6.4, 19.2), loc=0.15, scale=0.03, phot_loss_sc=1).return_criterion()
    l = loss(output, target)
    print(l)
    print("Success.")