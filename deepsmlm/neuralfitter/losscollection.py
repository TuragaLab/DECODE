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
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class FocalLoss_2bdeleted(nn.Module):
    """
    (https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py)
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(self.num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Loss(ABC):
    """Abstract class for my loss functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def return_criterion(self):
        """
        Return loss function.

        :return: Return criterion function (not an evaluation!)
        """
        def dummyLoss(output, target):
            return 1

        return dummyLoss


class SpeiserLoss:
    def __init__(self, weight_sqrt_phot, class_freq_weight=None, pch_weight=1.):
        """

        :param weight_sqrt_phot: weight phot, dx, dy, dz etc. by sqrt(phot), i.e. weight the l2 loss
        :param class_freq_weight: weight positives by a factor (to balance fore / background). a good starting point
            might be num_px / avg. emitter per image
        """
        self.ce = torch.nn.BCELoss(reduction='none')
        self.l2 = torch.nn.MSELoss()
        self.weight_sqrt_phot = weight_sqrt_phot
        self.class_freq_weight = class_freq_weight
        self.pch_weight = pch_weight

    @staticmethod
    def loss(output, target, ce, l2, weight_by_phot, class_freq_weight, pch_weight):
        mask = target[:, [0], :, :]

        p_loss = ce(output[:, [0], :, :], target[:, [0], :, :])
        if class_freq_weight is not None:
            weight = torch.ones_like(p_loss)
            weight[target[:, [0], :, :] == 1.] = class_freq_weight
            p_loss *= weight

        p_loss = p_loss.mean()

        if weight_by_phot:
            weight = target[:, [1], :, :].sqrt()
        else:
            weight = torch.ones_like(target[:, [1], :, :])

        """Mask and weight the loss"""
        xyzi_loss = l2(mask * weight * output[:, 1:, :, :], mask * weight * target[:, 1:, :, :])

        return pch_weight * p_loss + xyzi_loss

    def return_criterion(self):

        def loss_return(output, target):
            return self.loss(output, target, self.ce, self.l2,
                             self.weight_sqrt_phot, self.class_freq_weight, self.pch_weight)

        return loss_return


class FocalVoronoiPointLoss(nn.Module):
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