from abc import ABC, abstractmethod  # abstract class
import torch
from torch.nn import functional

from deepsmlm.generic.noise import GaussianSmoothing


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


class BumpMSELoss(Loss):
    """
    Loss which comprises a L2 loss of heatmap (single one hot output convolved by gaussian)
    plus L1 loss (output, 0)
    """

    def __init__(self, kernel_sigma, cuda, l1_f=1):
        """

        :param kernel_sigma: sigma value of gaussian kernel
        :param cuda:
        :param l1_f: scaling factor of L1
        """
        super().__init__()

        self.l1_f = l1_f

        def padding(x): return functional.pad(x, [3, 3, 3, 3], mode='reflect')
        self.hm_kernel = GaussianSmoothing(channels=1,
                                           kernel_size=[7, 7],
                                           sigma=kernel_sigma,
                                           dim=2,
                                           cuda=cuda,
                                           padding=padding)
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
