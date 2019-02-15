from abc import ABC, abstractmethod  # abstract class
from functools import partial
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


class BumpMSELoss(Loss):
    """
    Loss which comprises a L2 loss of heatmap (single one hot output convolved by gaussian)
    plus L1 loss (output, 0)
    """

    def __init__(self, kernel_sigma, cuda, l1_f=1, l2_f=1):
        super().__init__()

        self.l1_f = l1_f
        self.l2_f = l2_f

        padding = lambda x: functional.pad(x, [3, 3, 3, 3], mode='reflect')
        self.gaussian_kernel = GaussianSmoothing(channels=1,
                                                 kernel_size=[7, 7],
                                                 sigma=kernel_sigma,
                                                 dim=2,
                                                 cuda=cuda,
                                                 padding=padding)

    @staticmethod
    def loss(output, target,
             kernel_out, kernel_target,
             l1=torch.nn.L1Loss(),
             l2=torch.nn.MSELoss(),
             l1_f=1, l2_f=1):

        heatmap_out = kernel_out(output)
        heatmap_target = kernel_target(target)

        l1_loss = l1(output, torch.zeros_like(target))
        l2_loss = l2(heatmap_out, heatmap_target)

        return l1_f * l1_loss + l2_f * l2_loss

    def return_criterion(self):
        def loss_return(output, target):
            x = self.loss(output, target,
                      kernel_out=self.gaussian_kernel,
                      kernel_target=self.gaussian_kernel,
                      l1_f=self.l1_f,
                      l2_f=self.l2_f)
            return x
        return loss_return


def bump_mse_loss_3d(output, target, kernel_pred, kernel_true, l2=torch.nn.MSELoss(), lz_sc=0.001):

    # call bump_mse_loss on first channel (photons)
    loss_photons = bump_mse_loss(output[:, [0], :, :], target[:, [0], :, :], kernel_pred, kernel_true, l1_sc=1, l2_sc=1)

    output_local_nz = kernel_pred(output[:, [0], :, :]) * kernel_pred(output[:, [1], :, :])
    target_local_nz = kernel_pred(target[:, [0], :, :]) * kernel_pred(target[:, [1], :, :])

    loss_z = l2(output_local_nz, target_local_nz)

    return loss_photons + lz_sc * loss_z


def interpoint_loss(input, target, threshold=500):
    def expanded_pairwise_distances(x, y=None):
        """
        Taken from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        if y is not None:
            differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances

    interpoint_dist = expanded_pairwise_distances(
        (input >= threshold).nonzero(), target.nonzero())
    # return distance to closest target point
    return interpoint_dist.min(1)[0].sum() / input.__len__()


def inverse_intens(output, target):
    pass


def num_active_emitter_loss(input, target, threshold=0.15):
    input_f = input.view(*input.shape[:2], -1)
    target_f = target.view(*target.shape[:2], -1)

    num_true_emitters = torch.sum(target_f > threshold * target_f.max(), 2)
    num_pred_emitters = torch.sum(input_f > threshold * input_f.max(), 2)

    loss = ((num_pred_emitters - num_true_emitters)
            ** 2).sum() / input.__len__()
    return loss.type(torch.FloatTensor)