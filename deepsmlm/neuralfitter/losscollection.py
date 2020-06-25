from abc import ABC, abstractmethod  # abstract class
from typing import Union, Tuple

import torch
from deprecated import deprecated

from . import MixtureSameFamily as mixture
from . import post_processing


class Loss(ABC):
    """Abstract class for my loss functions."""

    def __init__(self):
        super().__init__()

    def __call__(self, output, target, weight):
        """
        calls functional
        """
        return self.forward(output, target, weight)

    @abstractmethod
    def log(self, loss_val):
        """

        Args:
            loss_val:

        Returns:
            float: single scalar that is subject to the backprop algorithm
            dict:  dictionary with values being floats, describing additional information (e.g. loss components)
        """
        raise NotImplementedError

    def _forward_checks(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        """
        Some sanity checks for forward data

        Args:
            output:
            target:
            weight:

        """
        if not (output.size() == target.size() and target.size() == weight.size()):
            raise ValueError(f"Dimensions of output, target and weight do not match "
                             f"({output.size(), target.size(), weight.size()}.")

    @abstractmethod
    def forward(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss term

        Args:
            output (torch.Tensor): output of the network
            target (torch.Tensor): target data
            weight (torch.Tensor): px-wise weight map

        Returns:
            torch.Tensor

        """
        raise NotImplementedError


class PPXYZBLoss(Loss):
    """
    Loss implementation for 6 channel output for SMLM data, where the channels are

        0: probabilities (without sigmoid)
        1: photon count
        2: x pointers
        3: y pointers
        4: z pointers
        5: background
    """

    def __init__(self, device: Union[str, torch.device], chweight_stat: Union[tuple, list, torch.Tensor] = None,
                 p_fg_weight: float = 1., forward_safety: bool = True):
        """

        Args:
            device: device in forward method (e.g. 'cuda', 'cpu', 'cuda:0')
            chweight_stat: static channel weight
            p_fg_weight: foreground weight
            forward_safety: check sanity of forward arguments
        """

        super().__init__()
        self.forward_safety = forward_safety

        if chweight_stat is not None:
            self._ch_weight = chweight_stat if isinstance(chweight_stat, torch.Tensor) else torch.Tensor(chweight_stat)
        else:
            self._ch_weight = torch.tensor([1., 1., 1., 1., 1., 1.])
        self._ch_weight = self._ch_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

        self._p_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(p_fg_weight).to(device))
        self._phot_xyzbg_loss = torch.nn.MSELoss(reduction='none')

    def log(self, loss_val) -> (float, dict):
        loss_vec = loss_val.mean(-1).mean(-1).mean(0)
        return loss_vec.mean().item(), {
            'p': loss_vec[0].item(),
            'phot': loss_vec[1].item(),
            'x': loss_vec[2].item(),
            'y': loss_vec[3].item(),
            'z': loss_vec[4].item(),
            'bg': loss_vec[5].item()
        }

    def _forward_checks(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        super()._forward_checks(output, target, weight)

        if output.size(1) != 6:
            raise ValueError("Not supported number of channels for this loss function.")

    def forward(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        if self.forward_safety:
            self._forward_checks(output, target, weight)

        ploss = self._p_loss(output[:, [0]], target[:, [0]])
        chloss = self._phot_xyzbg_loss(output[:, 1:], target[:, 1:])
        tot_loss = torch.cat((ploss, chloss), 1)

        tot_loss = tot_loss * weight * self._ch_weight

        return tot_loss


class FourFoldPPXYZ(Loss):
    """
    Four-fold implementation of PPXYZ Loss, i.e. half pixel shifted pointers.

    """

    def __init__(self, components):
        """

        Args:
            components: respective PXYZ Loss instances

        """
        super().__init__()

        self.com = components
        self.bg_loss = torch.nn.MSELoss(reduction='none')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def log(self, loss_val):
        return loss_val.mean().item(), {}

    def _forward_checks(output, target, weight):
        pass

    def forward(self, output, target, weight):
        assert output.size(1) % len(self.com) == 1
        n_ch_per_com = (output.size(1) - 1) // len(self.com)

        loss = torch.cat(
            [self.com[i].forward(output[:, c:c + n_ch_per_com],
                                 target[:, c:c + n_ch_per_com],
                                 weight[:, c:c + n_ch_per_com]) for i, c in enumerate(range(0, 20, 5))], 1)

        loss = torch.cat((loss, self.bg_loss(output[:, [-1]], target[:, [-1]]) * weight[:, [-1]]), 1)

        return loss


class GaussianMMLoss(Loss):
    """
    Model output is a mean and sigma value which forms a gaussian mixture model.
    """

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple):
        super().__init__()

        self._bg_loss = torch.nn.MSELoss(reduction='none')
        # self._offset2coord = psf_kernel.DeltaPSF(xextent=xextent, yextent=yextent, img_shape=img_shape)
        self._offset2coord = post_processing.Offset2Coordinate(xextent=xextent, yextent=yextent, img_shape=img_shape)

    def log(self, loss_val):
        pass

    @staticmethod
    def _format_model_output(output: torch.Tensor) -> tuple:
        """
        Transforms solely channel based model output into more meaningful variables.

        Args:
            output: model output

        Returns:
            tuple containing
                p: N x H x W
                pxyz_mu: N x 4 x H x W
                pxyz_sig: N x 4 x H x W
                bg: N x H x W
        """
        p = output[:, 0]
        pxyz_mu = output[:, 1:5]
        pxyz_sig = output[:, 5:-1]
        bg = output[:, -1]

        return p, pxyz_mu, pxyz_sig

    def _compute_gmm_loss(self, p, pxyz_mu, pxyz_sig, pxyz_tar, mask) -> torch.Tensor:
        """
        Computes the Gaussian Mixture Loss.

        Args:
            p: the model's detection prediction (sigmoid already applied) size N x H x W
            pxyz_mu: prediction of parameters (phot, xyz) size N x C=4 x H x W
            pxyz_sig: prediction of uncertainties / sigma values (phot, xyz) size N x C=4 x H x W
            pxyz_tar: ground truth values (phot, xyz) size N x M x 4 (M being max number of tars)
            mask: activation mask of ground truth values (phot, xyz) size N x M

        Returns:
            torch.Tensor (size N x 1)

        """

        batch_size = pxyz_mu.size(0)
        log_prob = 0

        p_mean = p.sum(-1).sum(-1)
        p_var = (p - p ** 2).sum(-1).sum(-1)  # var estimate of bernoulli
        p_gauss = mixture.D.Normal(p_mean, torch.sqrt(p_var))

        log_prob += p_gauss.log_prob(mask.sum(-1)) * mask.sum(-1)

        prob_normed = p / (p.sum(-1).sum(-1)[:, None, None])

        """Convert px shifts to absolute coordinates"""
        pxyz_mu[:, 1], pxyz_mu[:, 2] = self._offset2coord._subpx_to_absolute(pxyz_mu[:, 1], pxyz_mu[:, 2])
        # pxyz_mu[:, 1] += self._offset2coord.bin_ctr_x[p_inds[1]]
        # pxyz_mu[:, 2] += self._offset2coord.bin_ctr_y[p_inds[2]]

        """Hacky way to get all prob indices"""
        p_inds = tuple((p + 1).nonzero().transpose(1, 0))
        pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]

        """Flatten img dimension --> N x (HxW) x 4"""
        pxyz_mu = pxyz_mu.reshape(batch_size, -1, 4)
        pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(batch_size, -1, 4)

        """Set up mixture family"""
        mix = mixture.D.Categorical(prob_normed[p_inds].reshape(batch_size, -1))
        comp = mixture.D.Independent(mixture.D.Normal(pxyz_mu, pxyz_sig), 1)
        gmm = mixture.MixtureSameFamily(mix, comp)

        """Calc log probs if there is anything there"""
        if mask.sum():
            gmm_log = gmm.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
            gmm_log = (gmm_log * mask).sum(-1)
            log_prob += gmm_log

        # log_prob = log_prob.reshape(batch_size, 1)  # need?

        return -log_prob

    def _forward_checks(self, output: torch.Tensor, target: tuple):

        if output.dim() != 4:
            raise ValueError(f"Output must have 4 dimensions (N,C,H,W).")

        if output.size(1) != 10:
            raise ValueError(f"Wrong number of channels.")

        if len(target) != 3:
            raise ValueError(f"Wrong length of target.")

    def forward(self, output: torch.Tensor, target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        if self.forward_safety:
            self._forward_checks(output, target)

        tar_param, tar_mask, tar_bg = target
        p, pxyz_mu, pxyz_sig, bg = self._format_model_output(output)

        bg_loss = self._bg_loss(bg.unsqueeze(1), tar_bg).sum(-1).sum(-1)  # ToDo: Check dim
        gmm_loss = self._compute_gmm_loss(p, pxyz_mu, pxyz_sig, tar_param, tar_mask)

        return torch.mean(bg_loss + gmm_loss)


@deprecated("Draft, not ready.")
class PPXYZBSigmaLoss(PPXYZBLoss):
    """
    Draft implementation. Not yet ready
    """

    def log(self, loss_val) -> (float, dict):
        return loss_val.item(), {}

    def forward(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        assert sigma.dim() == 1
        assert sigma.size(0) == output.size(1)

        loss = super().forward(output, target, weight)
        loss /= sigma.view(1, -1, 1, 1)  # expand to nchw format

        loss = loss.mean() + torch.log(sigma.prod())

        return loss
