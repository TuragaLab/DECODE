from abc import ABC, abstractmethod  # abstract class
from typing import Callable, Union, Optional

import torch
from torch import distributions

from ..simulation import psf_kernel


class Reduction:
    def __init__(self, reduction: Union[str, Callable]):
        """
        Reduction, mainly for loss

        Args:
            reduction: `None`, `mean` or Callable
        """
        self._reduction = reduction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._reduction is None:
            return x
        if callable(self._reduction):
            return self._reduction(x)
        if self._reduction == "mean":
            return x.mean()

        raise NotImplementedError(f"Reduction mode ({self._reduction}) not supported.")


class Loss(ABC):
    def __init__(self, return_loggable: bool, reduction: Optional[str]):
        super().__init__()
        self._return_loggable = return_loggable
        self._reduction = Reduction(reduction)

    def __call__(self, output, target, weight):
        return self.forward(output, target, weight)

    @abstractmethod
    def forward(
        self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
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

    def _forward_checks(
        self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ):
        """
        Sanity checks for forward method

        Args:
            output:
            target:
            weight:
        """
        if not (output.size() == target.size() and target.size() == weight.size()):
            raise ValueError(
                f"Dimensions of output, target and weight do not match "
                f"({output.size(), target.size(), weight.size()}."
            )


class PPXYZBLoss(Loss):
    """
    Loss implementation for 6 channel output where the channels are

        0: probabilities (without sigmoid)
        1: photon count
        2: x pointers
        3: y pointers
        4: z pointers
        5: background
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        chweight_stat: Union[None, tuple, list, torch.Tensor] = None,
        p_fg_weight: float = 1.0,
        forward_safety: bool = True,
        return_loggable: bool = False,
    ):
        """

        Args:
            device: device in forward method (e.g. 'cuda', 'cpu', 'cuda:0')
            chweight_stat: static channel weight
            p_fg_weight: foreground weight
            forward_safety: check sanity of forward arguments
        """
        super().__init__(return_loggable=return_loggable, reduction=None)
        if return_loggable:
            raise NotImplementedError

        self.forward_safety = forward_safety

        if chweight_stat is not None:
            self._ch_weight = (
                chweight_stat
                if isinstance(chweight_stat, torch.Tensor)
                else torch.Tensor(chweight_stat)
            )
        else:
            self._ch_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self._ch_weight = (
            self._ch_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        )

        self._p_loss = torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor(p_fg_weight).to(device)
        )
        self._phot_xyzbg_loss = torch.nn.MSELoss(reduction="none")

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:

        if self.forward_safety:
            self._forward_checks(output, target, weight)

        ploss = self._p_loss(output[:, [0]], target[:, [0]])
        chloss = self._phot_xyzbg_loss(output[:, 1:], target[:, 1:])
        tot_loss = torch.cat((ploss, chloss), 1)

        tot_loss = tot_loss * weight * self._ch_weight

        return tot_loss

    def _forward_checks(
        self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ):
        super()._forward_checks(output, target, weight)

        if output.size(1) != 6:
            raise ValueError("Not supported number of channels for this loss function.")


class GaussianMMLoss(Loss):
    def __init__(
        self,
        *,
        xextent: tuple,
        yextent: tuple,
        img_shape: tuple,
        device: Union[str, torch.device],
        chweight_stat: Union[None, tuple, list, torch.Tensor] = None,
        forward_safety: bool = True,
        reduction: str = "mean",
        return_loggable: bool = False,
    ):
        """
        Gaussian Mixture Model loss. Assumes the models output to be mean and sigma
        from which a gaussian is then constructed.

        Args:
            xextent: extent in x
            yextent: extent in y
            img_shape: image size
            device: device used in training (cuda / cpu)
            chweight_stat: static channel weight,
             e.g. to disable background prediction
            forward_safety: check inputs to the forward method
            return_loggable:
        """
        super().__init__(return_loggable=return_loggable, reduction=reduction)

        if chweight_stat is not None:
            self._ch_weight = (
                chweight_stat
                if isinstance(chweight_stat, torch.Tensor)
                else torch.Tensor(chweight_stat)
            )
        else:
            self._ch_weight = torch.ones(2)
        self._ch_weight = self._ch_weight.reshape(1, 2).to(device)

        self._bg_loss = torch.nn.MSELoss(reduction="none")
        self._offset2coord = psf_kernel.DeltaPSF(
            xextent=xextent, yextent=yextent, img_shape=img_shape
        )
        self.forward_safety = forward_safety

    def forward(
        self,
        output: torch.Tensor,
        target: tuple[torch.Tensor, torch.BoolTensor, torch.Tensor],
        weight: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Forwards output and target through loss.

        Args:
            output: model's output tensor of size N x 10 x H x W
            target: target tuple of
                - emitter_param of size N x 4 x H x W
                - emitter_mask of size N x H x W
                - background: of size N x H x W
            weight: not supported

        Returns:
            - loss value
            - (optional) loggable dict
        """

        if self.forward_safety:
            self._forward_checks(output, target, weight)

        tar_param, tar_mask, tar_bg = target
        p, pxyz_mu, pxyz_sig, bg = self._format_model_output(output)

        bg_loss = self._bg_loss(bg, tar_bg).sum(-1).sum(-1)
        gmm_loss = self._compute_gmm_loss(p, pxyz_mu, pxyz_sig, tar_param, tar_mask)

        # stack in 2 channels
        # factor 2 because original impl. adds the two terms, but this way
        # it's better for logging
        loss = 2 * torch.stack((gmm_loss, bg_loss), 1) * self._ch_weight

        if self._return_loggable:
            return self._reduction(loss), self._loggable(loss)

        return self._reduction(loss)

    @staticmethod
    def _format_model_output(output: torch.Tensor) -> tuple[torch.Tensor, ...]:
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

        return p, pxyz_mu, pxyz_sig, bg

    def _compute_gmm_loss(
        self,
        p: torch.Tensor,
        pxyz_mu: torch.Tensor,
        pxyz_sig: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Computes the Gaussian Mixture Loss.

        Args:
            p: the model's detection prediction (sigmoid already applied) size N x H x W
            pxyz_mu: prediction of parameters (phot, xyz) size N x C=4 x H x W
            pxyz_sig: prediction of uncertainties / sigma values (phot, xyz) size N x C=4 x H x W
            pxyz_tar: ground truth values (phot, xyz) size N x M x 4 (M being max number of tars)
            mask: activation mask of ground truth values (phot, xyz) size N x M

        Returns:
            - torch.Tensor of loss value

        """

        batch_size = pxyz_mu.size(0)
        log_prob = 0

        p_mean = p.sum(-1).sum(-1)
        p_var = (p - p**2).sum(-1).sum(-1)  # var estimate of bernoulli
        p_gauss = distributions.Normal(p_mean, torch.sqrt(p_var))

        log_prob = log_prob + p_gauss.log_prob(mask.sum(-1)) * mask.sum(-1)

        prob_normed = p / p.sum(-1).sum(-1).view(-1, 1, 1)

        # hacky way to get all prob indices
        p_inds = tuple((p + 1).nonzero(as_tuple=False).transpose(1, 0))
        pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]

        # convert px shifts to absolute coordinates
        pxyz_mu[:, 1] += self._offset2coord.bin_ctr_x[p_inds[1]].to(pxyz_mu.device)
        pxyz_mu[:, 2] += self._offset2coord.bin_ctr_y[p_inds[2]].to(pxyz_mu.device)

        # flatten img dimension --> N x (HxW) x 4
        pxyz_mu = pxyz_mu.reshape(batch_size, -1, 4)
        pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(
            batch_size, -1, 4
        )

        if not (
            torch.distributions.constraints.simplex.check(
                prob_normed[p_inds].reshape(batch_size, -1)
            )
        ).all():
            raise ValueError

        # set up mixture family
        mix = distributions.Categorical(prob_normed[p_inds].reshape(batch_size, -1))
        comp = distributions.Independent(distributions.Normal(pxyz_mu, pxyz_sig), 1)
        gmm = distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        if mask.sum():  # calc log probs if there is anything there
            pxyz_tar[~mask] = 0.0  # non target could be NaN, needs to be masked out
            gmm_log = gmm.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
            gmm_log = (gmm_log * mask).sum(-1)
            log_prob = log_prob + gmm_log

        loss = log_prob * (-1)

        return loss

    def _loggable(self, v: torch.Tensor) -> dict:
        """
        Construct loggable

        Args:
            v: loss value of size N x 2

        Returns:
             dictionary of relevant metrics for logging
        """
        if v.dim() != 2 or v.size(-1) != 2:
            raise ValueError
        return {
            "gmm": v[:, 0].mean().item(),
            "bg": v[:, 1].mean().item(),
        }

    def _forward_checks(self, output: torch.Tensor, target: tuple, weight: None):

        if weight is not None:
            raise NotImplementedError(
                f"Weight must be None for this loss implementation."
            )

        if output.dim() != 4:
            raise ValueError(f"Output must have 4 dimensions (N,C,H,W).")

        if output.size(1) != 10:
            raise ValueError(f"Wrong number of channels.")

        if len(target) != 3:
            raise ValueError(f"Wrong length of target.")
