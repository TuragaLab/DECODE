from abc import ABC, abstractmethod  # abstract class

import torch


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
    def __init__(self, device, chweight_stat: (tuple, list, torch.Tensor) = None, p_fg_weight: float = 1.,
                 forward_safety=True):

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


class PPXYZBSigmaLoss(PPXYZBLoss):

    def log(self, loss_val) -> (float, dict):
        return loss_val.item(), {}

    def forward(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:

        assert sigma.dim() == 1
        assert sigma.size(0) == output.size(1)

        loss = super().forward(output, target, weight)
        loss /= sigma.view(1, -1, 1, 1)  # expand to nchw format

        loss = loss.mean() + torch.log(sigma.prod())

        return loss


class FourFoldPXYZChecks(Loss):
    def __init__(self, components):
        super().__init__()

        self.com = components
        self.bg_loss = torch.nn.MSELoss(reduction='none')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def log(self, loss_val):
        return loss_val.mean().item(), {}

    def forward(self, output, target, weight):
        assert output.size(1) % len(self.com) == 1
        n_ch_per_com = (output.size(1) - 1) // len(self.com)

        loss = torch.cat(
            [self.com[i].forward(output[:, c:c+n_ch_per_com],
                              target[:, c:c+n_ch_per_com],
                              weight[:, c:c+n_ch_per_com]) for i, c in enumerate(range(0, 20, 5))], 1)

        loss = torch.cat((loss, self.bg_loss(output[:, [-1]], target[:, [-1]]) * weight[:, [-1]]), 1)

        return loss
