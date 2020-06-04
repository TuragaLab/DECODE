from typing import Tuple

import torch


def weight_by_gradient(layer: torch.nn.ModuleList, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    Args:
        layer: module layers
        loss: not reduced loss values
        optimizer: optimizer

    Returns:
        weight_cX_h1_w1: weight per channel (1x C x 1 x 1)
        loss_ch: channel-wise loss
        loss_w: weighted loss

    """

    """
    Reduce NCHW channel wise. Division over numel and multiply by ch_out is not needed inside this method, but if
    you want to use loss_wch, or loss_ch directly the numbers would be off by a factor
    """
    ch_out = len(layer)
    loss_ch = loss.sum(-1).sum(-1).sum(0) / loss.numel() * ch_out
    head_grads = torch.zeros((ch_out,)).to(loss.device)
    weighting = torch.ones_like(head_grads).to(loss.device)

    for i in range(ch_out):
        head_grads[i] = torch.autograd.grad(loss_ch[i], layer[i].out_conv.weight, retain_graph=True)[0].abs().sum()

    """Kill the channels which are completely inactive"""
    ix_on = head_grads != 0.
    weighting[~ix_on] = 0.  # set excluded to zero

    optimizer.zero_grad()
    N = (1 / head_grads[ix_on]).sum()
    weighting[ix_on] = weighting[ix_on] / head_grads[ix_on]
    weighting = weighting / N

    loss_wch = (loss_ch * weighting).sum()
    weight_cX_h1_w1 = weighting.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # weight tensor of size 1 x C x 1 x 1

    return weight_cX_h1_w1, loss_ch, loss_wch
