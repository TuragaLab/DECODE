import torch


def rescale_last_layer_grad(heads, loss, optimizer):
    """

    :param loss: non-reduced loss of size N x C x H x W
    :param optimizer:
    :return: weight, channelwise loss, channelwise weighted loss
    """
    """
    Reduce NCHW channel wise. Division over numel and multiply by ch_out is not needed inside this method, but if
    you want to use loss_wch, or loss_ch directly the numbers would be off by a factor
    """
    ch_out = heads.__len__()
    loss_ch = loss.sum(-1).sum(-1).sum(0) / loss.numel() * ch_out
    head_grads = torch.zeros((ch_out,)).to(loss.device)
    weighting = torch.ones_like(head_grads).to(loss.device)

    for i in range(ch_out):
        head_grads[i] = torch.autograd.grad(loss_ch[i], heads[i].out_conv.weight, retain_graph=True)[
            0].abs().sum()
    optimizer.zero_grad()
    N = (1 / head_grads).sum()
    weighting = weighting / head_grads
    weighting = weighting / N

    loss_wch = (loss_ch * weighting).sum()
    weight_cX_h1_w1 = weighting.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # weight tensor of size 1 x C x 1 x 1

    return weight_cX_h1_w1, loss_ch, loss_wch