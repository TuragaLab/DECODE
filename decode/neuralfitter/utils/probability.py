import torch


def binom_pdiverse(p):
    """
    binomial probability but unequal probabilities
    Args:
        p: (torch.Tensor) of probabilities

    Returns:
        z: (torch.Tensor) vector of probabilities with length p.size() + 1

    """
    n = p.size(0) + 1
    z = torch.zeros((n,))
    z[0] = 1

    for u in p:
        z = (1 - u) * z + u * torch.cat((torch.zeros((1,)), z[:-1]))

    return z