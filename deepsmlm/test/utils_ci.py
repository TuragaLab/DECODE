import torch


def tens_almeq(a, b, prec=1e-8):
    """
    Tests if two tensors are almost equal within prec as provided.
    :param a: tensor a
    :param b: tensor b
    :param prec: precision
    """
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), prec))