import torch


def tens_almeq(a, b, prec=1e-8):
    """
    Tests if two tensors are almost equal within prec as provided.
    :param a: tensor a
    :param b: tensor b
    :param prec: precision
    """
    if a.type() != b.type():
        raise TypeError("Both tensors must be of equal type.")

    if a.type != torch.FloatTensor:
        a = a.float()
        b = b.float()

    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), prec))


def tens_seq(a, b):
    if a.type() != b.type():
        raise TypeError("Both tensors must be of equal type.")

    if a.type != torch.FloatTensor:
        a = a.float()
        b = b.float()

    return ((a - b) <= 0).all().item()


def tens_eqshape(a, b):
    """Checks whether a and b are of same shape."""
    a_shape = torch.tensor(a.shape)
    b_shape = torch.tensor(b.shape)
    return torch.all(torch.eq(a_shape, b_shape))


def at_least_one_dim(*args):
    for arg in args:
        if arg.dim() == 0:
            arg.unsqueeze_(0)


def same_shape_tensor(dim, *args):
    for i in range(args.__len__() - 1):
        if args[i].size(dim) == args[i + 1].size(dim):
            continue
        else:
            return False

    return True


def same_dim_tensor(*args):
    for i in range(args.__len__() - 1):
        if args[i].dim() == args[i + 1].dim():
            continue
        else:
            return False

    return True