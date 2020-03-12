import torch


def tens_almeq(a: torch.Tensor, b: torch.Tensor, prec: float = 1e-8, nan: bool = False):
    """
    Tests if a and b are equal (i.e. all elements are the same) within a given precision. If both tensors have / are
    nan, the function will return False unless nan=True.

    Args:
        a (torch.Tensor):
        b (torch.Tensor):
        prec (float):
        nan (bool): if true, the function will return true if both tensors are all nan

    Returns:
        bool

    """
    if a.type() != b.type():
        raise TypeError("Both tensors must be of equal type.")

    if a.type != torch.FloatTensor:
        a = a.float()
        b = b.float()

    if nan:
        if torch.isnan(a).all() and torch.isnan(b).all():
            return True

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


def equal_nonzero(*a):
    """
    Test whether a and b have the same non-zero elements
    :param a: tensors
    :return: "torch.bool"
    """
    is_equal = torch.equal(a[0], a[0])
    for i in range(a.__len__() - 1):
        is_equal = is_equal * torch.equal(a[i].nonzero(), a[i + 1].nonzero())

    return is_equal