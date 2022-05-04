import hashlib
import pathlib
from typing import Union

import torch


def equal_none(*args):
    """tests if all arguments are none"""
    none = {a is None for a in args}
    return none == {True}


def equal_optional(*args):
    """tests if either all are none or all are not none"""
    none = {a is None for a in args}
    return len(none) == 1


def tens_almeq(
    a: torch.Tensor,
    b: torch.Tensor,
    prec: float = 1e-8,
    nan: bool = False,
    none: str = "raise",
) -> bool:
    """
    Tests if a and b are equal (i.e. all elements are the same) within a given precision.
    If both tensors have / are nan, the function will return False unless nan=True.

    Args:
        a: first tensor for comparison
        b: second tensor for comparison
        prec: precision comparison
        nan: if true, the function will return true if both tensors are all nan
        none: either `raise`,`both`,`either`.
            for `raise` no None values are allowed,
            for `both` True will be returned when both arguments are None,
                if either is and the other is not, there will be a ValueError
            for `either` False will be returned when only one is None while the other is not.
    """
    if (a is None or b is None) and none != "raise":
        if equal_none(a, b):  # all none
            return True
        # not equal none
        if none == "both":
            raise ValueError("One of the arguments is None while the other is not.")
        if none == "either":
            return False

    if a.type() != b.type():
        raise TypeError("Both tensors must be of equal type.")

    if a.type != torch.FloatTensor:
        a = a.float()
        b = b.float()

    # equal nan values
    if nan:
        if torch.isnan(a).all() and torch.isnan(b).all():
            return True

    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), prec)).item()


def open_n_hash(file: Union[str, pathlib.Path]) -> str:
    """
    Check SHA 256 hash of file
    """

    if not isinstance(file, pathlib.Path):
        file = pathlib.Path(file)
    hash_str = hashlib.sha256(file.read_bytes()).hexdigest()

    return hash_str


def file_loadable(
    path: Union[str, pathlib.Path], reader=None, mode=None, exceptions=None
) -> bool:
    """
    Check whether file is present and loadable.
    This function could be used in a while lood and sleep.

    Example:
        while not file_loadable(path, ...):
            time.sleep()
    """
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if not path.is_file():
        return False

    # try to actually load the file (or the handle)
    if reader is not None:
        try:
            if mode is not None:
                reader(path, mode=mode)
            else:
                reader(path)
            return True

        except exceptions:
            return False


def same_weights(model1, model2) -> bool:
    """Tests whether model1 and 2 have the same weights."""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def same_shape_tensor(dim, *args) -> bool:
    """Test if tensors are of same size in a certain dimension."""
    for i in range(args.__len__() - 1):
        if args[i].size(dim) == args[i + 1].size(dim):
            continue
        else:
            return False

    return True


def same_dim_tensor(*args) -> bool:
    """Test if tensors are of same dimensionality"""
    for i in range(args.__len__() - 1):
        if args[i].dim() == args[i + 1].dim():
            continue
        else:
            return False

    return True
