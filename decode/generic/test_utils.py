import torch
import pathlib
import hashlib

from typing import Union


def tens_almeq(a: torch.Tensor, b: torch.Tensor, prec: float = 1e-8, nan: bool = False) -> bool:
    """
    Tests if a and b are equal (i.e. all elements are the same) within a given precision. If both tensors have / are
    nan, the function will return False unless nan=True.

    Args:
        a: first tensor for comparison
        b: second tensor for comparison
        prec: precision comparison
        nan: if true, the function will return true if both tensors are all nan

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

    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), prec)).item()


def open_n_hash(file: Union[str, pathlib.Path]) -> str:
    """
    Check SHA 256 hash of file

    Args:
        file:

    Returns:
        str

    """

    if not isinstance(file, pathlib.Path):
        file = pathlib.Path(file)
    hash_str = hashlib.sha256(file.read_bytes()).hexdigest()

    return hash_str
