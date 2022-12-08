import math
import torch

from torch import nn as nn


def rmse(xyz: torch.Tensor, xyz_ref: torch.Tensor) -> tuple[float, ...]:
    """
    Root mean squared distances

    Args:
        xyz:
        xyz_ref:

    Returns:
        - rmse lateral
        - rmse axial
        - rmse volumetric
    """
    num_tp = xyz.size(0)
    num_gt = xyz_ref.size(0)

    if num_tp != num_gt:
        raise ValueError("The number of points must match.")

    if xyz.size(1) not in (2, 3):
        raise NotImplementedError("Unsupported dimension")

    if num_tp == 0:
        return (torch.ones(1) * float("nan"),) * 3

    mse_loss = nn.MSELoss(reduction="sum")

    rmse_lat = (
        (mse_loss(xyz[:, 0], xyz_ref[:, 0]) + mse_loss(xyz[:, 1], xyz_ref[:, 1]))
        / num_tp
    ).sqrt()

    rmse_axial = (mse_loss(xyz[:, 2], xyz_ref[:, 2]) / num_tp).sqrt()
    rmse_vol = (mse_loss(xyz, xyz_ref) / num_tp).sqrt()

    return rmse_lat.item(), rmse_axial.item(), rmse_vol.item()


def mad(xyz: torch.Tensor, xyz_ref: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Mean absolute distances

    Args:
        xyz:
        xyz_ref:

    Returns:
        - mad lateral
        - mad axial
        - mad volumetric
    """
    num_tp = xyz.size(0)
    num_gt = xyz_ref.size(0)

    if num_tp != num_gt:
        raise ValueError("The number of points must match.")

    if xyz.size(1) not in (2, 3):
        raise NotImplementedError("Unsupported dimensions")

    if num_tp == 0:
        return (torch.ones(1) * float("nan"),) * 3

    mad_loss = nn.L1Loss(reduction="sum")

    mad_vol = mad_loss(xyz, xyz_ref) / num_tp
    mad_lat = (mad_loss(xyz[:, 0], xyz_ref[:, 0]) + mad_loss(xyz[:, 1], xyz_ref[:, 1])) \
              / num_tp
    mad_axial = mad_loss(xyz[:, 2], xyz_ref[:, 2]) / num_tp

    return mad_lat.item(), mad_axial.item(), mad_vol.item()


def precision(tp: int, fp: int) -> float:
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return math.nan


def recall(tp: int, fn: int) -> float:
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return math.nan


def jaccard(tp: int, fp: int, fn: int) -> float:
    try:
        return tp / (tp + fp + fn)
    except ZeroDivisionError:
        return math.nan


def f1(tp: int, fp: int, fn: int):
    prec = precision(tp=tp, fp=fp)
    rec = recall(tp=tp, fn=fn)
    try:
        return (2 * prec * rec) / (prec + rec)
    except ZeroDivisionError:
        return math.nan


def efficiency(jac: float, rmse: float, alpha: float) -> float:
    """
    Calculate Efficiency following Sage et al. 2019, superres fight club

    Args:
        jac (float): jaccard index 0-1
        rmse (float) RMSE value
        alpha (float): alpha value

    Returns:
        effcy (float): efficiency 0-1
    """
    return (100 - ((100 * (1 - jac)) ** 2 + alpha**2 * rmse**2) ** 0.5) / 100
