import math
import torch

from torch import nn as nn
from typing import Tuple


def rmse_mad_dist(xyz_0: torch.Tensor, xyz_1: torch.Tensor) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate RMSE and mean absolute distance.

    Args:
        xyz_0: coordinates of set 0,
        xyz_1: coordinates of set 1

    Returns:
        rmse_lat (float): RMSE lateral
        rmse_ax (float): RMSE axial
        rmse_vol (float): RMSE volumetric
        mad_lat (float): Mean Absolute Distance lateral
        mad_ax (float): Mean Absolute Distance axial
        mad_vol (float): Mean Absolute Distance vol
    """

    num_tp = xyz_0.size(0)
    num_gt = xyz_1.size(0)

    if num_tp != num_gt:
        raise ValueError("The number of points must match.")

    if xyz_0.size(1) not in (2, 3):
        raise ValueError("Unsupported ")

    if num_tp == 0:
        return (float('nan'),) * 6

    mse_loss = nn.MSELoss(reduction='sum')

    rmse_lat = ((mse_loss(xyz_0[:, 0], xyz_1[:, 0]) +
                 mse_loss(xyz_0[:, 1], xyz_1[:, 1])) / num_tp).sqrt()

    rmse_axial = (mse_loss(xyz_0[:, 2], xyz_1[:, 2]) / num_tp).sqrt()
    rmse_vol = (mse_loss(xyz_0, xyz_1) / num_tp).sqrt()

    mad_loss = nn.L1Loss(reduction='sum')

    mad_vol = mad_loss(xyz_0, xyz_1) / num_tp
    mad_lat = (mad_loss(xyz_0[:, 0], xyz_1[:, 0]) + mad_loss(xyz_0[:, 1], xyz_1[:, 1])) / num_tp
    mad_axial = mad_loss(xyz_0[:, 2], xyz_1[:, 2]) / num_tp

    return rmse_lat.item(), rmse_axial.item(), rmse_vol.item(), mad_lat.item(), mad_axial.item(), mad_vol.item()


def precision_recall_jaccard(tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    """
    Calculates precision, recall, jaccard index and f1 score

    Args:
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives

    Returns:
        precision (float): precision value 0-1
        recall (float): recall value 0-1
        jaccard (float): jaccard index 0-1
        f1 (float): f1 score 0-1

    """

    # convert to float as safety measure
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    precision = math.nan if (tp + fp) == 0 else tp / (tp + fp)
    recall = math.nan if (tp + fn) == 0 else tp / (tp + fn)
    jaccard = math.nan if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
    f1score = math.nan if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return precision, recall, jaccard, f1score


def efficiency(jac: float, rmse: float, alpha: float):
    """
    Calculate Efficiency following Sage et al. 2019, superres fight club

    Args:
        jac (float): jaccard index 0-1
        rmse (float) RMSE value
        alpha (float): alpha value

    Returns:
        effcy (float): efficiency 0-1
    """
    return (100 - ((100 * (1 - jac)) ** 2 + alpha ** 2 * rmse ** 2) ** 0.5) / 100
