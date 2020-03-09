import math
import torch

from deprecated import deprecated
from torch import nn as nn


def rmse_mad_dist(xyz_0: torch.Tensor, xyz_1: torch.Tensor) -> tuple:
    """
    Calculate RMSE and mean absolute distance.

    Args:
        xyz_0:
        xyz_1:

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


def precision_recall_jaccard(tp: int, fp: int, fn: int):
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


"""Deprecated stuff. Can and will be removed without warning at any time."""


@deprecated("Not used? Write a test before reactivating this function.")
def min_pair_distances(x, y, threshold=float('inf')):
    """
    Function to find closest pairs of points from x 2 y and vice versa.
    Closest points have to be found within distance maximum.

    param x: M x D matrix
    param y: N x D matrix

    dist_a_x2ally_t is the minimum distance of a specific x to all y,
        given that there was a y within the search radius

    ix_x_has_y is the list of all x which had a y close by
    ix_y_a_x is the corresponding index of a y which is the closest to a x
    """
    dist_mat = torch.cdist(x, y)

    dist_a_x2ally = dist_mat.min(1)
    ix_x_has_y = (dist_a_x2ally[0] <= threshold).nonzero().squeeze()
    ix_y_a_x = dist_a_x2ally[1][dist_a_x2ally[0] <= threshold]
    dist_a_x2ally_t = dist_a_x2ally[0][ix_x_has_y]

    dist_a_y2allx = dist_mat.min(0)
    ix_y_has_x = (dist_a_y2allx[0] <= threshold).nonzero().squeeze()
    ix_x_a_y = dist_a_y2allx[1][dist_a_y2allx[0] <= threshold]
    dist_a_y2allx_t = dist_a_y2allx[0][ix_y_has_x]

    return dist_a_x2ally_t, ix_x_has_y, ix_y_a_x, dist_a_y2allx_t, ix_y_has_x, ix_x_a_y


@deprecated("Not used? Write a test before reactivating this function.")
def interpoint_loss(output, target, reduction='mean'):
    """
    Returns distance to closest target point
    """
    interpoint_dist = torch.cdist(output, target)

    if reduction is None:
        return interpoint_dist.min(1)[0]
    elif reduction == 'mean':
        # return distance to closest target point
        return interpoint_dist.min(1)[0].sum() / output.__len__()
    elif reduction == 'sum':
        return interpoint_dist.min(1)[0].sum()
    else:
        raise ValueError('Reduction type unsupported.')


@deprecated("Not used? Write a test before reactivating this function.")
def iterative_pos_neg(output, target, distance_threshold=1.):
    """
    return ix_pred2gt: Gives the index of the prediction ix_pred2gt[i] which is best for gt[i]
    """
    dist_mat = torch.cdist(output, target)

    is_true_positive = torch.zeros_like(target[:, 0]).type(torch.BoolTensor)
    is_false_positive = torch.zeros_like(output[:, 0]).type(torch.BoolTensor)
    is_false_negative = torch.zeros_like(is_true_positive)

    ix_gt = []
    ix_pred2gt = []  # torch.zeros_like(target[:, 0])

    for i in range(dist_mat.shape[1]):  # iterate over ground truth
        if dist_mat[:, i].min(0)[0] <= distance_threshold:
            is_true_positive[i] = 1
            ix_gt.append(i)
            ix_pred2gt.append(dist_mat[:, i].min(0)[1])
            dist_mat[dist_mat[:, i].min(0)[1], :] = float('inf')
        else:
            is_false_negative[i] = 1

    not_assigned_pred = (((dist_mat == float('inf')).sum(
        1) == dist_mat.shape[1]) == 0).nonzero()
    is_false_positive[not_assigned_pred] = 1

    ix_gt = torch.LongTensor(ix_gt)
    ix_pred2gt = torch.LongTensor(ix_pred2gt)

    ix_gt2pred = []
    ix_pred = []

    return is_true_positive, is_false_positive, is_false_negative, ix_pred2gt, ix_gt, ix_gt2pred, ix_pred


@deprecated("Deprecated in favour of torch.cdist")
def expanded_pairwise_distances(x, y=None, sqd=False):
    '''
    Taken from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
        differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0)

    distances = torch.sum(differences * differences, -1)
    if sqd is False:
        distances = distances.sqrt()
    return distances


@deprecated("Not used? Write a test before reactivating this function.")
def pos_neg_emitters(output, target, distance_threshold=1.):
    dist_mat = interpoint_loss(output, target, reduction=None)

    # true pos/neg --- closer than a certain px threshold
    tp = dist_mat <= distance_threshold

    # false pos/neg
    fp = dist_mat > distance_threshold

    # false negatives
    dist_mat_ = interpoint_loss(target, output, reduction=None)

    fn = dist_mat_ > distance_threshold

    # match index of true positive -- gt
    exp_dist = torch.cdist(output, target)

    ix_gt = (exp_dist.min(0)[0] <= distance_threshold).nonzero().squeeze()
    ix_pred2gt = exp_dist.min(0)[1][exp_dist.min(0)[0] <= distance_threshold]

    ix_pred = (exp_dist.min(1)[0] <= distance_threshold).nonzero().squeeze()
    ix_gt2pred = exp_dist.min(1)[1][exp_dist.min(1)[0] <= distance_threshold]

    return tp, fp, fn, ix_pred2gt, ix_gt, ix_gt2pred, ix_pred
