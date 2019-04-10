import torch
from torch import nn as nn
from math import sqrt


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


def test_expanded_pairwise_distances():
    passed = True
    try:
        x = torch.tensor([[0., 0., 0.]])
        y = torch.tensor([[0., 0., 0.], [1., 1., 1.]])

        dist_mat = expanded_pairwise_distances(x, y)
        passed *= (dist_mat == torch.tensor([[0., sqrt(3)]])).all()
    except:
        passed = False

    print("Test passed [expanded_pairwise_distance]: {}".format(passed))
    return


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
    dist_mat = expanded_pairwise_distances(x, y)

    dist_a_x2ally = dist_mat.min(1)
    ix_x_has_y = (dist_a_x2ally[0] <= threshold).nonzero().squeeze()
    ix_y_a_x = dist_a_x2ally[1][dist_a_x2ally[0] <= threshold]
    dist_a_x2ally_t = dist_a_x2ally[0][ix_x_has_y]

    dist_a_y2allx = dist_mat.min(0)
    ix_y_has_x = (dist_a_y2allx[0] <= threshold).nonzero().squeeze()
    ix_x_a_y = dist_a_y2allx[1][dist_a_y2allx[0] <= threshold]
    dist_a_y2allx_t = dist_a_y2allx[0][ix_y_has_x]

    return dist_a_x2ally_t, ix_x_has_y, ix_y_a_x, dist_a_y2allx_t, ix_y_has_x, ix_x_a_y


def iterative_pos_neg(output, target, distance_threshold=1.):
    """
    return ix_pred2gt: Gives the index of the prediction ix_pred2gt[i] which is best for gt[i]
    """
    dist_mat = expanded_pairwise_distances(output, target)

    is_true_positive = torch.zeros_like(target[:, 0], dtype=torch.uint8)
    is_false_positive = torch.zeros_like(output[:, 0], dtype=torch.uint8)
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
    exp_dist = expanded_pairwise_distances(output, target)

    ix_gt = (exp_dist.min(0)[0] <= distance_threshold).nonzero().squeeze()
    ix_pred2gt = exp_dist.min(0)[1][exp_dist.min(0)[0] <= distance_threshold]

    ix_pred = (exp_dist.min(1)[0] <= distance_threshold).nonzero().squeeze()
    ix_gt2pred = exp_dist.min(1)[1][exp_dist.min(1)[0] <= distance_threshold]

    return tp, fp, fn, ix_pred2gt, ix_gt, ix_gt2pred, ix_pred


def precision_recall(tp, fp, fn):
    """
    Calculates precision and recall.
    :param tp: (int) number of true positives
    :param fp: (int) number of false positives
    :param fn: (int) number of false negatives
    :return: precision (float), recall (float)
    """

    # convert to float, because otherwise torch division is integer based ...
    tp = tp.to(dtype=torch.float)
    fp = fp.to(dtype=torch.float)
    fn = fn.to(dtype=torch.float)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def interpoint_loss(output, target, reduction='mean'):
    """
    Returns distance to closest target point
    """
    interpoint_dist = expanded_pairwise_distances(output, target)  # something fishy here

    if reduction is None:
        return interpoint_dist.min(1)[0]
    elif reduction is 'mean':
        # return distance to closest target point
        return interpoint_dist.min(1)[0].sum() / output.__len__()
    elif reduction is 'sum':
        return interpoint_dist.min(1)[0].sum()
    else:
        raise ValueError('Reduction type unsupported.')


def rmse_mad(tp, r):
    """
    Calculate RMSE values and mad.

    :param tp: (emitterset) true positives
    :param r:  (emitterset) reference
    :return: various rmse values
    """
    num_tp = tp.num_emitter
    # convenience to get the coordinates
    tp_ = tp.xyz
    r_ = r.xyz

    mse_loss = nn.MSELoss(reduction='sum')

    rmse_vol = (mse_loss(tp_, r_) / num_tp).sqrt()
    rmse_lat = ((mse_loss(tp_[:, 0], r_[:, 0]) +
                 mse_loss(tp_[:, 1], r_[:, 1])) / num_tp).sqrt()

    rmse_axial = (mse_loss(tp_[:, 2], r_[:, 2]) / num_tp).sqrt()

    mad_loss = nn.L1Loss(reduction='sum')

    mad_vol = mad_loss(tp_, r_) / num_tp
    mad_lat = (mad_loss(tp_[:, 0], r_[:, 0]) + mad_loss(tp_[:, 0], r_[:, 0])) / num_tp
    mad_axial = mad_loss(tp_[:, 2], r_[:, 2]) / num_tp

    return rmse_vol, rmse_lat, rmse_axial, mad_vol, mad_lat, mad_axial


if __name__ == '__main__':
    test_expanded_pairwise_distances()
