import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors
from math import sqrt
from deepsmlm.evaluation.metric_library import pos_neg_emitters, PrecisionRecallJacquard, RMSEMAD


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        """
        Reset instance.
        :return:
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update AverageMeter.

        :param val: value
        :param n: weight of sample
        :return: None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BatchEvaluation:

    def __init__(self, matching, segmentation_eval, distance_eval):
        self.matching = matching
        self.segmentation_eval = segmentation_eval
        self.distance_eval = distance_eval

    def forward(self, output, target):
        """
        Evaluate metrics on a whole batch, i.e. average the stuff. This is based lists of outputs, targets.

        :param output:
        :param target:
        :return:
        """
        prec, rec, jaq = AverageMeter(), AverageMeter(), AverageMeter()
        rmse_vol, rmse_lat, rmse_axial = AverageMeter(), AverageMeter(), AverageMeter()
        mad_vol, mad_lat, mad_axial = AverageMeter(), AverageMeter(), AverageMeter()

        if output.__len__() != target.__len__():
            raise ValueError("Output and Target batch size must be of equal length.")

        for i in range(output.__len__()):
            tp, fp, fn, tp_match = self.matching(output[i], target[i])
            prec_, rec_, jaq_ = self.segmentation_eval.forward(tp, fp, fn)
            rmse_vol_, rmse_lat_, rmse_axial_, mad_vol_, mad_lat_, mad_axial_ = self.distance_eval.forward()

            prec.update(prec_)
            rec.update(rec_)
            jaq.update(jaq_)
            rmse_vol.update(rmse_vol_)
            rmse_lat.update(rmse_lat_)
            rmse_axial.update(rmse_axial_)
            mad_vol.update(mad_vol_)
            mad_lat.update(mad_lat_)
            mad_axial.update(mad_axial_)


class NNMatching:
    """
    A class to match outputs and targets based on 1neighbor nearest neighbor classifier.
    """
    def __init__(self, dist_lat=2.5, dist_ax=500, match_dims=3):
        """

        :param dist_lat: (float) lateral distance threshold
        :param dist_ax: (float) axial distance threshold
        :param match_dims: should we match the emitters only in 2D or also 3D
        """
        self.dist_lat_thresh = dist_lat
        self.dist_ax_thresh = dist_ax
        self.match_dims = match_dims

        if self.match_dims not in [2, 3]:
            raise ValueError("You must compare in either 2 or 3 dimensions.")

    def forward(self, output, target):
        """Forward arbitrary output and target set. Does not care about the frame_ix.

        :param output: (emitterset)
        :param target: (emitterset)
        :return tp: (emitterset) true positives
        :return fp: (emitterset) false positives
        :return tp_match: (emitterset) the ground truth points which are matched to the true positives.
        """
        xyz_tar = target.xyz.numpy()
        xyz_out = output.xyz.numpy()

        if self.match_dims == 2:
            xyz_tar_ = xyz_tar[:, :2]
            xyz_out_ = xyz_out[:, :2]
        else:
            xyz_tar_ = xyz_tar
            xyz_out_ = xyz_out

        nbrs = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(xyz_tar_)
        distances_nn, indices = nbrs.kneighbors(xyz_out_)

        distances_nn = distances_nn.squeeze()
        indices = np.atleast_1d(indices.squeeze())

        xyz_match = target.get_subset(indices).xyz.numpy()

        # calculate distances lateral and axial seperately
        dist_lat = np.linalg.norm(xyz_out[:, :2] - xyz_match[:, :2], axis=1, ord=2)
        dist_ax = np.linalg.norm(xyz_out[:, [2]] - xyz_match[:, [2]], axis=1, ord=2)

        # remove those which are too far
        if self.match_dims == 3:
            is_tp = (dist_lat <= self.dist_lat_thresh) * (dist_ax <= self.dist_ax_thresh)
        elif self.match_dims == 2:
            is_tp = (dist_lat <= self.dist_lat_thresh)

        is_fp = ~is_tp

        indices[is_fp] = -1
        indices_cleared = indices[indices != -1]

        # create indices of targets
        tar_ix = np.arange(target.num_emitter)
        # remove indices which were found
        fn_ix = np.setdiff1d(tar_ix, indices)

        is_tp = torch.from_numpy(is_tp.astype(np.uint8)).type(torch.ByteTensor)
        is_fp = torch.from_numpy(is_fp.astype(np.uint8)).type(torch.ByteTensor)
        fn_ix = torch.from_numpy(fn_ix)

        tp = output.get_subset(is_tp)
        fp = output.get_subset(is_fp)
        fn = target.get_subset(fn_ix)
        tp_match = target.get_subset(indices_cleared)

        return tp, fp, fn, tp_match


class SegmentationEvaluation:
    """
    Evaluate performance on finding the right emitters.
    """
    def __init__(self, print_mode=True):
        """
        """
        self.print_mode = print_mode

    def forward_frame(self, tp, fp, fn):
        """
        Run the complete segmentation evaluation for two arbitrary emittersets.
        This disregards the frame_ix

        :param tp: (instance of emitterset) true postiives
        :param fp:  (instance of emitterset) false positives
        :param fn: (instance of emitterset) false negatives
        :return: several metrics
        """

        prec, rec, jac = PrecisionRecallJacquard.forward(tp.num_emitter, fp.num_emitter, fn.num_emitter)
        actual_em = tp.num_emitter + fn.num_emitter
        pred_em = tp.num_emitter + fp.num_emitter

        if self.print_mode:
            print("Number of actual emitters: {} Predicted emitters: {}".format(actual_em, pred_em))
            print("Number of TP: {} FP: {} FN: {}".format(tp.num_emitter,
                                                          fp.num_emitter,
                                                          fn.num_emitter))
            print("Jacquard: {:.3f}".format(jac))
            print("Precision: {:.3f}, Recall: {:.3f}".format(prec, rec))
        return prec, rec, jac


class DistanceEvaluation:
    """
    Evaluate performance on how precise we are.
    """
    def __init__(self, print_mode=True):
        """
        :param print_mode: Print the evaluation to the console.
        """
        self.print_mode = print_mode

    def forward(self, output, target):
        """

        :param output: (instance of Emitterset)
        :param target: (instance of Emitterset)
        :return:
        """

        rmse_vol, rmse_lat, rmse_axial, mad_vol, mad_lat, mad_axial = RMSEMAD.forward(target, output)

        if self.print_mode:
            print("RMSE: Vol. {:.3f} Lat. {:.3f} Axial. {:.3f}".format(rmse_vol, rmse_lat, rmse_axial))
            print("MAD: Vol. {:.3f} Lat. {:.3f} Axial. {:.3f}".format(mad_vol, mad_lat, mad_axial))

        return rmse_vol, rmse_lat, rmse_axial, mad_vol, mad_lat, mad_axial

