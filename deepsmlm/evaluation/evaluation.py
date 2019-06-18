import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import sqrt

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from deepsmlm.evaluation.metric_library import pos_neg_emitters, PrecisionRecallJaccard, RMSEMAD
import deepsmlm.evaluation.metric_library as metric_lib
import deepsmlm.generic.emitter as emitter


class MetricMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = None
        self.vals = None
        self.count = None
        self.reset()

    @property
    def std(self):
        return torch.tensor(self.vals).std().item()

    @property
    def mean(self):
        return torch.tensor(self.vals).mean().item()

    @property
    def avg(self):
        return self.mean

    def hist(self, bins=30, range=None):
        _ = plt.hist(self.vals.view(-1).numpy(), bins=30, range=range)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        return ax

    def reset(self):
        """
        Reset instance.
        :return:
        """
        self.val = None
        self.vals = []
        self.count = 0

    def update(self, val):
        """
        Update AverageMeter.

        :param val: value
        :return: None
        """
        self.val = val
        self.vals.append(val)
        self.count += 1

    def __str__(self):
        return "(avg) - (sig) = {:.3f} - {:.3f}".format(self.avg, self.std)


class CumulantMeter(MetricMeter):
    def __init__(self):
        super().__init__()
        self.vals = torch.empty(0)

    def update(self, val):
        """
        Assuming a 1D tensor
        :param val:
        :return:
        """

        self.vals = torch.cat((self.vals, val), 0)


class EvalSet:
    """Just a dummy class to combine things into one object."""
    def __init__(self, prec, rec, jac,
                 rmse_vol, rmse_lat, rmse_axial,
                 mad_vol, mad_lat, mad_axial,
                 dx, dy, dz, dxw, dyw, dzw):
        self.prec = prec
        self.rec = rec
        self.jac = jac

        self.rmse_vol = rmse_vol
        self.rmse_lat = rmse_lat
        self.rmse_axial = rmse_axial
        self.mad_vol = mad_vol
        self.mad_lat = mad_lat
        self.mad_axial = mad_axial

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dxw = dxw
        self.dyw = dyw
        self.dzw = dzw

    def __str__(self):
        str = "------------ Evaluation Set ------------\n"
        str += "Precision {}\n".format(self.prec.__str__())
        str += "Recall {}\n".format(self.rec.__str__())
        str += "Jaccard {}\n".format(self.jac.__str__())
        str += "RMSE lat. {}\n".format(self.rmse_lat.__str__())
        str += "RMSE ax. {}\n".format(self.rmse_axial.__str__())
        str += "RMSE vol. {}\n".format(self.rmse_vol.__str__())
        str += "MAD lat. {}\n".format(self.mad_lat.__str__())
        str += "MAD ax. {}\n".format(self.mad_axial.__str__())
        str += "MAD vol. {}\n".format(self.mad_vol.__str__())
        str += "-----------------------------------------"
        return str


class BatchEvaluation:

    def __init__(self, matching, segmentation_eval, distance_eval):
        self._matching = matching
        self._segmentation_eval = segmentation_eval
        self._distance_eval = distance_eval
        self._distance_delta = metric_lib.Deltas()

        self.values = None

    def forward(self, output, target):
        """
        Evaluate metrics on a whole batch, i.e. average the stuff. This is based lists of outputs, targets.

        :param output:
        :param target:
        :return:
        """
        prec, rec, jac = MetricMeter(), MetricMeter(), MetricMeter()
        rmse_vol, rmse_lat, rmse_axial = MetricMeter(), MetricMeter(), MetricMeter()
        mad_vol, mad_lat, mad_axial = MetricMeter(), MetricMeter(), MetricMeter()
        dx, dy, dz, dxw, dyw, dzw = CumulantMeter(), CumulantMeter(), CumulantMeter(), \
                                    CumulantMeter(), CumulantMeter(), CumulantMeter()

        if output.__len__() != target.__len__():
            raise ValueError("Output and Target batch size must be of equal length.")

        for i in range(output.__len__()):
            tp, fp, fn, tp_match = self._matching.forward(output[i], target[i])
            prec_, rec_, jaq_ = self._segmentation_eval.forward(tp, fp, fn)
            rmse_vol_, rmse_lat_, rmse_axial_, mad_vol_, mad_lat_, mad_axial_ = self._distance_eval.forward(tp, tp_match)
            dx_, dy_, dz_, dxw_, dyw_, dzw_ = self._distance_delta.forward(tp, tp_match)

            prec.update(prec_)
            rec.update(rec_)
            jac.update(jaq_)
            rmse_vol.update(rmse_vol_)
            rmse_lat.update(rmse_lat_)
            rmse_axial.update(rmse_axial_)
            mad_vol.update(mad_vol_)
            mad_lat.update(mad_lat_)
            mad_axial.update(mad_axial_)

            dx.update(dx_)
            dy.update(dy_)
            dz.update(dz_)
            dxw.update(dxw_)
            dyw.update(dyw_)
            dzw.update(dzw_)

        self.values = EvalSet(prec, rec, jac,
                              rmse_vol, rmse_lat, rmse_axial,
                              mad_vol, mad_lat, mad_axial,
                              dx, dy, dz, dxw, dyw, dzw)


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

        self.nearest_neigh = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2)

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

        nbrs = self.nearest_neigh.fit(xyz_tar_)

        """If no emitter has been found, all are false negatives. No tp, no fp."""
        if xyz_out_.shape[0] == 0:
            tp = emitter.EmptyEmitterSet()
            fp = emitter.EmptyEmitterSet()
            fn = target
            tp_match = emitter.EmptyEmitterSet()

            return tp, fp, fn, tp_match

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

    def forward(self, tp, fp, fn):
        """
        Run the complete segmentation evaluation for two arbitrary emittersets.
        This disregards the frame_ix

        :param tp: (instance of emitterset) true postiives
        :param fp:  (instance of emitterset) false positives
        :param fn: (instance of emitterset) false negatives
        :return: several metrics
        """

        prec, rec, jac = PrecisionRecallJaccard.forward(tp.num_emitter, fp.num_emitter, fn.num_emitter)
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

