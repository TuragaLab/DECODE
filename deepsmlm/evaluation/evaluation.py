import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
import seaborn as sns

import numpy as np
import math
import torch

from deepsmlm.evaluation.metric_library import PrecisionRecallJaccard, RMSEMAD
import deepsmlm.evaluation.metric_library as metric_lib
import deepsmlm.generic.emitter as emitter
from scipy.stats import gaussian_kde


class MetricMeter:
    """Computes and stores the average and current value"""
    def __init__(self, reduce_nan=True):
        self.val = None
        self.vals = None
        self.count = None
        self.reset()
        self.reduce_nan = reduce_nan

    @property
    def std(self):
        if isinstance(self.vals, torch.Tensor):
            self.vals.std().item()
        else:
            return torch.tensor(self.vals).std().item()

    @property
    def mean(self):
        if isinstance(self.vals, torch.Tensor):
            return self.vals.mean().item()
        else:
            return torch.tensor(self.vals).mean().item()

    @property
    def avg(self):
        return self.mean

    def hist(self, bins=30, range=None, fit=stats.norm):
        """

        :param bins: number of bins
        :param range: specify range
        :return: plt.figure
        """
        if range is not None:
            bins_ = np.linspace(range[0], range[1], bins + 1)
        else:
            bins_ = bins

        vals = self.vals.view(-1).numpy()
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.1, .9)})

        """If we don't have values return empty figure"""
        if vals.shape[0] <= 2:
            return f

        """Plot boxplot and distplot."""
        sns.boxplot(vals, ax=ax_box)
        sns.distplot(vals, ax=ax_hist, kde=False, fit=fit, bins=bins_, norm_hist=True)

        """Get the fit values."""
        if fit is not None:
            (mu, sigma) = stats.norm.fit(vals)
            plt.legend(["N $ (\mu$ = {0:.3g}, $\sigma^2$ = {1:.3g}$^2$)".format(mu, sigma)], frameon=False)

        # Cosmetics
        ax_box.set(yticks=[])
        ax_box.xaxis.set_minor_locator(AutoMinorLocator())
        ax_hist.xaxis.set_minor_locator(AutoMinorLocator())
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True)
        return f

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

        val = float(val)
        if math.isnan(val) and self.reduce_nan:
            return

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
                 delta_num,
                 rmse_vol, rmse_lat, rmse_axial,
                 mad_vol, mad_lat, mad_axial,
                 dx, dy, dz, dxw, dyw, dzw):
        self.prec = prec
        self.rec = rec
        self.jac = jac

        self.delta_num = delta_num

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
        str = "------------------------ Evaluation Set ------------------------\n"
        str += "Precision {}\n".format(self.prec.__str__())
        str += "Recall {}\n".format(self.rec.__str__())
        str += "Jaccard {}\n".format(self.jac.__str__())
        str += "Delta num. emitters (out - tar.) {}\n".format(self.delta_num.__str__())
        str += "RMSE lat. {}\n".format(self.rmse_lat.__str__())
        str += "RMSE ax. {}\n".format(self.rmse_axial.__str__())
        str += "RMSE vol. {}\n".format(self.rmse_vol.__str__())
        str += "MAD lat. {}\n".format(self.mad_lat.__str__())
        str += "MAD ax. {}\n".format(self.mad_axial.__str__())
        str += "MAD vol. {}\n".format(self.mad_vol.__str__())
        str += "-----------------------------------------------------------------"
        return str


class BatchEvaluation:

    def __init__(self, matching, segmentation_eval, distance_eval, batch_size, px_size=None):
        self._matching = matching
        self._segmentation_eval = segmentation_eval
        self._distance_eval = distance_eval
        self._batch_size = batch_size
        self._distance_delta = metric_lib.Deltas()

        self.values = None
        self.px_size = px_size

    def forward(self, output, target):
        """
        Evaluate metrics on a whole batch, i.e. average the stuff. This is based lists of outputs, targets.

        :param output: list of emitter sets
        :param target: list of emitter sets
        :return:
        """
        prec, rec, jac = MetricMeter(), MetricMeter(), MetricMeter()
        delta_num = MetricMeter()
        rmse_vol, rmse_lat, rmse_axial = MetricMeter(), MetricMeter(), MetricMeter()
        mad_vol, mad_lat, mad_axial = MetricMeter(), MetricMeter(), MetricMeter()
        dx, dy, dz, dxw, dyw, dzw = CumulantMeter(), CumulantMeter(), CumulantMeter(), \
                                    CumulantMeter(), CumulantMeter(), CumulantMeter()

        if self.px_size is not None:
            output = output.convert_em(factor=self.px_size)
            target = target.convert_em(factor=self.px_size)

        tp, fp, fn, tp_match = self._matching.forward(output, target)
        prec_, rec_, jaq_ = self._segmentation_eval.forward(tp, fp, fn)
        rmse_vol_, rmse_lat_, rmse_axial_, mad_vol_, mad_lat_, mad_axial_ = self._distance_eval.forward(tp, tp_match)
        dx_, dy_, dz_, dxw_, dyw_, dzw_ = self._distance_delta.forward(tp, tp_match)

        prec.update(prec_)
        rec.update(rec_)
        jac.update(jaq_)

        delta_num.update(output.num_emitter - target.num_emitter)

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
                              delta_num,
                              rmse_vol, rmse_lat, rmse_axial,
                              mad_vol, mad_lat, mad_axial,
                              dx, dy, dz, dxw, dyw, dzw)

    def plot_sophisticated(self, output, target):
        """
        Plots more interesting stuff.
        """
        if self.px_size is not None:
            output = output.convert_em(factor=self.px_size)
            target = target.convert_em(factor=self.px_size)




class SegmentationEvaluation:
    """
    Evaluate performance on finding the right emitters.
    """
    def __init__(self, print_mode=True):
        """
        """
        self.print_mode = print_mode
        self.cached_result = [None] * 3

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

        self.cached_result[0] = prec
        self.cached_result[1] = rec
        self.cached_result[2] = jac
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

