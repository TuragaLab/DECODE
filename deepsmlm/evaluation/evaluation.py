# from abc import ABC
import torch
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

from collections import namedtuple
from deprecated import deprecated

from deepsmlm.evaluation.metric_library import precision_recall_jaccard, rmse_mad_dist
from ..generic import emitter as emitter
from ..generic.emitter import EmitterSet


class SegmentationEvaluation:
    """
    A small wrapper class that holds segementation evaluations and accepts emittersets as input.
    """

    _seg_eval_return = namedtuple("seg_eval", ["prec", "rec", "jac", "f1"])

    def __init__(self):
        """Bookkeeping"""

        self._tp = None
        self._fp = None
        self._fn = None
        self._prec = None
        self._rec = None
        self._jac = None
        self._f1 = None

    def __str__(self):
        if self._tp is None or self._fp is None or self._fn is None:
            return "Segmentation evaluation unavailable. Run .forward(tp, fp, fn)"

        actual_em = len(self._tp) + len(self._fn)
        pred_em = len(self._tp) + len(self._fp)

        str_repr = "Segmentation evaluation (cached values)\n"
        str_repr += f"Number of actual emitters: {actual_em} Predicted emitters: {pred_em}\n"
        str_repr += f"Number of TP: {len(self._tp)} FP: {len(self._fp)} FN: {len(self._fn)}\n"
        str_repr += f"Jacquard: {self._jac:.3f}\n"
        str_repr += f"F1Score: {self._f1:.3f}\n"
        str_repr += f"Precision: {self._prec:.3f}, Recall: {self._rec:.3f}\n"

        return str_repr

    def forward(self, tp: EmitterSet, fp: EmitterSet, fn: EmitterSet):
        """

        Args:
            tp (EmitterSet): true positives
            fp (EmitterSet): false positives
            fn (EmitterSet): false negatives

        Returns:
            prec (float): precision value
            rec (float): recall value
            jac (float): jaccard value
            f1 (float): f1 score value

        """

        prec, rec, jac, f1 = precision_recall_jaccard(len(tp), len(fp), len(fn))

        """Store last result to cache"""
        self._tp, self._fp, self._fn = tp, fp, fn
        self._prec, self._rec, self._jac, self._f1 = prec, rec, jac, f1

        return self._seg_eval_return(prec=prec, rec=rec, jac=jac, f1=f1)  # namedtuple


class DistanceEvaluation:
    """
    A small wrapper calss that holds distance evaluations and accepts emittersets as inputs.
    """
    _dist_eval_return = namedtuple("dist_eval", ["rmse_lat", "rmse_ax", "rmse_vol", "mad_lat", "mad_ax", "mad_vol"])

    def __init__(self):

        self._rmse_lat = None
        self._rmse_ax = None
        self._rmse_vol = None

        self._mad_lat = None
        self._mad_ax = None
        self._mad_vol = None

    def __str__(self):
        if self._rmse_lat is None:
            return "Distance Evaluation unavailable. Run .forward(tp, tp_match)."

        str_repr = "Distance Evaluation (cached values)\n"
        str_repr += f"RMSE: Lat. {self._rmse_lat:.3f} Axial. {self._rmse_ax:.3f} Vol. {self._rmse_vol:.3f}\n"
        str_repr += f"MAD: Lat. {self._mad_lat:.3f} Axial. {self._mad_ax:.3f} Vol. {self._mad_vol:.3f}\n"

        return str_repr

    def forward(self, tp: EmitterSet, tp_match: EmitterSet):
        """

        Args:
            tp (EmitterSet): true positives
            tp_match (EmitterSet): matching ground truths

        Returns:
            rmse_lat: RMSE lateral
            rmse_ax: RMSE axial
            rmse_vol: RMSE volumetric
            mad_lat: MAD lateral
            mad_ax: MAD axial
            mad_vol: MAD volumetric
        """

        rmse_vol, rmse_lat, rmse_axial, mad_vol, mad_lat, mad_axial = rmse_mad_dist(tp.xyz, tp_match.xyz)

        """Store in cache"""
        self._rmse_lat, self._rmse_ax, self._rmse_vol = rmse_lat, rmse_axial, rmse_vol
        self._mad_lat, self._mad_ax, self._mad_vol = mad_lat, mad_axial, mad_vol

        return self._dist_eval_return(rmse_lat=rmse_lat, rmse_ax=rmse_axial, rmse_vol=rmse_vol,
                                      mad_lat=mad_lat, mad_ax=mad_axial, mad_vol=mad_vol)  # namedtuple


class WeightedErrors:
    """
    Weighted deviations.
    """
    _modes_all = ('phot', 'crlb')
    _reduction_all = (None, 'mstd', 'gaussian')
    _return = namedtuple("weighted_err", ["dxyz_w", "dphot_w", "dbg_w"])
    def __init__(self, mode, reduction):

        self.mode = mode
        self.reduction = reduction

        """Sanity check"""
        if self.mode not in self._modes_all:
            raise ValueError(f"Mode {self.mode} not implemented. Available modes are {self._modes_all}")

        if self.reduction not in self._reduction_all:
            raise ValueError(f"Reduction type {self.reduction} not implemented. Available reduction types"
                             f"are {self._reduction_all}.")

    @staticmethod
    def _reduce(dxyz: torch.Tensor, dphot: torch.Tensor, dbg: torch.Tensor, reduction):
        """
        Reduce the weighted errors as by the specified method.

        Args:
            dxyz (torch.Tensor): weighted err in xyz, N x 3
            dphot (torch.Tensor): weighted err in phot, N
            dbg (torch.Tensor): weighted err in bg, N
            reduction (string,None): reduction type

        Returns:
            (torch.Tensor or tuple of tensors)

        """

        if reduction is None:
            return dxyz, dphot, dbg

        elif reduction == 'mstd':
            return (dxyz.mean(0), dxyz.std(0)), (dphot.mean(), dphot.std()), (dbg.mean(), dbg.std())

        elif reduction == 'gaussian':
            dxyz_mu_sig = torch.tensor([scipy.stats.norm.fit(dxyz[:, i]) for i in range(3)])
            dphot_mu_sig = torch.tensor(scipy.stats.norm.fit(dphot))
            dbg_mu_sig = torch.tensor(scipy.stats.norm.fit(dbg))

            return (dxyz_mu_sig[:, 0], dxyz_mu_sig[:, 1]), \
                   (dphot_mu_sig[0], dphot_mu_sig[1]), \
                   (dbg_mu_sig[0], dbg_mu_sig[1])

        else:
            raise ValueError

    @staticmethod
    def plot_error(dxyz, dphot, dbg, axes=None):
        """
        Plot the histograms

        Args:
            dxyz (torch.Tensor): weighted err in xyz, N x 3
            dphot (torch.Tensor): weighted err in phot, N
            dbg (torch.Tensor): weighted err in bg, N
            axes (tuple of axes,None): axes to which to plot to, tuple of size 6 or None

        Returns:
            axes

        """

        if axes is None:
            _, axes = plt.subplots(5)
            # axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

        else:
            if len(axes) != 5:
                raise ValueError("You must parse exactly 6 axes objects or None.")

        if len(dxyz) == 0:
            return axes

        if len(dxyz[:, 0]) != len(dphot) or len(dphot) != len(dbg):
            raise ValueError("Inconsistent number of elements.")

        sns.distplot(dxyz[:, 0].numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[0])
        sns.distplot(dxyz[:, 1].numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[1])
        sns.distplot(dxyz[:, 2].numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[2])

        sns.distplot(dphot.numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[3])
        sns.distplot(dbg, norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[4])

        return axes

    def forward(self, tp: emitter.EmitterSet, ref: emitter.EmitterSet, plot: bool = False, axes=None) -> namedtuple:
        """

        Args:
            tp (EmitterSet): true positives
            ref (EmitterSet): matching ground truth
            plot (bool): plot histograms
            axes (list,tuple): axis to which to plot the histograms

        Returns:

        """

        if len(tp) != len(ref):
            raise ValueError(f"Size of true positives ({len(tp)}) does not match size of reference ({len(ref)}).")

        if len(ref) == 0:  # if empty EmitterSets, return empty tensor
            return self._return(dxyz_w=torch.empty_like(ref.xyz),
                                dphot_w=torch.empty_like(ref.phot),
                                dbg_w=torch.empty_like(ref.bg))

        if self.mode == 'phot':
            """Definition of the 0st / 1st order approximations for the sqrt cramer rao"""
            xyz_scr_est = 1 / ref.phot.unsqueeze(1).sqrt()
            phot_scr_est = ref.phot.sqrt()
            bg_scr_est = ref.bg.sqrt()

            dxyz = (tp.xyz_nm - ref.xyz_nm) / xyz_scr_est
            dphot = (tp.phot - ref.phot) / phot_scr_est
            dbg = (tp.bg - ref.bg) / bg_scr_est

        elif self.mode == 'crlb':
            dxyz = (tp.xyz_nm - ref.xyz_nm) / ref.xyz_scr_nm
            dphot = (tp.phot - ref.phot) / ref.phot_scr
            dbg = (tp.bg - ref.bg) / ref.bg_scr

        else:
            raise ValueError

        if plot:
            _ = self.plot_error(dxyz, dphot, dbg, axes=axes)

        dxyz, dphot, dbg = self._reduce(dxyz, dphot, dbg, reduction=self.reduction)
        return self._return(dxyz_w=dxyz, dphot_w=dphot, dbg_w=dbg)


@deprecated(version="0.1.dev0", reason="Replaced by CRLB")
class Deltas:
    def __init__(self, weight='photons'):
        self.weight = weight

        assert self.weight in ('crlb_sqr', 'photons')

    def forward(self, tp: emitter.EmitterSet, ref: emitter.EmitterSet):
        """
        Calculate the dx / dy / dz values and their weighted values (weighted by the photons).
        :param tp: true positives (instance of emitterset)
        :param ref: reference (instance of emitterset)
        :return: dx, dy, dz, dx_weighted, dy_weighted, dz_weighted
        """
        dxyz = tp.xyz - ref.xyz
        if self.weight == 'photons':
            dxyz_weighted = dxyz / (ref.phot.unsqueeze(1)).sqrt()
        elif self.weight == 'crlb_sqr':
            if ref.xy_unit == 'nm':
                dxyz_weighted = dxyz / ref.xyz_scr_nm
            elif ref.xy_unit == 'px':
                dxyz_weighted = dxyz / ref.xyz_scr
        else:
            raise ValueError("Not supported mode.")

        return dxyz[:, 0], dxyz[:, 1], dxyz[:, 2], dxyz_weighted[:, 0], dxyz_weighted[:, 1], dxyz_weighted[:, 2]