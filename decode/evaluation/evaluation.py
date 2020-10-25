# from abc import ABC
import warnings
from collections import namedtuple

import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import torch

from decode.evaluation.metric import precision_recall_jaccard, rmse_mad_dist, efficiency
from ..generic import emitter as emitter
from ..generic.emitter import EmitterSet


class SegmentationEvaluation:
    """
    Wrapper class that holds all segementation evaluations in one place.

    """
    _seg_eval_return = namedtuple("seg_eval", ["prec", "rec", "jac", "f1"])

    def __init__(self):
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
        Forward emitters through evaluation.

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives

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
    A small wrapper calss that holds distance evaluations and accepts sets of emitters as inputs.
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
            tp: true positives
            tp_match: matching ground truths

        Returns:
            rmse_lat: RMSE lateral
            rmse_ax: RMSE axial
            rmse_vol: RMSE volumetric
            mad_lat: MAD lateral
            mad_ax: MAD axial
            mad_vol: MAD volumetric

        """

        rmse_lat, rmse_axial, rmse_vol, mad_lat, mad_axial, mad_vol = rmse_mad_dist(tp.xyz_nm, tp_match.xyz_nm)

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
    _reduction_all = ('mstd', 'gaussian')
    _return = namedtuple("weighted_err", ["dxyz_red", "dphot_red", "dbg_red", "dxyz_w", "dphot_w", "dbg_w"])

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

        def norm_fit_nan(input_data, warning=True):
            try:
                out = scipy.stats.norm.fit(input_data)
                out = torch.tensor(out)

            except RuntimeError:
                warnings.warn("Non-Finite values encountered during fitting.")
                out = float('nan') * torch.ones(2)

            return out

        if reduction == 'mstd':
            return (dxyz.mean(0), dxyz.std(0)), (dphot.mean(), dphot.std()), (dbg.mean(), dbg.std())

        elif reduction == 'gaussian':

            dxyz_mu_sig = torch.stack([norm_fit_nan(dxyz[:, i]) for i in range(3)], 0)
            dphot_mu_sig = norm_fit_nan(dphot)
            dbg_mu_sig = norm_fit_nan(dbg)

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
        sns.distplot(dbg.numpy(), norm_hist=True, kde=False, fit=scipy.stats.norm, ax=axes[4])

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

        dxyz = tp.xyz_nm - ref.xyz_nm
        dphot = tp.phot - ref.phot
        dbg = tp.bg - ref.bg

        if self.mode == 'phot':
            """Definition of the 0st / 1st order approximations for the sqrt cramer rao"""
            xyz_scr_est = 1 / ref.phot.unsqueeze(1).sqrt()
            phot_scr_est = ref.phot.sqrt()
            bg_scr_est = ref.bg.sqrt()

            dxyz_w = dxyz / xyz_scr_est
            dphot_w = dphot / phot_scr_est
            dbg_w = dbg / bg_scr_est

        elif self.mode == 'crlb':
            dxyz_w = dxyz / ref.xyz_scr_nm
            dphot_w = dphot / ref.phot_scr
            dbg_w = dbg / ref.bg_scr

        else:
            raise ValueError

        if plot:
            _ = self.plot_error(dxyz_w, dphot_w, dbg_w, axes=axes)

        dxyz_wred, dphot_wred, dbg_wred = self._reduce(dxyz_w, dphot_w, dbg_w, reduction=self.reduction)
        return self._return(dxyz_red=dxyz_wred, dphot_red=dphot_wred, dbg_red=dbg_wred,
                            dxyz_w=dxyz_w, dphot_w=dphot_w, dbg_w=dbg_w)


class SMLMEvaluation:
    """
    Just a wrapper class to combine things into one.
    """
    alpha_lat = 1  # nm
    alpha_ax = 0.5  # nm

    _return = namedtuple("eval_set", ["prec", "rec", "jac", "f1", "effcy_lat", "effcy_ax", "effcy_vol",
                                      "rmse_lat", "rmse_ax", "rmse_vol", "mad_lat", "mad_ax", "mad_vol",
                                      "dx_red_mu", "dx_red_sig", "dy_red_mu", "dy_red_sig", "dz_red_mu", "dz_red_sig",
                                      "dphot_red_mu", "dphot_red_sig"])

    descriptors = {
        'pred': 'Precision',
        'rec': 'Recall',
        'jac': 'Jaccard Index',
        'rmse_lat': 'RMSE lateral',
        'rmse_ax': 'RMSE axial',
        'rmse_vol': 'RMSE volumetric',
        'mad_lat': 'Mean average distance lateral',
        'mad_ax': 'Mean average distance axial',
        'mad_vol': 'Mean average distance in 3D',
        'dx_red_sig': 'CRLB normalised error in x',
        'dy_red_sig': 'CRLB normalised error in y',
        'dz_red_sig': 'CRLB normalised error in z',
        'dx_red_mu': 'CRLB normalised bias in x',
        'dy_red_mu': 'CRLB normalised bias in y',
        'dz_red_mu': 'CRLB normalised bias in z',
    }

    def __init__(self, seg_eval=SegmentationEvaluation(),
                 dist_eval=DistanceEvaluation(),
                 weighted_eval=WeightedErrors(mode='crlb', reduction='gaussian')):
        self.seg_eval = seg_eval
        self.dist_eval = dist_eval
        self.weighted_eval = weighted_eval

        self.prec = None
        self.rec = None
        self.jac = None
        self.f1 = None

        self.rmse_vol = None
        self.rmse_lat = None
        self.rmse_ax = None
        self.mad_vol = None
        self.mad_lat = None
        self.mad_ax = None

    @property
    def effcy_lat(self):
        return efficiency(self.jac, self.rmse_lat, self.alpha_lat)

    @property
    def effcy_ax(self):
        return efficiency(self.jac, self.rmse_ax, self.alpha_ax)

    @property
    def effcy_vol(self):
        return (self.effcy_lat + self.effcy_ax) / 2

    def __str__(self):
        str = "------------------------ Evaluation Set ------------------------\n"
        str += "Precision {}\n".format(self.prec.__str__())
        str += "Recall {}\n".format(self.rec.__str__())
        str += "Jaccard {}\n".format(self.jac.__str__())
        str += "F1Score {}\n".format(self.f1.__str__())
        str += "RMSE lat. {}\n".format(self.rmse_lat.__str__())
        str += "RMSE ax. {}\n".format(self.rmse_axial.__str__())
        str += "RMSE vol. {}\n".format(self.rmse_vol.__str__())
        str += "MAD lat. {}\n".format(self.mad_lat.__str__())
        str += "MAD ax. {}\n".format(self.mad_axial.__str__())
        str += "MAD vol. {}\n".format(self.mad_vol.__str__())
        str += "Efficiency lat. {}\n".format(self.effcy_lat.__str__())
        str += "Efficiency ax. {}\n".format(self.effcy_ax.__str__())
        str += "Efficiency vol. {}\n".format(self.effcy_vol.__str__())
        str += "-----------------------------------------------------------------"
        return str

    def forward(self, tp, fp, fn, p_ref) -> _return:
        """
        Evaluate sets of emitters by all available metrics.

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
            p_ref: true positive references (i.e. the ground truth that has been matched to tp)

        Returns:
            namedtuple: A namedtuple of floats containing

                - **prec** (*float*): Precision
                - **rec** (*float*): Recall
                - **jac** (*float*): Jaccard
                - **f1** (*float*): F1-Score
                - **effcy_lat** (*float*): Efficiency lateral
                - **effcy_ax** (*float*): Efficiency axial
                - **effcy_vol** (*float*): Efficiency volumetric
                - **rmse_lat** (*float*): RMSE lateral
                - **rmse_ax** (*float*): RMSE axial
                - **rmse_vol** (*float*): RMSE volumetric
                - **mad_lat** (*float*): MAD lateral
                - **mad_ax** (*float*): MAD axial
                - **mad_vol** (*float*): MAD volumetric


        """
        seg_out = self.seg_eval.forward(tp, fp, fn)
        dist_out = self.dist_eval.forward(tp, p_ref)
        weight_out = self.weighted_eval.forward(tp, p_ref, plot=False)

        self.prec, self.rec, self.jac, self.f1 = seg_out.prec, seg_out.rec, seg_out.jac, seg_out.f1

        self.rmse_lat = dist_out.rmse_lat
        self.rmse_ax = dist_out.rmse_ax
        self.rmse_vol = dist_out.rmse_vol

        self.mad_lat = dist_out.mad_lat
        self.mad_ax = dist_out.mad_ax
        self.mad_vol = dist_out.mad_vol

        dx_red = (weight_out.dxyz_red[0][0].item(), weight_out.dxyz_red[1][0].item())
        dy_red = (weight_out.dxyz_red[0][1].item(), weight_out.dxyz_red[1][1].item())
        dz_red = (weight_out.dxyz_red[0][2].item(), weight_out.dxyz_red[1][2].item())

        return self._return(prec=seg_out.prec, rec=seg_out.rec, jac=seg_out.jac, f1=seg_out.f1,
                            effcy_lat=self.effcy_lat, effcy_ax=self.effcy_ax, effcy_vol=self.effcy_vol,
                            rmse_lat=dist_out.rmse_lat, rmse_ax=dist_out.rmse_ax, rmse_vol=dist_out.rmse_vol,
                            mad_lat=dist_out.mad_lat, mad_ax=dist_out.mad_ax, mad_vol=dist_out.mad_vol,
                            dx_red_mu=dx_red[0], dx_red_sig=dx_red[1],
                            dy_red_mu=dy_red[0], dy_red_sig=dy_red[1],
                            dz_red_mu=dz_red[0], dz_red_sig=dz_red[1],
                            dphot_red_mu=weight_out.dphot_red[0].item(), dphot_red_sig=weight_out.dphot_red[1].item())
