import math
import random

import numpy as np
import seaborn as sns
import torch

from deprecated import deprecated
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.stats import gaussian_kde

import deepsmlm.evaluation.evaluation
from deepsmlm.evaluation.evaluation import EvalSet


class MetricMeter:
    """Computes and stores the average and current value"""

    def __init__(self, vals=torch.zeros((0,)), reduce_nan=True):
        self.val = None
        self.vals = vals
        self.reduce_nan = reduce_nan

    @property
    def count(self):
        return self.vals.numel()

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

    def hist(self, bins=50, range_hist=None, fit=stats.norm):
        # If no vals, return empty figure
        if self.vals.numel() <= 10:
            f = plt.figure()
            return f
        elif self.vals.unique().numel() == 1:
            f = plt.figure()
            plt.vlines(self.vals[0], 0, 1)
            return f

        if range_hist is not None:
            bins_ = np.linspace(range_hist[0], range_hist[1], bins + 1)
        else:
            # use 97 percent as range
            range_hist = np.percentile(self.vals, [1, 99])  # funnily percentile is in percent not 0...1
            bins_ = np.linspace(*range_hist, bins + 1)

        vals = self.vals.view(-1).numpy()
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.1, .9)})

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
        self.vals = torch.zeros((0,))

    def update(self, val):
        """
        Update AverageMeter.

        :param val: value
        :return: None
        """

        val = float(val)
        if math.isnan(val) and self.reduce_nan:
            return

        # convert to torch.tensor
        self.val = val
        self.vals = torch.cat((self.vals, torch.Tensor([val])), 0)

    def __str__(self):
        if self.count >= 2:
            return "{:.3f} +/- {:.3f}".format(self.avg, self.std)
        else:
            return "{:.3f}".format(self.avg)

    def __neg__(self):
        m = MetricMeter(reduce_nan=self.reduce_nan)
        m.vals = -self.vals
        return m

    def __add__(self, other):
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            m.vals = self.vals + other.vals
        else:
            m.vals = self.vals + other
        return m

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return -other + self

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            m.vals = self.vals * other.vals
        else:
            m.vals = self.vals * other
        return m

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            raise ValueError("Power not implemented for both operands being MetricMeters.")
        else:
            m.vals = self.vals ** other
        return m

    def __truediv__(self, other):
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            m.vals = self.vals / other.vals
        else:
            m.vals = self.vals / other
        return m


@deprecated("I don't remember what this thing does.")
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


class BatchEvaluation:

    def __init__(self, matching, segmentation_eval, distance_eval, batch_size, weight='crlb_sqr', px_size=None):
        import deepsmlm.evaluation.metric_library as metric_lib
        self._matching = matching
        self._segmentation_eval = segmentation_eval
        self._distance_eval = distance_eval
        self._batch_size = batch_size
        self._distance_delta = deepsmlm.evaluation.evaluation.Deltas(weight=weight)

        self.values = None
        self.px_size = px_size

        # True positives, matched ground truth (tp_match), false positives, false negatives
        self.tp = None
        self.tp_match = None
        self.fp = None
        self.fn = None

        if self.px_size is not None:
            if not isinstance(self.px_size, torch.Tensor):
                self.px_size = torch.Tensor(self.px_size)

    def forward(self, output, target):
        """
        Evaluate metrics on a whole batch, i.e. average the stuff. This is based lists of outputs, targets.

        :param output: list of emitter sets
        :param target: list of emitter sets
        :return:
        """
        prec, rec, jac, f1 = MetricMeter(), MetricMeter(), MetricMeter(), MetricMeter()
        delta_num = MetricMeter()
        rmse_vol, rmse_lat, rmse_axial = MetricMeter(), MetricMeter(), MetricMeter()
        mad_vol, mad_lat, mad_axial = MetricMeter(), MetricMeter(), MetricMeter()
        dx, dy, dz, dxw, dyw, dzw = CumulantMeter(), CumulantMeter(), CumulantMeter(), \
                                    CumulantMeter(), CumulantMeter(), CumulantMeter()

        # Check that both output and target have a unit if we actually have emitters
        if len(output) >= 1:
            if output.xy_unit is None or output.xy_unit is None:
                raise ValueError("For Evaluation both output and target must have units.")

        if output.xy_unit != 'nm':
            output = output.convert_em(factor=self.px_size, new_xy_unit='nm')
        if target.xy_unit != 'nm':
            target = target.convert_em(factor=self.px_size, new_xy_unit='nm')

        tp, fp, fn, tp_match = self._matching.forward(output, target)
        prec_, rec_, jaq_, f1_ = self._segmentation_eval.forward(tp, fp, fn)
        rmse_vol_, rmse_lat_, rmse_axial_, mad_vol_, mad_lat_, mad_axial_ = self._distance_eval.forward(tp, tp_match)
        dx_, dy_, dz_, dxw_, dyw_, dzw_ = self._distance_delta.forward(tp, tp_match)

        prec.update(prec_)
        rec.update(rec_)
        jac.update(jaq_)
        f1.update(f1_)

        delta_num.update(len(output) - len(target))

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

        self.values = EvalSet(prec, rec, jac, f1,
                              delta_num,
                              rmse_vol, rmse_lat, rmse_axial,
                              mad_vol, mad_lat, mad_axial,
                              dx, dy, dz, dxw, dyw, dzw)

        self.tp = tp.clone()
        self.tp_match = tp_match.clone()
        self.fp = fp.clone()
        self.fn = fn.clone()

    def plot_sophisticated(self):
        """
        Plots more interesting stuff.
        """

        # plot errors
        self.values.dx.hist()
        plt.xlabel('$x_{pred} - x_{gt}$')
        plt.show()

        self.values.dy.hist()
        plt.xlabel('$y_{pred} - y_{gt}$')
        plt.show()

        self.values.dz.hist()
        plt.xlabel('$z_{pred} - z_{gt}$')
        plt.show()

        # plot distances to px centre
        offsets_x = self.tp.xyz_px[:, 0] - self.tp.xyz_px[:, 0].round()
        plt.hist(offsets_x, 100)
        plt.xlabel('dx to px centre')
        plt.show()

        offsets_y = self.tp.xyz_px[:, 1] - self.tp.xyz_px[:, 1].round()
        plt.hist(offsets_y, 100)
        plt.xlabel('dy to px centre')
        plt.show()

        pm = MetricMeter()
        pm.vals = self.tp.prob
        pm.hist(bins=100, range_hist=[0., 1.], fit=None)
        plt.title('Prob. Distribution of Prediction')
        plt.xlabel('p')
        plt.show()

        phm = MetricMeter()
        phm.vals = self.tp.phot
        phm.hist(bins=200, fit=None)
        plt.title('Photon Distribution of Prediction')
        plt.xlabel('phot')
        plt.show()

        # plot z over z_gt
        plt.figure(figsize=(12, 8))
        z, x, y = kde_sorted(self.tp_match.xyz[:, 2], self.tp.xyz[:, 2], True)
        plt.plot(x, x, 'r', label='unbiased estimate')
        plt.legend()
        plt.xlabel('$z_{gt} [nm]$')
        plt.ylabel('$z_{pred} [nm]$')
        plt.title('Distribution of z prediction vs ground truth')
        plt.show()

        # plot phot over phot_gt
        plt.figure(figsize=(12, 8))
        z, x, y = kde_sorted(self.tp_match.phot, self.tp.phot, True)
        plt.plot(x, x, 'r', label='unbiased estimate')
        plt.legend()
        plt.xlabel('$phot_{gt} [nm]$')
        plt.ylabel('$phot_{pred} [nm]$')
        plt.title('Distribution of phot prediction vs ground truth')
        plt.show()

    @staticmethod
    def plot_sophisticated_raws(raw_output, frame_subset=1000, p_eps=0.05, sim_extent_z=None):
        """
        Plots some fancy statistics of the raw outputs. Relatively costly
        :param: raw_output should be scaled!
        """
        if frame_subset is not None:
            f_all = np.arange(raw_output.size(0))
            frame_subset = min([frame_subset, raw_output.size(0)])
            f_subset = random.sample(list(f_all), k=frame_subset)
        else:
            f_subset = slice(0, raw_output.size(0))

        ix = raw_output[f_subset, 0] >= p_eps
        p_values = raw_output[f_subset, 0][ix].reshape(-1)
        phot_vals = raw_output[f_subset, 1][ix].reshape(-1)
        dx_vals = raw_output[f_subset, 2][ix].reshape(-1)
        dy_vals = raw_output[f_subset, 3][ix].reshape(-1)
        dz_vals = raw_output[f_subset, 4][ix].reshape(-1)

        """MultiProcessing since this KDE thing can take quite some time."""
        dens_x, pxs, dxs = kde_sorted(p_values, dx_vals)
        dens_y, pys, dys = kde_sorted(p_values, dy_vals)
        dens_z, pzs, dzs = kde_sorted(p_values, dz_vals)

        dens_xphot, pxsp, dxsp = kde_sorted(phot_vals, dx_vals)
        dens_yphot, pysp, dysp = kde_sorted(phot_vals, dy_vals)
        dens_zphot, pzsp, dzsp = kde_sorted(phot_vals, dz_vals)

        plt.figure(figsize=(24, 12))
        plt.subplot(131)
        plt.scatter(pxs, dxs, c=dens_x, s=10, edgecolor='', cmap='RdBu_r')
        plt.hlines([-0.5, 0.5], 0, 1, 'r', label='px border')
        plt.xlabel('$p$')
        plt.ylabel('$dx$')
        plt.legend()
        plt.title('Distribution of dx offset vs probability')

        plt.subplot(132)
        plt.scatter(pys, dys, c=dens_y, s=10, edgecolor='', cmap='RdBu_r')
        plt.hlines([-0.5, 0.5], 0, 1, 'r', label='px border')
        plt.xlabel('$p$')
        plt.ylabel('$dy$')
        plt.legend()
        plt.title('Distribution of dy offset vs probability')

        plt.subplot(133)
        plt.scatter(pzs, dzs, c=dens_z, s=10, edgecolor='', cmap='RdBu_r')
        if sim_extent_z is not None:
            plt.hlines(sim_extent_z, 0, 1, 'r', label='simulation extent')
        plt.xlabel('$p$')
        plt.ylabel('$z [nm]$')
        plt.title('Distribution of z offset vs probability')
        plt.show()

        plt.figure(figsize=(24, 12))
        plt.subplot(131)
        plt.scatter(pxsp, dxsp, c=dens_xphot, s=10, edgecolor='', cmap='RdBu_r')
        plt.hlines([-0.5, 0.5], 0, phot_vals.max().item(), 'r', label='px border')
        plt.xlabel('$phot$')
        plt.ylabel('$dx$')
        plt.legend()
        plt.title('Distribution of dx offset vs photons')

        plt.subplot(132)
        plt.scatter(pysp, dysp, c=dens_yphot, s=10, edgecolor='', cmap='RdBu_r')
        plt.hlines([-0.5, 0.5], 0, phot_vals.max().item(), 'r', label='px border')
        plt.xlabel('$phot$')
        plt.ylabel('$dy$')
        plt.legend()
        plt.title('Distribution of dy offset vs photons')

        plt.subplot(133)
        plt.scatter(pzsp, dzsp, c=dens_zphot, s=10, edgecolor='', cmap='RdBu_r')
        if sim_extent_z is not None:
            plt.hlines(sim_extent_z, 0, 1, 'r', label='simulation extent')
        plt.xlabel('$phot$')
        plt.ylabel('$z [nm]$')
        plt.title('Distribution of z offset vs photons')
        plt.show()


def kde_sorted(x, y, plot=False, band_with=None):
    """
    Gives a density estimates useful for plotting many datapoints
    """
    xy = np.vstack([x, y])
    if band_with:
        z = gaussian_kde(xy, bw_method=band_with)(xy)
    else:
        z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    if plot:
        plt.scatter(x, y, c=z, s=10, edgecolor='', cmap='RdBu_r')
    return z, x, y