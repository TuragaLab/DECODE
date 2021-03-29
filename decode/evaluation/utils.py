import math

import numpy as np
import seaborn as sns
import torch

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.stats import gaussian_kde
import warnings


def kde_sorted(x: torch.Tensor, y: torch.Tensor, plot=False, ax=None, band_with=None, sub_sample: (None, int) = None,
               nan_inf_ignore=False):
    """
    Computes a kernel density estimate. Ability to sub-sample for very large datasets.

    Args:
        x:
        y:
        plot:
        ax:
        band_with:
        sub_sample:
        nan_inf_ignore:

    """

    if sub_sample:
        ix_sample = np.random.permutation(x.size(0))
        ix_in, ix_out = ix_sample[:sub_sample], ix_sample[sub_sample:]

    else:
        ix_in = np.arange(x.size(0))
        ix_out = np.zeros(0, dtype=int)

    xy = np.vstack([x, y])
    xy_in = xy[:, ix_in]
    xy_out = xy[:, ix_out]

    if xy_in.size == 0:
        z = np.zeros(0)

    elif nan_inf_ignore:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if band_with:
                    z = gaussian_kde(xy_in, bw_method=band_with)(xy_in)
                else:
                    z = gaussian_kde(xy_in)(xy_in)

        except (np.linalg.LinAlgError, ValueError):  # ToDo: replace by robust kde
            z = np.ones_like(xy_in[0]) * float('nan')

    else:
        if band_with:
            z = gaussian_kde(xy_in, bw_method=band_with)(xy_in)
        else:
            z = gaussian_kde(xy_in)(xy_in)

    # Sort the points by density, so that the most dense points are plotted last
    idx = z.argsort()
    x_in, y_in, z = xy_in[0, idx], xy_in[1, idx], z[idx]

    if plot:

        if ax is None:
            ax = plt.gca()

        if sub_sample:
            # get rid of nan
            xy_out = xy_out[:, np.prod((~np.isnan(xy_out)), 0).astype('bool')]
            ax.scatter(xy_out[0, :], xy_out[1, :], c='k', s=10, edgecolors='none')

        not_nan = (~np.isnan(x_in) * ~np.isnan(y_in))
        ax.scatter(x_in[not_nan], y_in[not_nan], c=z[None, not_nan], s=10, edgecolors='none', cmap='RdBu_r')

    return z, x_in, y_in


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

        """Plot boxplot and histplot."""
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
