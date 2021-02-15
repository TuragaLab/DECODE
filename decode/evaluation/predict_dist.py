import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings

from . import utils


def deviation_dist(x: torch.Tensor, x_gt: torch.Tensor, residuals=False, kde=True, ax=None, nan_okay=True):
    """Log z vs z_gt"""
    if ax is None:
        ax = plt.gca()

    if len(x) == 0:
        ax.set_ylabel('no data')
        return ax

    if residuals:
        x = x - x_gt

    if not torch.isnan(x).any():
        if kde:
            utils.kde_sorted(x_gt, x, True, ax, sub_sample=10000, nan_inf_ignore=True)
        else:
            ax.plot(x_gt, x, 'x')

    else:
        if not nan_okay:
            raise ValueError(f"Some of the values are NaN.")

    if residuals:
        ax.plot([x_gt.min(), x_gt.max()], [0, 0], 'green')
        ax.set_ylabel('residuals')

    else:
        ax.plot([x_gt.min(), x_gt.max()], [x_gt.min(), x_gt.max()], 'green')
        ax.set_ylabel('prediction')

    ax.set_xlabel('ground truth')
    return ax


def px_pointer_dist(pointer, px_border: float, px_size: float):
    """

    Args:
        pointer:
        px_border: lower limit of pixel (most commonly -0.5)
        px_size: size of pixel (most commonly 1.)

    Returns:

    """
    x = (pointer - px_border) % px_size + px_border
    return x


def emitter_deviations(tp, tp_match, px_border: float, px_size: float, axes, residuals=False, kde=True):
    """Plot within px distribution"""
    assert len(axes) == 4

    """XY within px"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.distplot(px_pointer_dist(tp.xyz_px[:, 0], px_border=px_border, px_size=px_size), norm_hist=True, ax=axes[0], bins=50)
        sns.distplot(px_pointer_dist(tp.xyz_px[:, 1], px_border=px_border, px_size=px_size), norm_hist=True, ax=axes[1], bins=50)


    """Z and Photons"""
    deviation_dist(tp.xyz_nm[:, 2], tp_match.xyz_nm[:, 2], residuals=residuals, kde=kde, ax=axes[2])
    deviation_dist(tp.phot, tp_match.phot, residuals=residuals, kde=kde, ax=axes[3])
