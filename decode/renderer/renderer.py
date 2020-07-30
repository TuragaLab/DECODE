from abc import ABC
import torch
from ..generic import emitter
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter


class Renderer(ABC):
    """
    Renderer. Takes emitters and outputs a rendered image.

    """

    def __init__(self, xextent: tuple, yextent: tuple, px_size: float):
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.px_size = px_size

    @property
    def _npx_x(self):
        return math.ceil(se)

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:
        """
        Forward emitterset through rendering and output rendered data.

        Args:
            em: emitter set

        """
        raise NotImplementedError

    def render(self, em: emitter.EmitterSet, ax=None):
        """
        Render emitters

        Args:
            em: emitter set
            ax: plot axis

        Returns:

        """
        raise NotImplementedError


class Renderer2D(Renderer):
    """
    2D Renderer with constant gaussian.

    """

    def __init__(self, px_size, sigma_blur, xextent=None, yextent=None, clip_percentile=None):
        super().__init__(xextent=xextent, yextent=yextent, px_size=px_size)

        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile

    def render(self, em, ax=None, cmap: str = 'gray'):

        hist = self.forward(em).numpy()

        if ax is None:
            ax = plt.gca()

        ax = ax.imshow(np.transpose(hist), cmap=cmap)  # because imshow use different ordering
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        if self.xextent is None:
            self.xextent = (em.xyz_nm[:, 0].min(), em.xyz_nm[:, 0].max())
        if self.yextent is None:
            self.yextent = (em.xyz_nm[:, 1].min(), em.xyz_nm[:, 1].max())

        hist = self._hist2d(em.xyz_nm[:, :2].numpy(), xextent=self.xextent, yextent=self.yextent, px_size=self.px_size)

        if self.clip_percentile is not None:
            hist = np.clip(hist, 0., np.percentile(hist, self.clip_percentile))

        if self.sigma_blur is not None:
            hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])

        return torch.from_numpy(hist)

    @staticmethod
    def _hist2d(xy: np.array, xextent, yextent, px_size) -> np.array:

        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        return hist


# ToDo: Not yet ready
class RenderHist3D:

    def __init__(self, size, z_range, pixel_size, sigma_blur, clip_percentile, gamma):
        self.size = size
        self.z_range = z_range
        self.pixel_size = pixel_size
        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile
        self.gamma = gamma

    def plot(self, xyz_nm, figsize=(10, 10), fontsize=(15)):
        hist, z_hist = get_2d_hist(xyz_nm, self.size, self.pixel_size, self.z_range)
        with np.errstate(divide='ignore', invalid='ignore'):
            z_avg = z_hist / hist

        hist = np.clip(hist, 0, np.percentile(hist, self.clip_percentile))
        z_avg[np.isnan(z_avg)] = 0

        val = (hist - hist.min()) / (hist.max() - hist.min())
        sat = np.ones(hist.shape)
        hue = z_avg

        HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
        RGB = hsv_to_rgb(HSV) ** (1 / self.gamma)

        if self.sigma_blur:
            RGB = np.array([gaussian_filter(RGB[:, :, i], sigma=[self.sigma_blur / self.pixel_size,
                                                                 self.sigma_blur / self.pixel_size]) for i in
                            range(3)]).transpose(1, 2, 0)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(RGB, cmap='hsv')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.25, pad=-0.25)
        colb = plt.colorbar(im, cax=cax, orientation='vertical', ticks=[])
        colb.outline.set_visible(False)

        cax.text(0.12, 0.04, f'{self.z_range[0]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.text(0.12, 0.88, f'{self.z_range[1]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)


def get_2d_hist(xyz_nm, size=None, pixel_size=10, z_range=None):
    xyz_pos = np.array(xyz_nm)
    x_pos = xyz_pos[:, 0]
    y_pos = xyz_pos[:, 1]
    z_pos = xyz_pos[:, 2]

    if z_range is None:
        z_range = [z_pos.min(), z_pos.max()]

    z_pos = np.clip(z_pos, z_range[0], z_range[1])
    z_weight = ((z_pos - z_pos.min()) / (z_pos.max() - z_pos.min()))

    if size is None:
        hist_dim = int(x_pos.max() // pixel_size), int(y_pos.max() // pixel_size)
    else:
        hist_dim = int(size[0] // pixel_size), int(size[1] // pixel_size)

    hist = \
    np.histogram2d(x_pos, y_pos, bins=hist_dim, range=[[0, hist_dim[0] * pixel_size], [0, hist_dim[1] * pixel_size]])[0]
    z_hist = \
    np.histogram2d(x_pos, y_pos, bins=hist_dim, range=[[0, hist_dim[0] * pixel_size], [0, hist_dim[1] * pixel_size]],
                   weights=z_weight)[0]
    return hist, z_hist